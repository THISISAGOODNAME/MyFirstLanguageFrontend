#include "Warning.h"

#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

#include <iostream>
#include <vector>
#include <map>

#pragma region Lexer
//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

namespace Lexer {
    // The lexer returns tokens [0-255] if it is an unknown character, otherwise one
    // of these for known things.
    enum Token {
        tok_eof = -1,

        // commands
        tok_def = -2,
        tok_extern = -3,

        // primary
        tok_identifier = -4,
        tok_number = -5,
    };

    static std::string IdentifierStr; // Filled in if tok_identifier
    static double NumVal;             // Filled in if tok_number

    static int gettok() {
        static int LastChar = ' ';

        // Skip any whitespace.
        while (isspace(LastChar))
            LastChar = getchar();

        if (isalpha(LastChar)) {
            // identifier: [a-zA-Z][a-zA-Z0-9]*
            IdentifierStr = LastChar;
            while (isalnum((LastChar = getchar())))
                IdentifierStr += LastChar;

            if (IdentifierStr == "def")
                return tok_def;

            if (IdentifierStr == "extern")
                return tok_extern;

            return tok_identifier;
        }

        if (isdigit(LastChar) || LastChar == '.') {
            // Number: [0-9.]+
            std::string NumStr;
            do {
                NumStr += LastChar;
                LastChar = getchar();
            } while (isdigit(LastChar) || LastChar == '.');

            NumVal = strtod(NumStr.c_str(), 0);
            return tok_number;
        }

        if (LastChar == '#') {
            // Comment until end of line.
            do
                LastChar = getchar();
            while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

            if (LastChar != EOF)
                return gettok();
        }

        // Check for end of file.  Don't eat the EOF.
        if (LastChar == EOF)
            return tok_eof;

        // Otherwise, just return the character as its ascii value.
        int ThisChar = LastChar;
        LastChar = getchar();
        return ThisChar;
    }

}
#pragma endregion Lexer

#pragma region AST
//===----------------------------------------------------------------------===//
// Abstract Syntax Tree (aka Parse Tree)
//===----------------------------------------------------------------------===//

namespace AST
{
    /// ExprAST - Base class for all expression nodes.
    class ExprAST
    {
    public:
        virtual ~ExprAST() = default;

        virtual llvm::Value *codegen() = 0;
    };

    /// NumberExprAST - Expression class for numeric literals like "1.0".
    class NumberExprAST : public ExprAST
    {
        double Val;

    public:
        NumberExprAST(double Val) : Val(Val) {}

        llvm::Value *codegen() override;
    };

    /// VariableExprAST - Expression class for referencing a variable, like "a".
    class VariableExprAST : public ExprAST
    {
        std::string Name;

    public:
        VariableExprAST(const std::string &Name) : Name(Name) {}

        llvm::Value *codegen() override;
    };

    /// BinaryExprAST - Expression class for a binary operator.
    class BinaryExprAST : public ExprAST
    {
        char Op;
        std::unique_ptr<ExprAST> LHS, RHS;

    public:
        BinaryExprAST(char op, std::unique_ptr<ExprAST> LHS,
                      std::unique_ptr<ExprAST> RHS)
                : Op(op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}

        llvm::Value *codegen() override;
    };

    /// CallExprAST - Expression class for function calls.
    class CallExprAST : public ExprAST
    {
        std::string Callee;
        std::vector<std::unique_ptr<ExprAST>> Args;

    public:
        CallExprAST(const std::string &Callee,
                    std::vector<std::unique_ptr<ExprAST>> Args)
                    : Callee(Callee)
                    , Args(std::move(Args)) {}

        llvm::Value *codegen() override;
    };

    /// PrototypeAST - This class represents the "prototype" for a function,
    /// which captures its name, and its argument names (thus implicitly the number
    /// of arguments the function takes).
    class PrototypeAST
    {
        std::string Name;
        std::vector<std::string> Args;

    public:
        PrototypeAST(const std::string &name, std::vector<std::string> Args)
                : Name(name), Args(std::move(Args)) {}

        const std::string &getName() const { return Name; }

        llvm::Function *codegen();
    };

    /// FunctionAST - This class represents a function definition itself.
    class FunctionAST
    {
        std::unique_ptr<PrototypeAST> Proto;
        std::unique_ptr<ExprAST> Body;

    public:
        FunctionAST(std::unique_ptr<PrototypeAST> Proto,
                    std::unique_ptr<ExprAST> Body)
                : Proto(std::move(Proto)), Body(std::move(Body)) {}

        llvm::Function *codegen();
    };


}

#pragma endregion AST

#pragma region PARSER
//===----------------------------------------------------------------------===//
// Parser
//===----------------------------------------------------------------------===//

namespace Parser
{
    /// CurTok/getNextToken - Provide a simple token buffer.  CurTok is the current
    /// token the parser is looking at.  getNextToken reads another token from the
    /// lexer and updates CurTok with its results.
    static int CurTok;
    static int getNextToken() { return CurTok = Lexer::gettok(); }

    /// BinopPrecedence - This holds the precedence for each binary operator that is
    /// defined.
    static std::map<char, int> BinopPrecedence;

    /// GetTokPrecedence - Get the precedence of the pending binary operator token.
    static int GetTokPrecedence()
    {
        if (!isascii(CurTok))
            return -1;

        // Make sure it's a declared binop.
        int TokPrec = BinopPrecedence[CurTok];
        if (TokPrec <= 0) return -1;
        return TokPrec;
    }

    /// LogError* - These are little helper functions for error handling.
    std::unique_ptr<AST::ExprAST> LogError(const char *Str)
    {
        fprintf(stderr, "LogError: %s\n", Str);
        return nullptr;
    }

    std::unique_ptr<AST::PrototypeAST> LogErrorP(const char *Str)
    {
        LogError(Str);
        return nullptr;
    }

    static std::unique_ptr<AST::ExprAST> ParseExpression();

    /// numberexpr ::= number
    static std::unique_ptr<AST::ExprAST> ParseNumberExpr()
    {
        auto Result = std::make_unique<AST::NumberExprAST>(Lexer::NumVal);
        getNextToken(); // consume the number
        return std::move(Result);
    }

    /// parenexpr ::= '(' expression ')'
    static std::unique_ptr<AST::ExprAST> ParseParenExpr()
    {
        getNextToken(); // eat (.
        auto V = ParseExpression();
        if (!V)
            return nullptr;

        if (CurTok != ')')
            return LogError("expected ')'");
        getNextToken(); // eat ).
        return V;
    }

    /// identifierexpr
    ///   ::= identifier
    ///   ::= identifier '(' expression* ')'
    static std::unique_ptr<AST::ExprAST> ParseIdentifierExpr()
    {
        std::string IdName = Lexer::IdentifierStr;

        getNextToken();  // eat identifier.

        if (CurTok != '(') // Simple variable ref.
            return std::make_unique<AST::VariableExprAST>(IdName);

        // Call.
        getNextToken();  // eat (
        std::vector<std::unique_ptr<AST::ExprAST>> Args;
        if (CurTok != ')') {
            while (1) {
                if (auto Arg = ParseExpression())
                    Args.push_back(std::move(Arg));
                else
                    return nullptr;

                if (CurTok == ')')
                    break;

                if (CurTok != ',')
                    return LogError("Expected ')' or ',' in argument list");
                getNextToken();
            }
        }

        // Eat the ')'.
        getNextToken();

        return std::make_unique<AST::CallExprAST>(IdName, std::move(Args));
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    static std::unique_ptr<AST::ExprAST> ParsePrimary()
    {
        switch (CurTok) {
            default:
                return LogError("unknown token when expecting an expression");
            case Lexer::tok_identifier:
                return ParseIdentifierExpr();
            case Lexer::tok_number:
                return ParseNumberExpr();
            case '(':
                return ParseParenExpr();
        }
    }

    /// binoprhs
    ///   ::= ('+' primary)*
    static std::unique_ptr<AST::ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<AST::ExprAST> LHS)
    {
        // If this is a binop, find its precedence.
        while (true)
        {
            int TokPrec = GetTokPrecedence();

            // If this is a binop that binds at least as tightly as the current binop,
            // consume it, otherwise we are done.
            if (TokPrec < ExprPrec)
                return LHS;

            // Okay, we know this is a binop.
            int BinOp = CurTok;
            getNextToken(); // eat binop

            // Parse the primary expression after the binary operator.
            auto RHS = ParsePrimary();
            if (!RHS)
                return nullptr;

            // If BinOp binds less tightly with RHS than the operator after RHS, let
            // the pending operator take RHS as its LHS.
            int NextPrec = GetTokPrecedence();
            if (TokPrec < NextPrec) {
                RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
                if (!RHS)
                    return nullptr;
            }

            // Merge LHS/RHS.
            LHS = std::make_unique<AST::BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
        }
    }

    /// expression
    ///   ::= primary binoprhs
    ///
    static std::unique_ptr<AST::ExprAST> ParseExpression()
    {
        auto LHS = ParsePrimary();
        if (!LHS)
            return nullptr;

        return ParseBinOpRHS(0, std::move(LHS));
    }

    /// prototype
    ///   ::= id '(' id* ')'
    static std::unique_ptr<AST::PrototypeAST> ParsePrototype()
    {
        if (CurTok != Lexer::tok_identifier)
            return LogErrorP("Expected function name in prototype");

        std::string FnName = Lexer::IdentifierStr;
        getNextToken();

        if (CurTok != '(')
            return LogErrorP("Expected '(' in prototype");

        // Read the list of argument names.
        std::vector<std::string> ArgNames;
        while (getNextToken() == Lexer::tok_identifier)
            ArgNames.push_back(Lexer::IdentifierStr);
        if (CurTok != ')')
            return LogErrorP("Expected ')' in prototype");

        // success.
        getNextToken();  // eat ')'.

        return std::make_unique<AST::PrototypeAST>(FnName, std::move(ArgNames));
    }

    /// definition
    ///   ::= 'def' prototype expression
    static std::unique_ptr<AST::FunctionAST> ParseDefinition()
    {
        getNextToken();  // eat def.
        auto Proto = ParsePrototype();
        if (!Proto) return nullptr;

        if (auto E = ParseExpression())
            return std::make_unique<AST::FunctionAST>(std::move(Proto), std::move(E));
        return nullptr;
    }

    /// external ::= 'extern' prototype
    static std::unique_ptr<AST::PrototypeAST> ParseExtern()
    {
        getNextToken();  // eat extern.
        return ParsePrototype();
    }

    /// toplevelexpr ::= expression
    static std::unique_ptr<AST::FunctionAST> ParseTopLevelExpr()
    {
        if (auto E = ParseExpression()) {
            // Make an anonymous proto.
            auto Proto = std::make_unique<AST::PrototypeAST>("", std::vector<std::string>());
            return std::make_unique<AST::FunctionAST>(std::move(Proto), std::move(E));
        }
        return nullptr;
    }
}

#pragma endregion PARSER

#pragma region Code Generation
//===----------------------------------------------------------------------===//
// Code Generation
//===----------------------------------------------------------------------===//

static std::unique_ptr<llvm::LLVMContext> TheContext;
static std::unique_ptr<llvm::IRBuilder<>> Builder;
static std::unique_ptr<llvm::Module> TheModule;
static std::map<std::string, llvm::Value *> NamedValues;

llvm::Value *LogErrorV(const char *Str)
{
    Parser::LogError(Str);
    return nullptr;
}

llvm::Value *AST::NumberExprAST::codegen()
{
    return llvm::ConstantFP::get(*TheContext, llvm::APFloat(Val));
}

llvm::Value *AST::VariableExprAST::codegen()
{
    // Look this variable up in the function.
    llvm::Value *V = NamedValues[Name];
    if (!V)
        LogErrorV("Unknown variable name");
    return V;
}

llvm::Value *AST::BinaryExprAST::codegen()
{
    llvm::Value *L = LHS->codegen();
    llvm::Value *R = RHS->codegen();
    if (!L || !R)
        return nullptr;

    switch (Op) {
        case '+':
            return Builder->CreateFAdd(L, R, "addtmp");
        case '-':
            return Builder->CreateFSub(L, R, "subtmp");
        case '*':
            return Builder->CreateFMul(L, R, "multmp");
        case '<':
            L = Builder->CreateFCmpULT(L, R, "cmptmp");
            // Convert bool 0/1 to double 0.0 or 1.0
            return Builder->CreateUIToFP(L, llvm::Type::getDoubleTy(*TheContext), "booltmp");
        default:
            return LogErrorV("invalid binary operator");
    }
}

llvm::Value *AST::CallExprAST::codegen()
{
    // Look up the name in the global module table.
    llvm::Function *CalleeF = TheModule->getFunction(Callee);
    if (!CalleeF)
        return LogErrorV("Unknown function referenced");

    // If argument mismatch error.
    if (CalleeF->arg_size() != Args.size())
        return LogErrorV("Incorrect # arguments passed");

    std::vector<llvm::Value *> ArgsV;
    for (unsigned i = 0, e = Args.size(); i != e; ++i) {
        ArgsV.push_back(Args[i]->codegen());
        if (!ArgsV.back())
            return nullptr;
    }

    return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

llvm::Function *AST::PrototypeAST::codegen()
{
    // Make the function type:  double(double,double) etc.
    std::vector<llvm::Type *> Doubles(Args.size(), llvm::Type::getDoubleTy(*TheContext));
    llvm::FunctionType *FT = llvm::FunctionType::get(llvm::Type::getDoubleTy(*TheContext), Doubles, false);
    llvm::Function *F = llvm::Function::Create(FT, llvm::Function::ExternalLinkage, Name, TheModule.get());

    // Set names for all arguments.
    unsigned Idx = 0;
    for (auto &Arg : F->args())
        Arg.setName(Args[Idx++]);

    return F;
}

llvm::Function *AST::FunctionAST::codegen() {
    // First, check for an existing function from a previous 'extern' declaration.
    llvm::Function *TheFunction = TheModule->getFunction(Proto->getName());

    if (!TheFunction)
        TheFunction = Proto->codegen();

    if (!TheFunction)
        return nullptr;

    // Create a new basic block to start insertion into.
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    NamedValues.clear();
    for (auto &Arg : TheFunction->args())
        NamedValues[std::string(Arg.getName())] = &Arg;

    if (llvm::Value *RetVal = Body->codegen()) {
        // Finish off the function.
        Builder->CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        llvm::verifyFunction(*TheFunction);

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();
    return nullptr;
}

#pragma endregion Code Generation

#pragma region JIT Driver
//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

namespace TestDriver
{
    static void InitializeModule()
    {
        // Open a new context and module.
        TheContext = std::make_unique<llvm::LLVMContext>();
        TheModule = std::make_unique<llvm::Module>("My first jit", *TheContext);

        // Create a new builder for the module.
        Builder = std::make_unique<llvm::IRBuilder<>>(*TheContext);
    }

    static void HandleDefinition()
    {
        if (auto FnAST = Parser::ParseDefinition()) {
            if (auto *FnIR = FnAST->codegen()) {
                fprintf(stderr, "Read function definition: \n");
                FnIR->print(llvm::errs());
                fprintf(stderr, "\n");
            }
        } else {
            // Skip token for error recovery.
            Parser::getNextToken();
        }
    }

    static void HandleExtern()
    {
        if (auto ProtoAST = Parser::ParseExtern()) {
            if (auto *FnIR = ProtoAST->codegen()) {
                fprintf(stderr, "Read extern: \n");
                FnIR->print(llvm::errs());
                fprintf(stderr, "\n");
            }
        } else {
            // Skip token for error recovery.
            Parser::getNextToken();
        }
    }

    static void HandleTopLevelExpression()
    {
        // Evaluate a top-level expression into an anonymous function.
        if (auto FnAST = Parser::ParseTopLevelExpr()) {
            if (auto *FnIR = FnAST->codegen()) {
                fprintf(stderr, "Read top-level expression: \n");
                FnIR->print(llvm::errs());
                fprintf(stderr, "\n");

                // Remove the anonymous expression.
                FnIR->eraseFromParent();
            }
        } else {
            // Skip token for error recovery.
            Parser::getNextToken();
        }
    }

    /// top ::= definition | external | expression | ';'
    static void MainLoop()
    {
        while (1)
        {
            fprintf(stderr, "ready> ");
            switch (Parser::CurTok) {
                case Lexer::tok_eof:
                    return;
                case ';': // ignore top-level semicolons.
                    Parser::getNextToken();
                    break;
                case Lexer::tok_def:
                    HandleDefinition();
                    break;
                case Lexer::tok_extern:
                    HandleExtern();
                    break;
                default:
                    HandleTopLevelExpression();
                    break;
            }
        }
    }
}
#pragma endregion JIT Driver

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
int main() {
//    std::cout << "Hello, World!" << std::endl;

    // Install standard binary operators.
    // 1 is lowest precedence.
    Parser::BinopPrecedence['<'] = 10;
    Parser::BinopPrecedence['+'] = 20;
    Parser::BinopPrecedence['-'] = 20;
    Parser::BinopPrecedence['*'] = 40; // highest.

    // Prime the first token.
    fprintf(stderr, "ready> ");
    Parser::getNextToken();

    // Make the module, which holds all the code.
    TestDriver::InitializeModule();

    // Run the main "interpreter loop" now.
    TestDriver::MainLoop();

    // Print out all of the generated code.
    TheModule->print(llvm::errs(), nullptr);

    return 0;
}
