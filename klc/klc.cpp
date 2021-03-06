#include "../Warning.h"

#include "llvm/IR/Value.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Verifier.h"

#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Utils.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/TargetRegistry.h"

#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

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

        // control
        tok_if = -6,
        tok_then = -7,
        tok_else = -8,
        tok_for = -9,
        tok_in = -10,

        // operators
        tok_binary = -11,
        tok_unary = -12,

        // var definition
        tok_var = -13,
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
            if (IdentifierStr == "if")
                return tok_if;
            if (IdentifierStr == "then")
                return tok_then;
            if (IdentifierStr == "else")
                return tok_else;
            if (IdentifierStr == "for")
                return tok_for;
            if (IdentifierStr == "in")
                return tok_in;

            if (IdentifierStr == "binary")
                return tok_binary;
            if (IdentifierStr == "unary")
                return tok_unary;

            if (IdentifierStr == "var")
                return tok_var;

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
        const std::string &getName() const { return Name; }
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

    /// UnaryExprAST - Expression class for a unary operator.
    class UnaryExprAST : public ExprAST
    {
        char Opcode;
        std::unique_ptr<ExprAST> Operand;

    public:
        UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
                : Opcode(Opcode), Operand(std::move(Operand)) {}

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

        bool IsOperator;
        unsigned Precedence;  // Precedence if a binary op.

    public:
        PrototypeAST(const std::string &name, std::vector<std::string> Args, bool IsOperator = false, unsigned Prec = 0)
                : Name(name), Args(std::move(Args)), IsOperator(IsOperator), Precedence(Prec) {}

        const std::string &getName() const { return Name; }

        llvm::Function *codegen();

        bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
        bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

        char getOperatorName() const {
            assert(isUnaryOp() || isBinaryOp());
            return Name[Name.size() - 1];
        }

        unsigned getBinaryPrecedence() const { return Precedence; }
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

    /// IfExprAST - Expression class for if/then/else.
    class IfExprAST : public ExprAST
    {
        std::unique_ptr<ExprAST> Cond, Then, Else;

    public:
        IfExprAST(std::unique_ptr<ExprAST> Cond,
                  std::unique_ptr<ExprAST> Then,
                  std::unique_ptr<ExprAST> Else)
                : Cond(std::move(Cond))
                , Then(std::move(Then))
                , Else(std::move(Else)) {}

        llvm::Value *codegen() override;
    };

    /// ForExprAST - Expression class for for/in.
    class ForExprAST : public ExprAST
    {
        std::string VarName;
        std::unique_ptr<ExprAST> Start, End, Step, Body;

    public:
        ForExprAST(const std::string &VarName,
                   std::unique_ptr<ExprAST> Start,
                   std::unique_ptr<ExprAST> End,
                   std::unique_ptr<ExprAST> Step,
                   std::unique_ptr<ExprAST> Body)
                : VarName(VarName)
                , Start(std::move(Start))
                , End(std::move(End))
                , Step(std::move(Step))
                , Body(std::move(Body)) {}

        llvm::Value *codegen() override;
    };

    /// VarExprAST - Expression class for var/in
    class VarExprAST : public ExprAST
    {
        std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames;
        std::unique_ptr<ExprAST> Body;

    public:
        VarExprAST(
                std::vector<std::pair<std::string, std::unique_ptr<ExprAST>>> VarNames,
                std::unique_ptr<ExprAST> Body)
                : VarNames(std::move(VarNames))
                , Body(std::move(Body)) {}

        llvm::Value *codegen() override;
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

    /// ifexpr ::= 'if' expression 'then' expression 'else' expression
    static std::unique_ptr<AST::ExprAST> ParseIfExpr()
    {
        getNextToken();  // eat the if.

        // condition.
        auto Cond = ParseExpression();
        if (!Cond)
            return nullptr;

        if (CurTok != Lexer::tok_then)
            return LogError("expected then");
        getNextToken();  // eat the then

        auto Then = ParseExpression();
        if (!Then)
            return nullptr;

        if (CurTok != Lexer::tok_else)
            return LogError("expected else");

        getNextToken();

        auto Else = ParseExpression();
        if (!Else)
            return nullptr;

        return std::make_unique<AST::IfExprAST>(std::move(Cond), std::move(Then), std::move(Else));
    }

    /// forexpr ::= 'for' identifier '=' expr ',' expr (',' expr)? 'in' expression
    static std::unique_ptr<AST::ExprAST> ParseForExpr() {
        getNextToken();  // eat the for.

        if (CurTok != Lexer::tok_identifier)
            return LogError("expected identifier after for");

        std::string IdName = Lexer::IdentifierStr;
        getNextToken();  // eat identifier.

        if (CurTok != '=')
            return LogError("expected '=' after for");
        getNextToken();  // eat '='.


        auto Start = ParseExpression();
        if (!Start)
            return nullptr;
        if (CurTok != ',')
            return LogError("expected ',' after for start value");
        getNextToken();

        auto End = ParseExpression();
        if (!End)
            return nullptr;

        // The step value is optional.
        std::unique_ptr<AST::ExprAST> Step;
        if (CurTok == ',') {
            getNextToken();
            Step = ParseExpression();
            if (!Step)
                return nullptr;
        }

        if (CurTok != Lexer::tok_in)
            return LogError("expected 'in' after for");
        getNextToken();  // eat 'in'.

        auto Body = ParseExpression();
        if (!Body)
            return nullptr;

        return std::make_unique<AST::ForExprAST>(IdName, std::move(Start), std::move(End), std::move(Step), std::move(Body));
    }

    /// varexpr ::= 'var' identifier ('=' expression)? (',' identifier ('=' expression)?)* 'in' expression
    static std::unique_ptr<AST::ExprAST> ParseVarExpr()
    {
        getNextToken();  // eat the var.

        std::vector<std::pair<std::string, std::unique_ptr<AST::ExprAST>>> VarNames;

        // At least one variable name is required.
        if (CurTok != Lexer::tok_identifier)
            return LogError("expected identifier after var");

        while (1) {
            std::string Name = Lexer::IdentifierStr;
            getNextToken();  // eat identifier.

            // Read the optional initializer.
            std::unique_ptr<AST::ExprAST> Init;
            if (CurTok == '=') {
                getNextToken(); // eat the '='.

                Init = ParseExpression();
                if (!Init) return nullptr;
            }

            VarNames.push_back(std::make_pair(Name, std::move(Init)));

            // End of var list, exit loop.
            if (CurTok != ',') break;
            getNextToken(); // eat the ','.

            if (CurTok != Lexer::tok_identifier)
                return LogError("expected identifier list after var");
        }

        // At this point, we have to have 'in'.
        if (CurTok != Lexer::tok_in)
            return LogError("expected 'in' keyword after 'var'");
        getNextToken();  // eat 'in'.

        auto Body = ParseExpression();
        if (!Body)
            return nullptr;

        return std::make_unique<AST::VarExprAST>(std::move(VarNames), std::move(Body));
    }

    /// primary
    ///   ::= identifierexpr
    ///   ::= numberexpr
    ///   ::= parenexpr
    ///   ::= ifexpr
    ///   ::= forexpr
    ///   ::= varexpr
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
            case Lexer::tok_if:
                return ParseIfExpr();
            case Lexer::tok_for:
                return ParseForExpr();
            case Lexer::tok_var:
                return ParseVarExpr();
        }
    }

    /// unary
    ///   ::= primary
    ///   ::= '!' unary
    static std::unique_ptr<AST::ExprAST> ParseUnary()
    {
        // If the current token is not an operator, it must be a primary expr.
        if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
            return ParsePrimary();

        // If this is a unary operator, read it.
        int Opc = CurTok;
        getNextToken();
        if (auto Operand = ParseUnary())
            return std::make_unique<AST::UnaryExprAST>(Opc, std::move(Operand));
        return nullptr;
    }

    /// binoprhs
    ///   ::= ('+' unary)*
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

            // Parse the unary expression after the binary operator.
            auto RHS = ParseUnary();
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
    ///   ::= unary binoprhs
    ///
    static std::unique_ptr<AST::ExprAST> ParseExpression()
    {
        auto LHS = ParseUnary();
        if (!LHS)
            return nullptr;

        return ParseBinOpRHS(0, std::move(LHS));
    }

    /// prototype
    ///   ::= id '(' id* ')'
    ///   ::= binary LETTER number? (id, id)
    ///   ::= unary LETTER (id)
    static std::unique_ptr<AST::PrototypeAST> ParsePrototype()
    {
        std::string FnName;

        unsigned Kind = 0;  // 0 = identifier, 1 = unary, 2 = binary.
        unsigned BinaryPrecedence = 30;

        switch (CurTok) {
            default:
                return LogErrorP("Expected function name in prototype");
            case Lexer::tok_identifier:
                FnName = Lexer::IdentifierStr;
                Kind = 0;
                getNextToken();
                break;
            case Lexer::tok_unary:
                getNextToken();
                if (!isascii(CurTok))
                    return LogErrorP("Expected unary operator");
                FnName = "unary";
                FnName += (char)CurTok;
                Kind = 1;
                getNextToken();
                break;
            case Lexer::tok_binary:
                getNextToken();
                if (!isascii(CurTok))
                    return LogErrorP("Expected binary operator");
                FnName = "binary";
                FnName += (char)CurTok;
                Kind = 2;
                getNextToken();

                // Read the precedence if present.
                if (CurTok == Lexer::tok_number) {
                    if (Lexer::NumVal < 1 || Lexer::NumVal > 100)
                        return LogErrorP("Invalid precedence: must be 1..100");
                    BinaryPrecedence = (unsigned)Lexer::NumVal;
                    getNextToken();
                }
                break;
        }

        if (CurTok != '(')
            return LogErrorP("Expected '(' in prototype");

        std::vector<std::string> ArgNames;
        while (getNextToken() == Lexer::tok_identifier)
            ArgNames.push_back(Lexer::IdentifierStr);
        if (CurTok != ')')
            return LogErrorP("Expected ')' in prototype");

        // success.
        getNextToken();  // eat ')'.

        // Verify right number of names for operator.
        if (Kind && ArgNames.size() != Kind)
            return LogErrorP("Invalid number of operands for operator");

        return std::make_unique<AST::PrototypeAST>(FnName, std::move(ArgNames), Kind != 0, BinaryPrecedence);
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
            auto Proto = std::make_unique<AST::PrototypeAST>("__anon_expr", std::vector<std::string>());
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
static std::map<std::string, llvm::AllocaInst *> NamedValues;

//static std::unique_ptr<llvm::legacy::FunctionPassManager> TheFPM;

static std::map<std::string, std::unique_ptr<AST::PrototypeAST>> FunctionProtos;
static llvm::ExitOnError ExitOnErr;

llvm::Value *LogErrorV(const char *Str)
{
    Parser::LogError(Str);
    return nullptr;
}

llvm::Function *getFunction(std::string Name)
{
    // First, see if the function has already been added to the current module.
    if (auto *F = TheModule->getFunction(Name))
        return F;

    // If not, check whether we can codegen the declaration from some existing prototype.
    auto FI = FunctionProtos.find(Name);
    if (FI != FunctionProtos.end())
        return FI->second->codegen();

    // If no existing prototype exists, return null.
    return nullptr;
}

/// CreateEntryBlockAlloca - Create an alloca instruction in the entry block of
/// the function.  This is used for mutable variables etc.
static llvm::AllocaInst *CreateEntryBlockAlloca(llvm::Function *TheFunction, const std::string &VarName) {
    llvm::IRBuilder<> TmpB(&TheFunction->getEntryBlock(), TheFunction->getEntryBlock().begin());
    return TmpB.CreateAlloca(llvm::Type::getDoubleTy(*TheContext), 0, VarName.c_str());
}

llvm::Value *AST::NumberExprAST::codegen()
{
    return llvm::ConstantFP::get(*TheContext, llvm::APFloat(Val));
}

llvm::Value *AST::VariableExprAST::codegen()
{
    // Look this variable up in the function.
    llvm::AllocaInst *A = NamedValues[Name];
    if (!A)
        LogErrorV("Unknown variable name");

    // Load the value.
    return Builder->CreateLoad(A->getAllocatedType(), A, Name.c_str());
}

llvm::Value *AST::UnaryExprAST::codegen()
{
    llvm::Value *OperandV = Operand->codegen();
    if (!OperandV)
        return nullptr;

    llvm::Function *F = getFunction(std::string("unary") + Opcode);
    if (!F)
        return LogErrorV("Unknown unary operator");

    return Builder->CreateCall(F, OperandV, "unop");
}

llvm::Value *AST::BinaryExprAST::codegen()
{
    // Special case '=' because we don't want to emit the LHS as an expression.
    if (Op == '=') {
        // Assignment requires the LHS to be an identifier.
        // This assume we're building without RTTI because LLVM builds that way by
        // default.  If you build LLVM with RTTI this can be changed to a
        // dynamic_cast for automatic error checking.
        AST::VariableExprAST *LHSE = static_cast<AST::VariableExprAST *>(LHS.get());
        if (!LHSE)
            return LogErrorV("destination of '=' must be a variable");
        // Codegen the RHS.
        llvm::Value *Val = RHS->codegen();
        if (!Val)
            return nullptr;

        // Look up the name.
        llvm::Value *Variable = NamedValues[LHSE->getName()];
        if (!Variable)
            return LogErrorV("Unknown variable name");

        Builder->CreateStore(Val, Variable);
        return Val;
    }

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
            break;
    }

    // If it wasn't a builtin binary operator, it must be a user defined one. Emit
    // a call to it.
    llvm::Function *F = getFunction(std::string("binary") + Op);
    assert(F && "binary operator not found!");

    llvm::Value *Ops[2] = { L, R };
    return Builder->CreateCall(F, Ops, "binop");
}

llvm::Value *AST::CallExprAST::codegen()
{
    // Look up the name in the global module table.
    llvm::Function *CalleeF = getFunction(Callee);
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
    // Transfer ownership of the prototype to the FunctionProtos map, but keep a
    // reference to it for use below.
    auto &P = *Proto;
    FunctionProtos[Proto->getName()] = std::move(Proto);
    llvm::Function *TheFunction = getFunction(P.getName());
    if (!TheFunction)
        return nullptr;

    // If this is an operator, install it.
    if (P.isBinaryOp())
        Parser::BinopPrecedence[P.getOperatorName()] = P.getBinaryPrecedence();

    // Create a new basic block to start insertion into.
    llvm::BasicBlock *BB = llvm::BasicBlock::Create(*TheContext, "entry", TheFunction);
    Builder->SetInsertPoint(BB);

    // Record the function arguments in the NamedValues map.
    NamedValues.clear();
    for (auto &Arg : TheFunction->args()) {
        // Create an alloca for this variable.
        llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, (std::string)Arg.getName());

        // Store the initial value into the alloca.
        Builder->CreateStore(&Arg, Alloca);

        // Add arguments to variable symbol table.
        NamedValues[std::string(Arg.getName())] = Alloca;
    }


    if (llvm::Value *RetVal = Body->codegen()) {
        // Finish off the function.
        Builder->CreateRet(RetVal);

        // Validate the generated code, checking for consistency.
        llvm::verifyFunction(*TheFunction);

        // Optimize the function.
//        TheFPM->run(*TheFunction);

//        TheFunction->viewCFG();
//        TheFunction->viewCFGOnly();

        return TheFunction;
    }

    // Error reading body, remove function.
    TheFunction->eraseFromParent();
    return nullptr;
}

llvm::Value *AST::IfExprAST::codegen()
{
    llvm::Value *CondV = Cond->codegen();
    if (!CondV)
        return nullptr;

    // Convert condition to a bool by comparing non-equal to 0.0.
    CondV = Builder->CreateFCmpONE(CondV, llvm::ConstantFP::get(*TheContext, llvm::APFloat(0.0)), "ifcond");

    llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();

    // Create blocks for the then and else cases.  Insert the 'then' block at the end of the function.
    // llvm::BasicBlock *ThenBB =llvm::BasicBlock::Create(*TheContext, "then", TheFunction);
    llvm::BasicBlock *ThenBB = llvm::BasicBlock::Create(*TheContext, "then");
    llvm::BasicBlock *ElseBB = llvm::BasicBlock::Create(*TheContext, "else");
    llvm::BasicBlock *MergeBB = llvm::BasicBlock::Create(*TheContext, "ifcont");

    Builder->CreateCondBr(CondV, ThenBB, ElseBB);

    // Emit then block.
    TheFunction->getBasicBlockList().push_back(ThenBB);
    Builder->SetInsertPoint(ThenBB);

    llvm::Value *ThenV = Then->codegen();
    if (!ThenV)
        return nullptr;

    Builder->CreateBr(MergeBB);
    // Codegen of 'Then' can change the current block, update ThenBB for the PHI.
    ThenBB = Builder->GetInsertBlock();

    // Emit else block.
    TheFunction->getBasicBlockList().push_back(ElseBB);
    Builder->SetInsertPoint(ElseBB);

    llvm::Value *ElseV = Else->codegen();
    if (!ElseV)
        return nullptr;

    Builder->CreateBr(MergeBB);
    // Codegen of 'Else' can change the current block, update ElseBB for the PHI.
    ElseBB = Builder->GetInsertBlock();

    // Emit merge block.
    TheFunction->getBasicBlockList().push_back(MergeBB);
    Builder->SetInsertPoint(MergeBB);

    llvm::PHINode *PN = Builder->CreatePHI(llvm::Type::getDoubleTy(*TheContext), 2, "iftmp");

    PN->addIncoming(ThenV, ThenBB);
    PN->addIncoming(ElseV, ElseBB);
    return PN;
}

// Output for-loop as:
//   var = alloca double
//   ...
//   start = startexpr
//   store start -> var
//   goto loop
// loop:
//   ...
//   bodyexpr
//   ...
// loopend:
//   step = stepexpr
//   endcond = endexpr
//
//   curvar = load var
//   nextvar = curvar + step
//   store nextvar -> var
//   br endcond, loop, endloop
// outloop:
llvm::Value *AST::ForExprAST::codegen()
{
    llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();

    // Create an alloca for the variable in the entry block.
    llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);

    // Emit the start code first, without 'variable' in scope.
    llvm::Value *StartVal = Start->codegen();
    if (!StartVal)
        return nullptr;

    // Store the value into the alloca.
    Builder->CreateStore(StartVal, Alloca);

    // Make the new basic block for the loop header, inserting after current block.
    llvm::BasicBlock *LoopBB = llvm::BasicBlock::Create(*TheContext, "loop", TheFunction);

    // Insert an explicit fall through from the current block to the LoopBB.
    Builder->CreateBr(LoopBB);

    // Start insertion in LoopBB.
    Builder->SetInsertPoint(LoopBB);

    // Within the loop, the variable is defined equal to the PHI node.  If it
    // shadows an existing variable, we have to restore it, so save it now.
    llvm::AllocaInst *OldVal = NamedValues[VarName];
    NamedValues[VarName] = Alloca;

    // Emit the body of the loop.  This, like any other expr, can change the
    // current BB.  Note that we ignore the value computed by the body, but don't
    // allow an error.
    if (!Body->codegen())
        return nullptr;

    // Emit the step value.
    llvm::Value *StepVal = nullptr;
    if (Step) {
        StepVal = Step->codegen();
        if (!StepVal)
            return nullptr;
    } else {
        // If not specified, use 1.0.
        StepVal = llvm::ConstantFP::get(*TheContext, llvm::APFloat(1.0));
    }

    // Compute the end condition.
    llvm::Value *EndCond = End->codegen();
    if (!EndCond)
        return nullptr;

    // Convert condition to a bool by comparing non-equal to 0.0.
    EndCond = Builder->CreateFCmpONE( EndCond, llvm::ConstantFP::get(*TheContext, llvm::APFloat(0.0)), "loopcond");

    // Create the "after loop" block and insert it.
    llvm::BasicBlock *AfterBB = llvm::BasicBlock::Create(*TheContext, "afterloop", TheFunction);

    // Insert the conditional branch into the end of LoopEndBB.
    Builder->CreateCondBr(EndCond, LoopBB, AfterBB);

    // Any new code will be inserted in AfterBB.
    Builder->SetInsertPoint(AfterBB);

    // Restore the unshadowed variable.
    if (OldVal)
        NamedValues[VarName] = OldVal;
    else
        NamedValues.erase(VarName);

    // for expr always returns 0.0.
    return llvm::Constant::getNullValue(llvm::Type::getDoubleTy(*TheContext));
}

llvm::Value *AST::VarExprAST::codegen()
{
    std::vector<llvm::AllocaInst *> OldBindings;

    llvm::Function *TheFunction = Builder->GetInsertBlock()->getParent();

    // Register all variables and emit their initializer.
    for (unsigned i = 0, e = VarNames.size(); i != e; ++i) {
        const std::string &VarName = VarNames[i].first;
        AST::ExprAST *Init = VarNames[i].second.get();

        // Emit the initializer before adding the variable to scope, this prevents
        // the initializer from referencing the variable itself, and permits stuff
        // like this:
        //  var a = 1 in
        //    var a = a in ...   # refers to outer 'a'.
        llvm::Value *InitVal;
        if (Init) {
            InitVal = Init->codegen();
            if (!InitVal)
                return nullptr;
        } else { // If not specified, use 0.0.
            InitVal = llvm::ConstantFP::get(*TheContext, llvm::APFloat(0.0));
        }

        llvm::AllocaInst *Alloca = CreateEntryBlockAlloca(TheFunction, VarName);
        Builder->CreateStore(InitVal, Alloca);

        // Remember the old variable binding so that we can restore the binding when
        // we unrecurse.
        OldBindings.push_back(NamedValues[VarName]);

        // Remember this binding.
        NamedValues[VarName] = Alloca;
    }

    // Codegen the body, now that all vars are in scope.
    llvm::Value *BodyVal = Body->codegen();
    if (!BodyVal)
        return nullptr;

    // Pop all our variables from scope.
    for (unsigned i = 0, e = VarNames.size(); i != e; ++i)
        NamedValues[VarNames[i].first] = OldBindings[i];

    // Return the body computation.
    return BodyVal;
}

#pragma endregion Code Generation

#pragma region JIT Driver
//===----------------------------------------------------------------------===//
// Top-Level parsing and JIT Driver
//===----------------------------------------------------------------------===//

namespace TestDriver
{
    static void InitializeModuleAndPassManager()
    {
        // Open a new context and module.
        TheContext = std::make_unique<llvm::LLVMContext>();
        TheModule = std::make_unique<llvm::Module>("My first jit", *TheContext);

        // Create a new builder for the module.
        Builder = std::make_unique<llvm::IRBuilder<>>(*TheContext);

//        // Create a new pass manager attached to it.
//        TheFPM = std::make_unique<llvm::legacy::FunctionPassManager>(TheModule.get());
//
//        // Promote allocas to registers.
//        TheFPM->add(llvm::createPromoteMemoryToRegisterPass());
//        // Do simple "peephole" optimizations and bit-twiddling optzns.
//        TheFPM->add(llvm::createInstructionCombiningPass());
//        // Reassociate expressions.
//        TheFPM->add(llvm::createReassociatePass());
//        // Eliminate Common SubExpressions.
//        TheFPM->add(llvm::createGVNPass());
//        // Simplify the control flow graph (deleting unreachable blocks, etc).
//        TheFPM->add(llvm::createCFGSimplificationPass());
//
//        TheFPM->doInitialization();
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
                FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
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
            if (!FnAST->codegen())
                fprintf(stderr, "Error generating code for top level expr");
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

#pragma region Extern Library
//===----------------------------------------------------------------------===//
// "Library" functions that can be "extern'd" from user code.
//===----------------------------------------------------------------------===//

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

/// putchard - putchar that takes a double and returns 0.
extern "C" DLLEXPORT double putchard(double x)
{
    fputc((char)x, stderr);
    return 0;
}

/// printd - printf that takes a double prints it as "%f\n", returning 0.
extern "C" DLLEXPORT double printd(double x)
{
    fprintf(stderr, "%f\n", x);
    return 0;
}

#ifdef _WIN32
/// msgboxd - MessageBox on win32
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
extern "C" DLLEXPORT double msgboxd(double x)
{
    std::string showMsg = "x = " + std::to_string(x) + "\n";
    MessageBox(NULL, (LPCSTR)(showMsg.c_str()), (LPCSTR)("Show value!"), MB_ICONWARNING | MB_CANCELTRYCONTINUE | MB_DEFBUTTON2);
    return 0;
}
#endif

#pragma endregion Extern Library

//===----------------------------------------------------------------------===//
// Main driver code.
//===----------------------------------------------------------------------===//
int main() {
//    std::cout << "Hello, World!" << std::endl;

    // Install standard binary operators.
    // 1 is lowest precedence.
    Parser::BinopPrecedence['='] = 2;
    Parser::BinopPrecedence['<'] = 10;
    Parser::BinopPrecedence['+'] = 20;
    Parser::BinopPrecedence['-'] = 20;
    Parser::BinopPrecedence['*'] = 40; // highest.

    // Prime the first token.
    Parser::getNextToken();

    // Make the module, which holds all the code.
    TestDriver::InitializeModuleAndPassManager();

    // Run the main "interpreter loop" now.
    TestDriver::MainLoop();

    // Initialize the target registry etc.
//    llvm::InitializeAllTargetInfos();
//    llvm::InitializeAllTargets();
//    llvm::InitializeAllTargetMCs();
//    llvm::InitializeAllAsmParsers();
//    llvm::InitializeAllAsmPrinters();

    LLVMInitializeX86TargetInfo();
    LLVMInitializeX86Target();
    LLVMInitializeX86TargetMC();
    LLVMInitializeX86AsmPrinter();
    LLVMInitializeX86AsmParser();

    auto TargetTriple = llvm::sys::getDefaultTargetTriple();
    TheModule->setTargetTriple(TargetTriple);

    std::string Error;
    auto Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);

    // Print an error and exit if we couldn't find the requested target.
    // This generally occurs if we've forgotten to initialise the
    // TargetRegistry or we have a bogus target triple.
    if (!Target) {
        llvm::errs() << Error;
        return 1;
    }

    auto CPU = "generic";
    auto Features = "";
//    auto Features = "avx";

    llvm::TargetOptions opt;
    auto RM = llvm::Optional<llvm::Reloc::Model>();
    auto TargetMachine = Target->createTargetMachine(TargetTriple, CPU, Features, opt, RM);

    TheModule->setDataLayout(TargetMachine->createDataLayout());

    auto Filename = "output.o";
    std::error_code EC;
    llvm::raw_fd_ostream dest(Filename, EC, llvm::sys::fs::OF_None);

    if (EC) {
        llvm::errs() << "Could not open file: " << EC.message();
        return 1;
    }

    llvm::legacy::PassManager pass;
    auto FileType = llvm::CGFT_ObjectFile;

    if (TargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
        llvm::errs() << "TargetMachine can't emit a file of this type";
        return 1;
    }

    pass.run(*TheModule);
    dest.flush();

    return 0;
}
