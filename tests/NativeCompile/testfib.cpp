#include <iostream>

extern "C"
{
    double fib(double x);
}

int main()
{
    double result = fib(10);
    std::cout << "Fibonacci sequence at 10 is : " << result << std::endl;
}
