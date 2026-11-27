#include <iostream>
#include <autodiff.hpp>
#include <fstream>
#include <vector>

using namespace ASC_ode;


template <typename T>
T func1 (T x, T y)
{
  return x * sin(y);
}



int main()
{
  double x = 1, y = 2;
  AutoDiff<2> adx = Variable<0>(x);
  AutoDiff<2> ady = Variable<1>(y);

  std::cout << "adx = " << adx << std::endl;
  std::cout << "ady = " << ady << std::endl;

  AutoDiff<2> prod = adx * ady;
  std::cout << "prod = " << prod << std::endl;

  std::cout << "func1(adx, ady) = " << func1(adx, ady) << std::endl;

  double eps = 1e-8;
  std::cout << "numdiff df/dx = " << (func1(x + eps, y) - func1(x-eps, y)) / (2*eps) << std::endl;
  std::cout << "numdiff df/dy = " << (func1(x, y + eps) - func1(x, y-eps)) / (2*eps) << std::endl;


  {
    // we can do second derivatives:
    AutoDiff<1, AutoDiff<1>> addx{Variable<0>(2)};
    std::cout << "addx = " << addx << std::endl;
    // func = x*x
    // func' = 2*x
    // func'' = 2
    std::cout << "addx*addx = " << addx * addx << std::endl;

    // sin(addx) needs cos for AutoDiff, which is an exercise:
    // std::cout << "sin(addx) = " << sin(addx) << std::endl;
  }
  //Legendre polynomials implementation
  std::vector<AutoDiff<1>> P;

  std::ofstream file("legendre.txt");

  file << "x";
  for (int k = 0; k <= 5; ++k)
    file << ",P" << k << ",dP" << k;
  file << "\n";

  // Evaluate on [-1, 1]
  const int num_points = 200;

  for (int j = 0; j <= num_points; ++j){
    //Ranges from -1 to 1 and takes a specific stepsize to get results for 200 x_values
    double x_value = -1.0 + 2.0 *j / num_points;
    AutoDiff<1> x_new{Variable<0, double>(x_value)};
    LegendrePolynomials(5,x_new, P);

    file << x_value;

    for (int k = 0; k <= 5; ++k)
      {
        file << "," << P[k].value()<< "," << P[k].deriv()[0];
      }

    file << "\n";
  }

  file.close();



  return 0;
};