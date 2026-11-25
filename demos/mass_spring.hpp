#ifndef MASS_SPRING_HPP
#define MASS_SPRING_HPP
#include <memory>
#include <string>
#include <stdexcept>

#include <nonlinfunc.hpp>
#include <timestepper.hpp>

namespace ASC_ode
{
  // m y'' = -k y, written as first-order system:
  //   y0' = y1
  //   y1' = -k/m y0
  class MassSpring : public NonlinearFunction
  {
  private:
    double mass;
    double stiffness;

  public:
    MassSpring(double m, double k) : mass(m), stiffness(k) {}

    size_t dimX() const override { return 2; }
    size_t dimF() const override { return 2; }

    void evaluate (VectorView<double> x, VectorView<double> f) const override
    {
      f(0) = x(1);
      f(1) = -stiffness/mass*x(0);
    }

    void evaluateDeriv (VectorView<double> x, MatrixView<double> df) const override
    {
      df = 0.0;
      df(0,1) = 1;
      df(1,0) = -stiffness/mass;
    }
  };
// create a time-stepper from its name, so programs can choose the method at run-time
  inline std::unique_ptr<TimeStepper> MakeStepper (const std::string & name,
                                                   std::shared_ptr<NonlinearFunction> rhs)
  {
    if (name == "explicit")       return std::make_unique<ExplicitEuler>(rhs);
    if (name == "improved")       return std::make_unique<ImprovedEuler>(rhs);
    if (name == "implicit")       return std::make_unique<ImplicitEuler>(rhs);
    if (name == "cranknicolson")  return std::make_unique<CrankNicolson>(rhs);
    throw std::invalid_argument("unknown method: " + name +
                                " (use explicit, improved, implicit or cranknicolson)");
  }


}


#endif