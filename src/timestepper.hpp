#ifndef TIMERSTEPPER_HPP
#define TIMERSTEPPER_HPP

#include <functional>
#include <exception>

#include "Newton.hpp"


namespace ASC_ode
{

  // base class of all time-stepping methods for y' = rhs(y)
  class TimeStepper
  {
  protected:
    std::shared_ptr<NonlinearFunction> m_rhs;
  public:
    TimeStepper(std::shared_ptr<NonlinearFunction> rhs) : m_rhs(rhs) {}
    virtual ~TimeStepper() = default;
    // advances y by one time-step tau
    virtual void doStep(double tau, VectorView<double> y) = 0;
  };

  // y_{n+1} = y_n + tau f(y_n)
  class ExplicitEuler : public TimeStepper
  {
    Vector<> m_vecf;   // allocated once, not in every step
  public:
    ExplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_vecf(rhs->dimF()) {}
    void doStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);
      y += tau * m_vecf;
    }
  };

  // Exercise 17.2: improved Euler (explicit midpoint rule)
  //   ytilde  = y_n + tau/2 f(y_n)     (half step with explicit Euler)
  //   y_{n+1} = y_n + tau f(ytilde)    (full step with the slope at the midpoint)
  class ImprovedEuler : public TimeStepper
  {
    Vector<> m_vecf;     // holds f(...), allocated once
    Vector<> m_ytilde;   // the midpoint value, allocated once
  public:
    ImprovedEuler(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs), m_vecf(rhs->dimF()), m_ytilde(rhs->dimX()) {}

    void doStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);         // m_vecf = f(y_n)
      m_ytilde = y + (tau/2) * m_vecf;          // half step
      this->m_rhs->evaluate(m_ytilde, m_vecf);  // m_vecf = f(ytilde)
      y += tau * m_vecf;                        // full step
    }
  };


  // solves y - y_n - tau f(y) = 0 with Newton's method. The equation is built once
  // by the function algebra, y_n and tau are updated in every step.
  class ImplicitEuler : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_tau;
    std::shared_ptr<ConstantFunction> m_yold;
  public:
    ImplicitEuler(std::shared_ptr<NonlinearFunction> rhs) 
    : TimeStepper(rhs), m_tau(std::make_shared<Parameter>(0.0)) 
    {
      m_yold = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_equ = ynew - m_yold - m_tau * m_rhs;
    }

    void doStep(double tau, VectorView<double> y) override
    {
      m_yold->set(y);
      m_tau->set(tau);
      NewtonSolver(m_equ, y);
    }
  };


  // Exercise 17.4: Crank-Nicolson (trapezoidal rule)
  //   y_{n+1} = y_n + tau/2 ( f(y_n) + f(y_{n+1}) )
  // Everything that only depends on the known y_n goes into one constant
  //   c = y_n + tau/2 f(y_n),
  // so Newton has to solve   y - c - tau/2 f(y) = 0   for y = y_{n+1}.
  // This is the same shape as implicit Euler, with tau replaced by tau/2.
  class CrankNicolson : public TimeStepper
  {
    std::shared_ptr<NonlinearFunction> m_equ;
    std::shared_ptr<Parameter> m_halftau;
    std::shared_ptr<ConstantFunction> m_known;   // c = y_n + tau/2 f(y_n)
    Vector<> m_vecf;
  public:
    CrankNicolson(std::shared_ptr<NonlinearFunction> rhs)
      : TimeStepper(rhs), m_halftau(std::make_shared<Parameter>(0.0)), m_vecf(rhs->dimF())
    {
      m_known = std::make_shared<ConstantFunction>(rhs->dimX());
      auto ynew = std::make_shared<IdentityFunction>(rhs->dimX());
      m_equ = ynew - m_known - m_halftau * m_rhs;
    }

    void doStep(double tau, VectorView<double> y) override
    {
      this->m_rhs->evaluate(y, m_vecf);   // f(y_n)
      m_vecf *= tau/2;                    // tau/2 f(y_n)
      m_vecf += y;                        // y_n + tau/2 f(y_n)
      m_known->set(m_vecf);
      m_halftau->set(tau/2);
      NewtonSolver(m_equ, y);             // y_n is the initial guess, overwritten by y_{n+1}
    }
  };



}


#endif
