/**
 * \file Fixtures.h
 * \brief Toy models used to exercise the engines.
 *
 * These are **test fixtures, not library code**. libKalman ships no physics: a
 * constant-velocity tracker or a pendulum is an application, and applications
 * live where the domain knowledge lives. They exist here only because testing an
 * estimator requires *some* system to estimate, and a system whose exact
 * behaviour is known makes the assertions meaningful.
 *
 * They double as the worked examples of the two ways to write a model:
 * ConstantVelocity sets matrices and keeps the analytic Jacobian; Pendulum
 * overrides f() and asks for central differences.
 */
#ifndef KALMAN_TESTS_FIXTURES_H
#define KALMAN_TESTS_FIXTURES_H

#include "Kalman/Models/Extended.h"
#include "Kalman/Models/Linear.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Fixture
{
    /**
     * \brief Constant velocity with white-noise acceleration — the canonical linear model.
     *
     * Per spatial axis the state is (position, velocity); only position is
     * observed. Axes are stacked, so 2-D has state (px, vx, py, vy).
     *
     * The process noise is the standard continuous white-noise-acceleration block
     *
     *     Q = sigma_a^2 [ dt^4/4  dt^3/2 ]
     *                   [ dt^3/2  dt^2   ]
     *
     * whose off-diagonal terms are the point: a position error and a velocity
     * error produced by the same unmodelled acceleration are correlated. A
     * diagonal Q here is the most common way to get a plausible-looking but
     * inconsistent filter, which is exactly what the NIS/NEES assertions catch.
     */
    class ConstantVelocity : public Kalman::Models::Linear
    {
        private:
            std::size_t fNAxis = 1;
            double fSigmaAcc = 1.;
            double fSigmaMeas = 1.;

        public:
            ConstantVelocity(std::size_t naxis, double fs)
                : Kalman::Models::Linear("constant-velocity", 2 * naxis, naxis), fNAxis(naxis)
            {
                if (naxis == 0) throw std::runtime_error("ConstantVelocity: naxis must be positive.");
                SetSampleRate(fs);
                Build();       // Models::Linear already declared the engine and the Jacobian
            }

            std::size_t NAxis() const { return fNAxis; }

            void SetProcessNoise(double sigmaAcc) { fSigmaAcc = sigmaAcc; Build(); }
            void SetMeasurementNoise(double sigmaMeas) { fSigmaMeas = sigmaMeas; Build(); }

            void Build()
            {
                const double dt = DeltaT();
                const double q = fSigmaAcc * fSigmaAcc;

                Kalman::Matrix F = Kalman::Matrix::Identity(NState());
                Kalman::Matrix H(NMeasure(), NState(), 0.);
                Kalman::Matrix Q(NState(), NState(), 0.);
                Kalman::Matrix R(NMeasure(), NMeasure(), 0.);

                for (std::size_t a = 0; a < fNAxis; ++a)
                {
                    const std::size_t p = 2 * a;      // position index of this axis
                    const std::size_t v = p + 1;      // velocity index

                    F(p, v) = dt;
                    H(a, p) = 1.;

                    Q(p, p) = q * dt * dt * dt * dt / 4.;
                    Q(p, v) = q * dt * dt * dt / 2.;
                    Q(v, p) = Q(p, v);
                    Q(v, v) = q * dt * dt;

                    R(a, a) = fSigmaMeas * fSigmaMeas;
                }

                SetF(F); SetH(H); SetQ(Q); SetR(R);
            }
    };

    /**
     * \brief Constant velocity with an **acceleration command** — the controlled plant.
     *
     * State (position, velocity), one input `a`: `B = (dt^2/2, dt)`. An optional
     * actuator limit clips `|a|` so the cost hook and MPPI's rollouts can be
     * checked against it.
     */
    class ControlledVelocity : public ConstantVelocity
    {
        private:
            double fLimit = 0.;           // 0 = no limit

        public:
            ControlledVelocity(double fs, double limit = 0.)
                : ConstantVelocity(1, fs), fLimit(limit)
            {
                Allocate(2, 1, 1);        // re-size with l = 1 (keeps F/H/Q/R placeholders)
                Build();
                const double dt = DeltaT();
                Kalman::Matrix B(2, 1, 0.);
                B(0, 0) = 0.5 * dt * dt;
                B(1, 0) = dt;
                SetB(B);
            }

            double Limit() const { return fLimit; }

            Kalman::Vector ClipControl(const Kalman::Vector &u) const override
            {
                if (fLimit <= 0.) return u;
                Kalman::Vector out = u;
                out(0) = std::max(-fLimit, std::min(fLimit, out(0)));
                return out;
            }

            Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &u,
                             double t = 0.) const override
            {
                return Kalman::Model::f(x, u.Empty() ? u : ClipControl(u), t);
            }
    };

    /**
     * \brief Damped pendulum — the smallest genuinely nonlinear model.
     *
     *     theta_ddot = -(g/l) sin(theta) - gamma theta_dot
     *
     * so the nonlinearity is the sin(theta) that a small-angle model would replace
     * by theta. Only the angle is observed, so the filter has to infer the angular
     * velocity from the dynamics. The discrete map is one RK4 step, accurate enough
     * that the residuals reflect the estimation rather than the integrator.
     *
     * It declares SetAnalyticJacobian(false), so the central-difference path is on
     * the tested route.
     */
    class Pendulum : public Kalman::Models::Extended
    {
        private:
            double fLength = 1.;
            double fGravity = 9.81;
            double fDamping = 0.;

        public:
            Pendulum(double length, double damping, double fs)
                : Kalman::Models::Extended("pendulum", 2, 1), fLength(length), fDamping(damping)
            {
                // Models::Extended already declared the engine and switched the
                // Jacobian to central differences -- f() is nonlinear, so F(t) is
                // not its Jacobian.
                if (length <= 0.) throw std::runtime_error("Pendulum: the length must be positive.");
                SetSampleRate(fs);

                Kalman::Matrix H(1, 2, 0.);
                H(0, 0) = 1.;
                SetH(H);

                SetNoise(1e-3, 1e-2);
                SetInitialCovariance(Kalman::Matrix::Identity(2));
            }

            double Length() const { return fLength; }
            double Damping() const { return fDamping; }
            double Gravity() const { return fGravity; }

            Kalman::Vector Derivative(const Kalman::Vector &x) const
            {
                Kalman::Vector dx(2);
                dx(0) = x(1);
                dx(1) = -(fGravity / fLength) * std::sin(x(0)) - fDamping * x(1);
                return dx;
            }

            Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &,
                             double = 0.) const override
            {
                const double dt = DeltaT();

                const Kalman::Vector k1 = Derivative(x);
                Kalman::Vector x2(x); x2 += 0.5 * dt * k1;
                const Kalman::Vector k2 = Derivative(x2);
                Kalman::Vector x3(x); x3 += 0.5 * dt * k2;
                const Kalman::Vector k3 = Derivative(x3);
                Kalman::Vector x4(x); x4 += dt * k3;
                const Kalman::Vector k4 = Derivative(x4);

                Kalman::Vector out(x);
                out += (dt / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
                return out;
            }

            Kalman::Vector h(const Kalman::Vector &x, double = 0.) const override
            {
                Kalman::Vector z(1);
                z(0) = x(0);
                return z;
            }

            void SetNoise(double sigmaTorque, double sigmaAngle)
            {
                const double dt = DeltaT();
                const double q = sigmaTorque * sigmaTorque;

                Kalman::Matrix Q(2, 2, 0.);
                Q(0, 0) = q * dt * dt * dt * dt / 4.;
                Q(0, 1) = q * dt * dt * dt / 2.;
                Q(1, 0) = Q(0, 1);
                Q(1, 1) = q * dt * dt;
                SetQ(Q);

                Kalman::Matrix R(1, 1, 0.);
                R(0, 0) = sigmaAngle * sigmaAngle;
                SetR(R);
            }
    };

    /**
     * \brief The same pendulum written as an **ODE**: Derivative() plus the model's
     * integrator, instead of a hand-written RK4 in f().
     *
     * With the RK4 integrator and one substep the two fixtures must agree to
     * round-off, which is the test that the integrator path is wired correctly.
     */
    class PendulumODE : public Pendulum
    {
        public:
            PendulumODE(double length, double damping, double fs)
                : Pendulum(length, damping, fs)
            {
                UseContinuousDynamics();          // f() now integrates Derivative()
            }

            Kalman::Vector Derivative(const Kalman::Vector &x, const Kalman::Vector &,
                                      double = 0.) const override
            {
                return Pendulum::Derivative(x);
            }

            // Back to the base f(): Pendulum's hand-stepped override is what we compare against.
            Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &u,
                             double t = 0.) const override
            {
                return Kalman::Model::f(x, u, t);
            }
    };
}

#endif // KALMAN_TESTS_FIXTURES_H
