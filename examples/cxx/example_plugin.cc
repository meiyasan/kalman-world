/**
 * \file example_plugin.cc
 * \brief Two models compiled into a plugin — how C++ physics reaches the Python side.
 *
 * This translation unit is built as a *shared library*, `libKalmanPlugin_example`,
 * not an executable. It defines two models and registers each under a name; that
 * is the entire integration. Anything that can dlopen the library — the Python
 * package (`kalman.api.load_plugin`), the MCP server, a ROOT+ analysis, the
 * `06_factory` test — then finds them in Kalman::Factory by name and constructs
 * them without knowing their types.
 *
 * A plugin written for ROOT+ works the same way and may link ROOT freely: the
 * framework only ever sees Kalman::Model, so ROOT stays inside the plugin.
 *
 *     make build && make examples
 *     python -c "import kalman.api as api; print(api.load_plugin('KalmanPlugin_example'))"
 */
#include <Kalman/Factory.h>
#include <Kalman/Models/Extended.h>
#include <Kalman/Models/Linear.h>

#include <cmath>

namespace
{
    /// Constant velocity in 1-D with white-noise acceleration: state (position, velocity).
    class ConstantVelocity : public Kalman::Models::Linear
    {
        private:
            double fSigmaAcc;

        public:
            ConstantVelocity(double fs = 100., double sigmaAcc = 0.5, double sigmaMeas = 0.2)
                : Kalman::Models::Linear("constant-velocity", 2, 1), fSigmaAcc(sigmaAcc)
            {
                SetSampleRate(fs);

                Kalman::Matrix H(1, 2, 0.);
                H(0, 0) = 1.;
                SetH(H);

                Kalman::Matrix R(1, 1, 0.);
                R(0, 0) = sigmaMeas * sigmaMeas;
                SetR(R);

                Kalman::Vector x0(2);
                x0(1) = 1.;                           // moving at 1 m/s
                SetInitialState(x0);
                SetInitialCovariance(Kalman::Matrix::Identity(2));
            }

            const char *Title() const override { return "Constant Velocity"; }
            const char *Description() const override
            { return "1-D constant velocity with white-noise acceleration; position observed."; }

            // F and Q are derived from DeltaT() when asked for rather than fixed in
            // the constructor, so a consumer that changes the sample rate after
            // construction — the Python wrapper's `fs=` does exactly that — gets
            // coefficients that agree with the new step.
            Kalman::Matrix F(double) const override
            {
                Kalman::Matrix F = Kalman::Matrix::Identity(2);
                F(0, 1) = DeltaT();
                return F;
            }

            // Discrete white-noise acceleration (standard deviation sigma_a, constant
            // over each step), off-diagonals included: the one acceleration moves both
            // position and velocity, and a diagonal Q here would make the filter
            // inconsistent.
            Kalman::Matrix Q(double) const override
            {
                const double dt = DeltaT();
                const double q = fSigmaAcc * fSigmaAcc;
                Kalman::Matrix Q(2, 2, 0.);
                Q(0, 0) = q * dt * dt * dt * dt / 4.;
                Q(0, 1) = q * dt * dt * dt / 2.;
                Q(1, 0) = Q(0, 1);
                Q(1, 1) = q * dt * dt;
                return Q;
            }
    };

    /// Damped pendulum: theta_ddot = -(g/l) sin(theta) - gamma theta_dot; only the angle is observed.
    class Pendulum : public Kalman::Models::Extended
    {
        private:
            double fOmega2;   // g / l
            double fGamma;    // damping

        public:
            Pendulum(double fs = 50., double length = 1., double gamma = 0.1)
                : Kalman::Models::Extended("pendulum", 2, 1), fOmega2(9.81 / length), fGamma(gamma)
            {
                SetSampleRate(fs);

                Kalman::Matrix H(1, 2, 0.);
                H(0, 0) = 1.;
                SetH(H);

                // Diagonal Q, unlike ConstantVelocity: these are two independent tuning
                // values (angle and angular-velocity noise per step), not derived from one
                // physical noise source. A random angular acceleration would instead give
                // the correlated form used in ConstantVelocity::Q.
                Kalman::Matrix Q(2, 2, 0.);
                Q(0, 0) = 1e-6;
                Q(1, 1) = 1e-4;
                SetQ(Q);

                Kalman::Matrix R(1, 1, 0.);
                R(0, 0) = 1e-2;
                SetR(R);

                Kalman::Vector x0(2);
                x0(0) = 0.5;                          // released from half a radian
                SetInitialState(x0);
                SetInitialCovariance(Kalman::Matrix::Identity(2));
            }

            const char *Title() const override { return "Damped Pendulum"; }
            const char *Description() const override
            { return "Nonlinear pendulum, one RK4 step per sample; angle observed."; }

            Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &, double) const override
            {
                const double dt = DeltaT();
                auto deriv = [this](double th, double om, double &dth, double &dom)
                {
                    dth = om;
                    dom = -fOmega2 * std::sin(th) - fGamma * om;
                };

                double k1t, k1o, k2t, k2o, k3t, k3o, k4t, k4o;
                deriv(x(0), x(1), k1t, k1o);
                deriv(x(0) + 0.5 * dt * k1t, x(1) + 0.5 * dt * k1o, k2t, k2o);
                deriv(x(0) + 0.5 * dt * k2t, x(1) + 0.5 * dt * k2o, k3t, k3o);
                deriv(x(0) + dt * k3t, x(1) + dt * k3o, k4t, k4o);

                Kalman::Vector out(2);
                out(0) = x(0) + dt / 6. * (k1t + 2. * k2t + 2. * k3t + k4t);
                out(1) = x(1) + dt / 6. * (k1o + 2. * k2o + 2. * k3o + k4o);
                return out;
            }
    };
}

// The registrations. Names are lowercase-hyphenated by convention, so they can
// never be mistaken for the CamelCase Python classes they sit beside in the
// catalogue; the second ConstantVelocity shows a *configured* registration.
KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity");
KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity-slow", 10., 0.5, 0.2);
KALMAN_REGISTER_MODEL(Pendulum, "pendulum");
