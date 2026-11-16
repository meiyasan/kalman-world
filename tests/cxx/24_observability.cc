/**
 * \file 24_observability.cc
 * \brief Kalman::Observability, and the tau of Models::Joint.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Observability.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <memory>

using namespace Kalman;

namespace
{
    /// A constant amplitude read through a gain: z = gain * a. The two enter
    /// only as their product, so no measurement tells them apart.
    class Gauge : public Models::Extended
    {
        private:
            double fGain = 2.;
            double fOffset = 0.;

        protected:
            void ParametersChanged() override { fGain = Parameter("gain"); fOffset = Parameter("offset"); }

        public:
            Gauge() : Models::Extended("gauge", 1, 1)
            {
                SetSampleRate(10.);
                SetDifferentiator(DifferentiatorKind::Forward, 1e-6);
                DeclareParameter("gain", 2.);
                DeclareParameter("offset", 0.);
                Matrix Q(1, 1, 0.); Q(0, 0) = 1e-8; SetQ(Q);
                Matrix R(1, 1, 0.); R(0, 0) = 1e-2; SetR(R);
                Vector x0(1); x0(0) = 3.; SetInitialState(x0);
                SetInitialCovariance(Matrix::Identity(1));
            }

            Vector f(const Vector &x, const Vector &, double = 0.) const override { return x; }
            Vector h(const Vector &x, double = 0.) const override
            {
                Vector z(1); z(0) = fGain * x(0) + fOffset; return z;
            }
    };
}

int main()
{
    // --- a pendulum measured in angle: both components are determined ----------------
    {
        Fixture::Pendulum pendulum(1.0, 0.15, 100.);
        Vector x0(2); x0(0) = 0.6; x0(1) = 0.;
        pendulum.SetInitialState(x0);
        const Observability obs(pendulum, 100);
        CHECK(obs.IsObservable() && obs.Rank() == 2, "pendulum: the angle determines the angle and the rate");
        CHECK(obs.WeakComponents().empty(), "pendulum: no weak component");
        CHECK(obs.Eigenvalues()(0) <= obs.Eigenvalues()(1), "eigenvalues: smallest first");
    }

    // --- a constant velocity measured in position: one step is not enough, two are -------
    {
        Fixture::ConstantVelocity cv(1, 10.);
        Vector x0(2); x0(0) = 0.; x0(1) = 1.;
        const Observability one(cv, 1, &x0), two(cv, 2, &x0);
        CHECK(one.Rank() == 1 && !one.IsObservable(), "constant velocity: one fix leaves a direction open");
        CHECK(two.IsObservable(), "constant velocity: two fixes give the velocity");
    }

    // --- a gain and an amplitude that only appear as their product ---------------------------
    {
        auto gauge = std::make_shared<Gauge>();
        Models::Joint both(gauge, {{"gain", 0.5, 0.}});
        const Observability obs(both, 50);
        CHECK(obs.Rank() == 1 && obs.NWeak() == 1, "product: one direction of two is determined");
        CHECK(obs.Weakness(0) > 0.05 && obs.Weakness(1) > 0.05, "product: the amplitude and the gain share the weak direction");
        const auto weak = obs.WeakComponents();
        CHECK(weak.size() == 2, "product: both are reported");
        // The weak direction leaves gain * a unchanged: gain da + a dgain = 0, in scaled
        // units (sigma_a = 1, sigma_gain = 0.5) the direction is (a sigma_a, -gain sigma_gain)... up to sign.
        const double ratio = obs.Directions()(0, 0) / obs.Directions()(1, 0);
        CHECK_CLOSE(ratio, -(3. * 0.5) / (2. * 1.), 1e-4, "product: the weak direction keeps the product constant");

        // The offset is no better: z = gain a + offset has one number for three unknowns.
        Models::Joint three(gauge, {{"gain", 0.5, 0.}, {"offset", 1., 0.}});
        CHECK(Observability(three, 50).Rank() == 1, "product and offset: still one direction");

        // An offset alone, with the gain known, is as hidden: a + offset / gain.
        Models::Joint shifted(gauge, {{"offset", 1., 0.}});
        CHECK(!Observability(shifted, 50).IsObservable(), "offset: cannot be told from the amplitude");
    }

    // --- the damping of an oscillator is determined by the positions ------------------------------
    {
        // (the fixture pendulum has no declared parameter; the damped pendulum's
        // rate is in its state, which is the point: motion reveals it.)
        Fixture::Pendulum pendulum(1.0, 0.15, 100.);
        Vector rest(2); rest(0) = 0.; rest(1) = 0.;
        CHECK(Observability(pendulum, 100, &rest).IsObservable(), "pendulum at rest: still determined (linear in both)");
    }

    // --- tau: a parameter that returns to where it started ------------------------------------------
    {
        auto gauge = std::make_shared<Gauge>();
        const double dt = 0.1, tau = 2., phi = std::exp(-dt / tau);
        Models::Joint walk(gauge, {{"offset", 1., 0.1, 0.}});
        Models::Joint markov(gauge, {{"offset", 1., 0.1, tau}});
        Vector x(2); x(0) = 3.; x(1) = 0.8;           // the offset started at 0
        CHECK(walk.f(x, Vector(), 0.)(1) == 0.8, "tau = 0: a random walk keeps its value");
        CHECK_CLOSE(markov.f(x, Vector(), 0.)(1), phi * 0.8, 1e-15, "tau: the value decays towards where it started");
        CHECK(walk.Jacobian(x, Vector(), 0.)(1, 1) == 1., "tau = 0: d theta' / d theta = 1");
        CHECK_CLOSE(markov.Jacobian(x, Vector(), 0.)(1, 1), phi, 1e-15, "tau: d theta' / d theta = exp(-dt / tau)");
        Vector w(2); w(0) = 0.; w(1) = 0.05;
        CHECK_CLOSE(markov.f(x, Vector(), w, 0.)(1), phi * 0.8 + 0.05, 1e-15, "tau: plus its noise");

        bool refused = false;
        try { Models::Joint bad(gauge, {{"offset", 1., 0.1, -1.}}); }
        catch (const std::invalid_argument &) { refused = true; }
        CHECK(refused, "refused: a negative tau");

        // With nothing measured the variance of a walk grows without bound; with a tau
        // it settles at drift^2 / (1 - phi^2).
        Engines::Extended engine;
        Vector xw = walk.X0(), xm = markov.X0(), xp;
        Matrix Pw = walk.P0(), Pm = markov.P0(), Pp, F;
        for (int k = 0; k < 2000; ++k)
        {
            engine.Predict(walk, xw, Pw, nullptr, 0., xp, Pp, F); xw = xp; Pw = Pp;
            engine.Predict(markov, xm, Pm, nullptr, 0., xp, Pp, F); xm = xp; Pm = Pp;
        }
        CHECK_CLOSE(Pm(1, 1), 0.01 / (1. - phi * phi), 1e-9, "tau: the variance settles at drift^2 / (1 - phi^2)");
        CHECK_CLOSE(Pw(1, 1), 1. + 2000 * 0.01, 1e-6, "tau = 0: the variance grows by drift^2 per step");
    }

    return Check::Summary("24_observability");
}
