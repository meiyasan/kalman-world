/**
 * \file 10_integrators.cc
 * \brief Integrators: accuracy ordering, energy, the reserved kind, and a model that steps an ODE.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <stdexcept>

using namespace Kalman;

int main()
{
    const Derivative decay = [](const Vector &x, const Vector &, double) { return -1. * x; };
    const Vector u;

    // --- xdot = -x: RK4 and Dormand-Prince are far more accurate than Euler ---------
    {
        Vector x0(1); x0(0) = 1.;
        const double dt = 0.1, exact = std::exp(-dt);
        const double e = std::abs(Integrators::Euler().Step(decay, x0, u, 0., dt)(0) - exact);
        const double r = std::abs(Integrators::RK4().Step(decay, x0, u, 0., dt)(0) - exact);
        const double d = std::abs(Integrators::DormandPrince().Step(decay, x0, u, 0., dt)(0) - exact);
        CHECK(r < e / 100., "decay: RK4 is 100x more accurate than Euler at dt = 0.1");
        CHECK(d < r + 1e-12, "decay: Dormand-Prince is at least as accurate as RK4");
        CHECK(d < 1e-9, "decay: Dormand-Prince at default tolerances is within 1e-9");

        for (const Integrator *I : {static_cast<const Integrator *>(new Integrators::Euler()),
                                    static_cast<const Integrator *>(new Integrators::RK4()),
                                    static_cast<const Integrator *>(new Integrators::DormandPrince())})
        {
            CHECK(std::abs(I->Step(decay, x0, u, 0., 1e-4)(0) - std::exp(-1e-4)) < 1e-3,
                  "decay: every scheme is fine at a tiny step");
            delete I;
        }
    }

    // --- simple harmonic motion: RK4 keeps the energy, Euler does not ---------------
    {
        const Derivative shm = [](const Vector &s, const Vector &, double) {
            Vector d(2); d(0) = s(1); d(1) = -s(0); return d;
        };
        auto energy = [&](const Integrator &I) {
            Vector s(2); s(0) = 1.; s(1) = 0.;
            s = I.Integrate(shm, s, u, 0., 0.05, 2000);
            return 0.5 * (s(0) * s(0) + s(1) * s(1));
        };
        CHECK(std::abs(energy(Integrators::RK4()) - 0.5) < 1e-3, "SHM: RK4 conserves the energy over 2000 steps");
        CHECK(std::abs(energy(Integrators::Euler()) - 0.5) > 0.1, "SHM: Euler visibly does not");
    }

    // --- Integrate() is n steps; the tableau is the same arithmetic --------------------
    {
        Vector x0(1); x0(0) = 1.;
        const double x = Integrators::RK4().Integrate(decay, x0, u, 0., 0.01, 100)(0);
        CHECK_CLOSE(x, std::exp(-1.), 1e-6, "integrate: 100 x 0.01 reaches e^-1");

        const Tableau tab = Integrators::RK4().GetTableau();
        CHECK(tab.Stages() == 4 && tab.c[1] == 0.5 && tab.b[0] == 1. / 6., "tableau: RK4's (c, A, b)");
        const Vector viaTableau = RungeKuttaStep(tab, [&](const Vector &xx, double tt) { return decay(xx, u, tt); },
                                                 x0, 0., 0.1);
        CHECK_CLOSE(viaTableau(0), Integrators::RK4().Step(decay, x0, u, 0., 0.1)(0), 1e-15,
                    "tableau: the generic RK step reproduces Step()");
        CHECK(Integrators::DormandPrince().GetTableau().Stages() == 7, "tableau: Dormand-Prince exposes its 7 stages");
        CHECK(Integrators::Euler().GetTableau().Stages() == 1, "tableau: Euler is one stage");
    }

    // --- resolution by kind -----------------------------------------------------------
    {
        CHECK(Integrators::Make(IntegratorKind::RK4)->Kind() == IntegratorKind::RK4, "make: RK4");
        CHECK(Integrators::Make(IntegratorKind::DormandPrince)->Kind() == IntegratorKind::DormandPrince, "make: Dormand-Prince");
        bool threw = false;
        std::string what;
        try { Integrators::Make(IntegratorKind::ImplicitEuler); }
        catch (const std::runtime_error &e) { threw = true; what = e.what(); }
        CHECK(threw && what.find("reserved") != std::string::npos, "make: ImplicitEuler is reserved and says so");
        CHECK(std::string(ToString(IntegratorKind::DormandPrince)) == "DormandPrince", "kinds: nameable");
    }

    // --- a model that steps an ODE through its integrator ------------------------------
    {
        Fixture::Pendulum hand(1.0, 0.15, 100.);
        Fixture::PendulumODE ode(1.0, 0.15, 100.);
        CHECK(ode.HasContinuousDynamics() && !hand.HasContinuousDynamics(), "model: the ODE fixture declares continuous dynamics");
        CHECK(ode.DefaultIntegrator() == IntegratorKind::RK4, "model: RK4 unless declared otherwise");
        CHECK(!ode.HasAnalyticJacobian(), "model: continuous dynamics imply a numerical Jacobian");
        CHECK(ode.Describe().find("integrator = RK4") != std::string::npos, "model: Describe() names the integrator");

        Vector x(2); x(0) = 0.6; x(1) = -0.2;
        const Vector a = hand.f(x, u, 0.), b = ode.f(x, u, 0.);
        CHECK_CLOSE(a(0), b(0), 1e-13, "model: RK4 integrator reproduces the hand-written RK4 (angle)");
        CHECK_CLOSE(a(1), b(1), 1e-13, "model: RK4 integrator reproduces the hand-written RK4 (rate)");

        // Substeps advance the same interval, more accurately.
        Fixture::PendulumODE fine(1.0, 0.15, 100.);
        fine.SetSubsteps(4);
        fine.SetIntegrator(IntegratorKind::DormandPrince);
        const Vector c = fine.f(x, u, 0.);
        CHECK(std::abs(c(0) - b(0)) < 1e-6 && std::abs(c(0) - b(0)) > 0., "model: 4 x Dormand-Prince substeps land within 1e-6 of one RK4 step");
        CHECK(fine.Describe().find("DormandPrince x4") != std::string::npos, "model: Describe() shows the substeps");

        Fixture::PendulumODE crude(1.0, 0.15, 100.);
        crude.SetIntegrator(IntegratorKind::Euler);
        const Vector d = crude.f(x, u, 0.);
        CHECK(std::abs(d(1) - b(1)) > 1e-6, "model: Euler differs from RK4 at dt = 0.01");

        // A custom scheme is handed in as an instance.
        Fixture::PendulumODE custom(1.0, 0.15, 100.);
        custom.SetIntegrator(std::make_shared<Integrators::DormandPrince>(1e-10, 1e-12));
        CHECK(custom.DefaultIntegrator() == IntegratorKind::DormandPrince, "model: an instance also sets the declared kind");
        CHECK(&custom.GetIntegrator() != &ode.GetIntegrator(), "model: each model resolves its own integrator");

        bool threw = false;
        try { crude.SetSubsteps(0); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "model: zero substeps is refused");

        // Declaring continuous dynamics without overriding Derivative() fails at the first step.
        Fixture::ConstantVelocity cv(1, 100.);
        cv.UseContinuousDynamics();
        threw = false;
        try { cv.f(Vector(2, 0.), u, 0.); } catch (const std::runtime_error &e)
        { threw = std::string(e.what()).find("Derivative") != std::string::npos; }
        CHECK(threw, "model: a missing Derivative() override is named, not silently zero");

        // A model states what it is: discrete, or continuous (to be decoded).
        CHECK(hand.Dynamics() == DynamicsKind::Discrete, "model: a hand-stepped model is discrete");
        CHECK(ode.Dynamics() == DynamicsKind::Continuous, "model: a model with a derivative is continuous");
        {
            // Linear, stated in continuous time: dx/dt = A x with white acceleration.
            Fixture::ConstantVelocity stated(1, 4.);          // 4 Hz, so dt = 0.25
            Matrix A(2, 2, 0.), Qc(1, 1, 0.), Lc(2, 1, 0.);
            A(0, 1) = 1.;
            Qc(0, 0) = 2.;
            Lc(1, 0) = 1.;
            stated.SetGenerator(A);
            stated.SetNoiseIntensity(Qc, Lc);
            CHECK(stated.Dynamics() == DynamicsKind::Continuous, "model: a generator makes the model continuous");
            CHECK_CLOSE(stated.F(0.)(0, 1), 0.25, 1e-14, "model: F decoded from the generator");
            CHECK_CLOSE(stated.Q(0.)(0, 0), 2. * 0.25 * 0.25 * 0.25 / 3., 1e-14, "model: Q decoded from the intensity");
            stated.SetSampleRate(8.);                         // decoded again for the new rate
            CHECK_CLOSE(stated.F(0.)(0, 1), 0.125, 1e-14, "model: F follows the sample rate");
            CHECK_CLOSE(stated.Q(0.)(1, 1), 2. * 0.125, 1e-14, "model: Q follows the sample rate");

            // Nonlinear: the intensity is decoded with the derivative's Jacobian.
            Fixture::PendulumODE noisy(1.0, 0.15, 100.);
            Matrix Qp(2, 2, 0.);
            Qp(1, 1) = 0.5;
            noisy.SetNoiseIntensity(Qp);
            const Matrix Qd = noisy.Q(0.);
            CHECK_CLOSE(Qd(1, 1), 0.5 * 0.01, 1e-4, "model: nonlinear, the leading term is intensity times dt");
            CHECK(Qd(0, 0) > 0. && Qd(0, 0) < 1e-6, "model: nonlinear, the angle gets its dt^3 share through the Jacobian");
        }

        // The whole filter runs on the ODE model exactly as on the hand-stepped one.
        Vector x0(2); x0(0) = 0.6; x0(1) = 0.;
        hand.SetNoise(1e-2, 5e-2); ode.SetNoise(1e-2, 5e-2);
        hand.SetInitialState(x0); ode.SetInitialState(x0);
        Sampler s(&hand, 4242);
        s.Generate(800, x0);
        Filters::EKF e1(&hand), e2(&ode);
        const Record r1 = e1.Forward(s.Measurements()), r2 = e2.Forward(s.Measurements());
        double maxdiff = 0.;
        for (std::size_t k = 0; k < 800; ++k)
            maxdiff = std::max(maxdiff, std::abs(r1.State(k)(0) - r2.State(k)(0)));
        CHECK(maxdiff < 1e-9, "filter: the EKF on the ODE model matches the hand-stepped model");

        Filters::UKF ukf(&ode);
        const Record r3 = ukf.Forward(s.Measurements());
        CHECK(r3.RMSE(0, s.Truth()) < 1.25 * r2.RMSE(0, s.Truth()), "filter: the UKF drives the ODE model too");
    }

    return Check::Summary("10_integrators");
}
