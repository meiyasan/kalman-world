/**
 * \file 13_controllers.cc
 * \brief Controllers: the closed loop, P / PID / LQR / MPPI, the actuator hook, the adapters.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <random>
#include <stdexcept>

using namespace Kalman;

namespace
{
    /**
     * \brief Plant-in-the-loop: the *truth* answers to the command.
     *
     * Filter::Forward(zk, controller) takes measurements recorded up front, so it
     * cannot show regulation of a real plant; this loop can. Each step: the
     * controller commands from the belief, the truth moves under f(x, u) plus
     * process noise, a measurement is drawn, and the KF step corrects the belief.
     * Returns (mean |position| over the second half, the controller's total cost).
     */
    std::pair<double, double> Simulate(Fixture::ControlledVelocity &plant, Controller &ctrl,
                                       std::size_t steps, std::uint64_t seed = 1)
    {
        std::mt19937_64 rng(seed);
        std::normal_distribution<double> gauss(0., 1.);
        Engines::Linear engine;
        const Matrix Qf = Utils::CholeskyFactor(plant.Q()), Rf = Utils::CholeskyFactor(plant.R());

        Vector x = plant.X0(), truth = plant.X0(), nu;
        Matrix P = plant.P0();
        const Vector *lastNu = nullptr;
        double t = 0., late = 0.;
        for (std::size_t k = 0; k < steps; ++k)
        {
            const Vector u = ctrl(x, P, t, lastNu);
            truth = plant.f(truth, u, t);
            Vector w(2); w(0) = gauss(rng); w(1) = gauss(rng);
            truth += Qf * w;
            Vector z = plant.h(truth, t);
            Vector v(1); v(0) = gauss(rng);
            z += Rf * v;

            Vector xp; Matrix Pp, F, S;
            engine.Predict(plant, x, P, &u, t, xp, Pp, F);
            engine.Innovation(plant, xp, Pp, z, t, nu, S);
            engine.Update(plant, xp, Pp, z, t, x, P);
            lastNu = &nu;
            t += plant.DeltaT();
            if (k >= steps / 2) late += std::abs(truth(0));
        }
        return {late / static_cast<double>(steps - steps / 2), ctrl.TotalCost()};
    }
}

int main()
{
    const double fs = 20.;

    // --- the contract ------------------------------------------------------------
    {
        Fixture::ControlledVelocity plant(fs);
        CHECK(plant.NControl() == 1 && plant.B().Rows() == 2 && plant.B().Cols() == 1, "fixture: one acceleration input");

        Matrix Kp(1, 2, 0.);
        Kp(0, 0) = 4.; Kp(0, 1) = 3.;
        Controllers::Proportional p(&plant, Kp);
        CHECK(p.NControl() == 1, "controller: command dimension");
        CHECK(p.XRef().Size() == 2 && p.XRef()(0) == 0., "controller: default set point is the origin");
        CHECK(p.QCost()(0, 0) == 1. && p.RCost()(0, 0) == 1., "controller: identity costs by default");

        Vector x(2); x(0) = 1.; x(1) = 0.5;
        const Vector u = p(x, Matrix::Identity(2), 0.);
        CHECK_CLOSE(u(0), -(4. * 1. + 3. * 0.5), 1e-12, "proportional: u = -Kp e");
        CHECK(p.CostLog().size() == 1 && p.ControlLog().size() == 1, "controller: one log entry per call");
        CHECK_CLOSE(p.CostLog()[0].second, 1. + 0.25 + u(0) * u(0), 1e-12, "controller: cost = e^T Q e + u^T R u");
        p.Reset();
        CHECK(p.CostLog().empty(), "controller: Reset() clears the logs");

        bool threw = false;
        try { Controllers::Proportional bad(&plant, Matrix(2, 2, 1.)); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "controller: a mis-shaped gain is refused");
        threw = false;
        try { Controllers::Proportional(&plant, Kp, Vector(3, 0.)); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "controller: a mis-sized set point is refused");
    }

    // --- the actuator hook: cost against what is delivered ---------------------------
    {
        Fixture::ControlledVelocity limited(fs, 0.5);
        Matrix Kp(1, 2, 0.); Kp(0, 0) = 10.;
        Controllers::Proportional p(&limited, Kp);
        Vector x(2); x(0) = 1.; x(1) = 0.;
        const Vector u = p(x, Matrix::Identity(2), 0.);
        CHECK_CLOSE(u(0), -10., 1e-12, "clip: the raw command is what the policy asked");
        CHECK_CLOSE(p.CostLog()[0].second, 1. + 0.25, 1e-12, "clip: the cost charges the clipped 0.5, not the raw 10");
        CHECK_CLOSE(limited.ClipControl(u)(0), -0.5, 1e-12, "clip: the plant delivers the limit");
    }

    // --- the closed loop drives the belief to the set point --------------------------
    {
        Fixture::ControlledVelocity plant(fs);
        plant.SetProcessNoise(0.1); plant.SetMeasurementNoise(0.05);
        Vector x0(2); x0(0) = 2.; x0(1) = 0.;
        plant.SetInitialState(x0);

        Vector ref(2); ref(0) = 0.; ref(1) = 0.;
        Matrix Kp(1, 2, 0.); Kp(0, 0) = 2.; Kp(0, 1) = 2.5;
        Controllers::Proportional p(&plant, Kp, ref);

        Sampler s(&plant, 3);
        s.Generate(200);                                   // truth stays near x0: no control in the truth
        Filters::KF kf(&plant);
        const Record rec = kf.Forward(s.Measurements(), p);
        CHECK(rec.NSteps() == 200, "loop: one estimate per step");
        CHECK(kf.Controls().size() == 200 && kf.Controls()[0].Size() == 1, "loop: a command per step is kept");
        CHECK(p.ControlLog().size() == 200, "loop: the controller logged every call");
        CHECK(kf.Controls()[5](0) == p.ControlLog()[5].second(0), "loop: the filter applied what the controller returned");

        // The command is fed to the *prediction*, so the prior follows the commanded
        // dynamics: prior_k = F post_{k-1} + B u_k.
        const Vector expect = plant.F() * rec.State(9) + plant.B() * kf.Controls()[10];
        CHECK_CLOSE(kf.PriorStates()[10](0), expect(0), 1e-12, "loop: the prior carries the command");

        bool threw = false;
        Fixture::ControlledVelocity other(fs);
        Controllers::Proportional foreign(&other, Kp);
        try { kf.Forward(s.Measurements(), foreign); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "loop: a controller built for another model is refused");
    }

    // --- PID ------------------------------------------------------------------------
    {
        Fixture::ControlledVelocity plant(fs);
        Matrix Kp(1, 2, 0.), Ki(1, 2, 0.), Kd(1, 2, 0.);
        Kp(0, 0) = 1.; Ki(0, 0) = 0.5; Kd(0, 0) = 0.1;
        Controllers::PID pid(&plant, Kp, Ki, Kd);
        Vector x(2); x(0) = 1.; x(1) = 0.;
        const Vector u1 = pid(x, Matrix::Identity(2), 0.);
        const double dt = plant.DeltaT();
        // First call: e = 1, integral = dt, de = 1/dt (from zero).
        CHECK_CLOSE(u1(0), -(1. + 0.5 * dt + 0.1 / dt), 1e-12, "PID: first command from P, I and D terms");
        const Vector u2 = pid(x, Matrix::Identity(2), dt);
        CHECK_CLOSE(u2(0), -(1. + 0.5 * 2. * dt + 0.), 1e-12, "PID: the integral accumulates, the derivative vanishes");
        pid.Reset();
        const Vector u3 = pid(x, Matrix::Identity(2), 0.);
        CHECK_CLOSE(u3(0), u1(0), 1e-12, "PID: Reset() clears the integral and the previous error");

        Controllers::PID ponly(&plant, Kp);
        CHECK_CLOSE(ponly(x, Matrix::Identity(2), 0.)(0), -1., 1e-12, "PID: empty Ki / Kd mean P-only");
    }

    // --- LQR --------------------------------------------------------------------------
    {
        Fixture::ControlledVelocity plant(fs);
        Controllers::LQR lqr(&plant);
        CHECK(!lqr.IsFitted(), "LQR: not fitted before Fit()");
        bool threw = false;
        try { lqr(Vector(2, 1.), Matrix::Identity(2), 0.); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "LQR: use before Fit() is refused");

        lqr.Fit();
        CHECK(lqr.IsFitted() && lqr.Gain().Rows() == 1 && lqr.Gain().Cols() == 2, "LQR: the gain is (l, n)");
        CHECK(lqr.Gain()(0, 0) > 0. && lqr.Gain()(0, 1) > 0., "LQR: positive gains on position and velocity");

        // The DARE fixed point is a fixed point: plugging P back changes nothing.
        const Matrix A = plant.F(), B = plant.B();
        const Matrix P = Controllers::LQR::SolveDARE(A, B, Matrix::Identity(2), Matrix::Identity(1));
        Matrix S = B.Transpose() * P * B; S += Matrix::Identity(1);
        Matrix rhs = A.Transpose() * P * A;
        rhs -= (A.Transpose() * P * B) * (Utils::Inverse(S) * (B.Transpose() * P * A));
        rhs += Matrix::Identity(2);
        double resid = 0.;
        for (std::size_t i = 0; i < 4; ++i) resid = std::max(resid, std::abs(rhs.Data()[i] - P.Data()[i]));
        CHECK(resid < 1e-8, "LQR: SolveDARE() returns a fixed point of the Riccati equation");

        // The closed loop under LQR is stable: the eigenvalues of F - B K lie inside the unit circle.
        const Matrix Acl = A - B * lqr.Gain();
        const double tr = Acl(0, 0) + Acl(1, 1), det = Acl(0, 0) * Acl(1, 1) - Acl(0, 1) * Acl(1, 0);
        CHECK(std::abs(det) < 1. && std::abs(tr) < 1. + det, "LQR: F - B K is Schur stable");

        // ... and on a real plant it regulates, at a lower cost than a weak proportional gain.
        Vector x0(2); x0(0) = 1.; x0(1) = 0.;
        Fixture::ControlledVelocity a(fs), b(fs);
        for (Fixture::ControlledVelocity *m : {&a, &b})
        {
            m->SetInitialState(x0);
            m->SetProcessNoise(0.05);
            m->SetMeasurementNoise(0.05);
        }
        Controllers::LQR la(&a); la.Fit();
        Matrix Kp(1, 2, 0.); Kp(0, 0) = 0.2; Kp(0, 1) = 0.1;
        Controllers::Proportional pb(&b, Kp);
        const auto ra = Simulate(a, la, 200), rb = Simulate(b, pb, 200);
        CHECK(ra.first < 0.05, "LQR: the plant is held within 0.05 of the set point");
        CHECK(ra.first < rb.first && ra.second < rb.second, "LQR: closer and cheaper than a weak proportional gain");

        Fixture::ConstantVelocity uncontrolled(1, fs);
        Controllers::LQR none(&uncontrolled);
        threw = false;
        try { none.Fit(); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "LQR: a plant without control cannot be fitted");
    }

    // --- MPPI ---------------------------------------------------------------------------
    {
        // 5 Hz, a 20-step (4 s) horizon, unit exploration noise and a cheap control:
        // the setting in which a sampled planner can see the effect of its command.
        const double fsSlow = 5.;
        Fixture::ControlledVelocity plant(fsSlow, 1.);
        Vector x0(2); x0(0) = 1.; x0(1) = 0.;
        plant.SetInitialState(x0);
        const Matrix Rc(1, 1, 0.01);
        Controllers::MPPI mppi(&plant, 64, 20, 1., 1., 2, 5, Vector(), Matrix(), Rc);
        CHECK(mppi.Nominal().size() == 20, "MPPI: a warm-start sequence of `horizon` commands");

        const Vector u = mppi(x0, Matrix::Identity(2), 0.);
        CHECK(u.Size() == 1 && u(0) < -0.3, "MPPI: from x = +1 the first command decelerates hard");
        CHECK(std::abs(u(0)) <= 1. + 1e-12, "MPPI: the candidates were clipped to the actuator limit");

        // Determinism by seed; Reset() restarts the stream and the nominal.
        Controllers::MPPI again(&plant, 64, 20, 1., 1., 2, 5, Vector(), Matrix(), Rc);
        CHECK(again(x0, Matrix::Identity(2), 0.)(0) == u(0), "MPPI: the same seed gives the same command");
        mppi.Reset();
        CHECK(mppi(x0, Matrix::Identity(2), 0.)(0) == u(0), "MPPI: Reset() restarts the stream and the nominal");

        // On a real plant it regulates.
        Fixture::ControlledVelocity plant2(fsSlow, 1.);
        plant2.SetInitialState(x0);
        plant2.SetProcessNoise(0.05); plant2.SetMeasurementNoise(0.05);
        Controllers::MPPI m2(&plant2, 64, 20, 1., 1., 2, 5, Vector(), Matrix(), Rc);
        const auto r = Simulate(plant2, m2, 100);
        CHECK(r.first < 0.15, "MPPI: the plant is held near the set point (mean |x| < 0.15 over the second half)");
    }

    // --- the adapters -----------------------------------------------------------------
    {
        // Boost: measurements far from the model's expectation drive NIS up; R shrinks.
        Fixture::ControlledVelocity plant(fs);
        plant.SetMeasurementNoise(0.01);
        Controllers::Boost boost(&plant, 3., 2., 5);
        const Matrix R0 = plant.R();
        std::vector<Vector> z(40, Vector(1, 5.));            // a jump the model does not expect
        Filters::KF kf(&plant);
        kf.Forward(z, boost);
        CHECK(!boost.ScaleLog().empty(), "boost: the scale was logged");
        bool shrank = false;
        for (const auto &kv : boost.ScaleLog()) shrank = shrank || kv.second < 1.;
        CHECK(shrank, "boost: R was scaled down while the filter was surprised");
        boost.Reset();
        CHECK_CLOSE(plant.R()(0, 0), R0(0, 0), 1e-15, "boost: Reset() restores the nominal R");

        // AdaptiveQR: innovations larger than S mean Q is too small; lambda > 1.
        Fixture::ControlledVelocity plant2(fs);
        plant2.SetProcessNoise(0.01); plant2.SetMeasurementNoise(0.05);
        const Matrix Q0 = plant2.Q();
        Controllers::AdaptiveQR aqr(&plant2, 10, 0.1, 10.);
        Sampler s(&plant2, 4);
        s.Generate(80);
        std::vector<Vector> noisy = s.Measurements();
        std::mt19937_64 rng(7);
        std::normal_distribution<double> big(0., 0.5);    // ten times the declared sigma
        for (Vector &v : noisy) v(0) += big(rng);
        Filters::KF kf2(&plant2);
        kf2.Forward(noisy, aqr);
        CHECK(!aqr.ScaleLog().empty(), "adaptive: lambda was logged once the window filled");
        CHECK(aqr.ScaleLog().back().second > 1., "adaptive: Q scaled up when the innovations exceed S");
        CHECK(aqr.ScaleLog().back().second <= 10., "adaptive: ... and clipped to q_max");
        aqr.Reset();
        CHECK_CLOSE(plant2.Q()(1, 1), Q0(1, 1), 1e-15, "adaptive: Reset() restores the nominal Q");
    }

    return Check::Summary("13_controllers");
}
