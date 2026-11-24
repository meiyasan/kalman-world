/**
 * \file 13_controllers.cc
 * \brief Controllers: the closed loop, P / PID / LQR / MPC / MPPI, the actuator hook, the adapters.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <algorithm>
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
        CHECK_CLOSE(limited.LimitControl(u)(0), -0.5, 1e-12, "clip: the plant delivers the limit");
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

        // Fit(A, B) against a given linearisation is Fit() on the same matrices; SetGain() is obeyed.
        Controllers::LQR given(&plant);
        given.Fit(A, B);
        CHECK_CLOSE(given.Gain()(0, 0), lqr.Gain()(0, 0), 1e-14, "LQR: Fit(A, B) on the model's F, B is Fit()");
        Matrix K(1, 2, 0.); K(0, 0) = 2.; K(0, 1) = 3.;
        given.SetGain(K);
        CHECK_CLOSE(given(Vector{1., 1.}, Matrix::Identity(2), 0.)(0), -5., 1e-14, "LQR: SetGain() is the gain used");
        threw = false;
        try { given.SetGain(Matrix(2, 2, 1.)); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "LQR: a gain of the wrong shape is refused");

        Fixture::ConstantVelocity uncontrolled(1, fs);
        Controllers::LQR none(&uncontrolled);
        threw = false;
        try { none.Fit(); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "LQR: a plant without control cannot be fitted");
    }

    // --- MPC ----------------------------------------------------------------------------
    {
        const double fsSlow = 5.;
        const Matrix Rc(1, 1, 0.01);
        const Matrix I2 = Matrix::Identity(2);

        // The cost of a noise-free run from x0: sum of e^T e + 0.01 u^2, u as delivered.
        const auto rollout = [&](Fixture::ControlledVelocity &plant, Controller &ctrl, Vector x, std::size_t steps,
                                 double *peak = nullptr, double *jump = nullptr) {
            double cost = 0., last = 0.;
            for (std::size_t k = 0; k < steps; ++k)
            {
                const Vector u = plant.LimitControl(ctrl.Policy(x, I2, static_cast<double>(k) * plant.DeltaT(), nullptr));
                if (peak) *peak = std::max(*peak, std::abs(u(0)));
                if (jump && k > 0) *jump = std::max(*jump, std::abs(u(0) - last));
                last = u(0);
                cost += x(0) * x(0) + x(1) * x(1) + 0.01 * u(0) * u(0);
                x = plant.f(x, u, 0.);
            }
            return cost;
        };

        // With no limit in the way the first command is the LQR command, whatever the horizon.
        Fixture::ControlledVelocity plant(fsSlow);
        Controllers::LQR lqr(&plant, Vector(), Matrix(), Rc);
        lqr.Fit();
        Vector x0(2); x0(0) = 1.; x0(1) = -0.3;
        const double uLqr = lqr.Policy(x0, I2, 0., nullptr)(0);
        for (std::size_t horizon : {1, 5, 25})
        {
            Controllers::MPC mpc(&plant, horizon, Vector(), Vector(), Vector(), 1, Vector(), Matrix(), Rc);
            CHECK_CLOSE(mpc.Policy(x0, I2, 0., nullptr)(0), uLqr, 1e-9, "MPC: with no limit, the LQR command (any horizon)");
            CHECK(mpc.Sweeps() == 0 && mpc.ActiveLimits() == 0, "MPC: and no limit was active");
        }
        {
            Controllers::MPC mpc(&plant, 10, Vector(), Vector(), Vector(), 1, Vector(), Matrix(), Rc);
            mpc.Policy(x0, I2, 0., nullptr);
            CHECK(mpc.Plan().size() == 10 && mpc.Predicted().size() == 11, "MPC: a plan of `horizon` commands and the states it leads through");
            const Vector next = plant.f(x0, mpc.Plan()[0], 0.);
            CHECK(mpc.Predicted()[1](0) == next(0) && mpc.Predicted()[1](1) == next(1), "MPC: the prediction is the model's own f");
            Controllers::MPC again(&plant, 10, Vector(), Vector(), Vector(), 1, Vector(), Matrix(), Rc);
            CHECK(again.Policy(x0, I2, 0., nullptr)(0) == mpc.Plan()[0](0), "MPC: the same input gives the same command");
            const double first = mpc.Plan()[0](0);
            mpc.Reset();
            CHECK(mpc.Policy(x0, I2, 0., nullptr)(0) == first, "MPC: Reset() forgets the plan");
        }

        // A limit on the command is a constraint of the plan.
        Fixture::ControlledVelocity limited(fsSlow, 1.);
        Vector far(2); far(0) = 6.; far(1) = 0.;
        Vector lo(1), hi(1); lo(0) = -1.; hi(0) = 1.;
        Controllers::MPC mpc(&limited, 40, lo, hi, Vector(), 1, Vector(), Matrix(), Rc);
        const Vector u0 = mpc.Policy(far, I2, 0., nullptr);
        bool inside = true;
        for (const Vector &u : mpc.Plan()) inside = inside && u(0) >= -1. && u(0) <= 1.;
        CHECK(inside && u0(0) == -1., "MPC: every planned command is inside the limits; the first is on one");
        CHECK(mpc.ActiveLimits() > 0 && mpc.Sweeps() > 0, "MPC: the limit was active in the plan");

        mpc.Reset();
        Controllers::LQR clipped(&limited, Vector(), Matrix(), Rc);
        clipped.Fit();
        double peak = 0.;
        const double costMpc = rollout(limited, mpc, far, 150, &peak), costLqr = rollout(limited, clipped, far, 150);
        std::printf("  from x = 6 under |u| <= 1: cost MPC %.2f, clipped LQR %.2f\n", costMpc, costLqr);
        CHECK(peak <= 1. + 1e-12, "MPC: no command beyond the limit over the run");
        CHECK(costMpc < costLqr, "MPC: planning with the limit beats LQR clipped after the fact");

        // A limit on how fast the command may change.
        Vector rate(1); rate(0) = 0.2;
        Controllers::MPC smooth(&limited, 40, lo, hi, rate, 1, Vector(), Matrix(), Rc);
        double jump = 0.;
        rollout(limited, smooth, far, 80, nullptr, &jump);
        CHECK(jump <= 0.2 + 1e-9, "MPC: the command changes by at most duMax per step");
        bool planSmooth = true;
        for (std::size_t k = 1; k < smooth.Plan().size(); ++k)
            planSmooth = planSmooth && std::abs(smooth.Plan()[k](0) - smooth.Plan()[k - 1](0)) <= 0.2 + 1e-9;
        CHECK(planSmooth, "MPC: and so does the plan");

        // A nonlinear plant: a cart with quadratic drag, linearised along the plan.
        class Cart : public Models::Extended
        {
            public:
                Cart() : Models::Extended("cart", 2, 1, 1)
                {
                    SetSampleRate(10.);
                    SetDifferentiator(DifferentiatorKind::Central);
                    Matrix H(1, 2, 0.); H(0, 0) = 1.; SetH(H);
                }
                Vector f(const Vector &x, const Vector &u, double = 0.) const override
                {
                    const double dt = DeltaT(), a = (u.Empty() ? 0. : u(0)) - 0.5 * x(1) * std::abs(x(1));
                    Vector out(2);
                    out(0) = x(0) + dt * x(1);
                    out(1) = x(1) + dt * a;
                    return out;
                }
        };
        Cart cart;
        Vector lim2(1), lim2n(1); lim2(0) = 2.; lim2n(0) = -2.;
        Controllers::MPC drive(&cart, 30, lim2n, lim2, Vector(), 2, Vector(), Matrix(), Rc);
        Vector xc(2); xc(0) = 3.; xc(1) = 0.;
        double peakCart = 0.;
        for (std::size_t k = 0; k < 120; ++k)
        {
            const Vector u = drive.Policy(xc, I2, 0., nullptr);
            peakCart = std::max(peakCart, std::abs(u(0)));
            xc = cart.f(xc, u, 0.);
        }
        std::printf("  nonlinear cart after 12 s: x = %.4f, v = %.4f\n", xc(0), xc(1));
        CHECK(std::abs(xc(0)) < 0.02 && std::abs(xc(1)) < 0.02, "MPC: a nonlinear plant is brought to the set point");
        CHECK(peakCart <= 2. + 1e-12, "MPC: within its limits");

        // Refusals.
        bool threw = false;
        try { Fixture::ConstantVelocity none(1, fsSlow); Controllers::MPC bad(&none); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "MPC: a plant without control is refused");
        threw = false;
        try { Controllers::MPC bad(&plant, 0); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "MPC: a zero horizon is refused");
        threw = false;
        try { Controllers::MPC bad(&plant, 5, hi, lo); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "MPC: uMin above uMax is refused");
        threw = false;
        try { Controllers::MPC bad(&plant, 5, Vector(2, 0.), Vector()); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "MPC: limits of the wrong size are refused");
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

        // The caller's draws stand in for the controller's own: two controllers with
        // different seeds given the same draws make the same decision; a wrong count throws.
        {
            Controllers::MPPI p(&plant, 16, 5, 1., 1., 2, 1, Vector(), Matrix(), Rc), q(&plant, 16, 5, 1., 1., 2, 99, Vector(), Matrix(), Rc);
            CHECK(p.NDraws() == 2 * 16 * 5, "MPPI: NDraws() is iterations x rollouts x horizon x l");
            std::mt19937_64 g(3);
            std::normal_distribution<double> n01(0., 1.);
            std::vector<double> draws(p.NDraws());
            for (double &d : draws) d = n01(g);
            p.SetDraws(draws);
            q.SetDraws(draws);
            CHECK(p(x0, Matrix::Identity(2), 0.)(0) == q(x0, Matrix::Identity(2), 0.)(0),
                  "MPPI: the same draws give the same command whatever the seed");
            bool wrong = false;
            try { p.SetDraws(std::vector<double>(7, 0.)); } catch (const std::invalid_argument &) { wrong = true; }
            CHECK(wrong, "MPPI: a wrong number of draws is refused");
        }

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
        Tuners::Boost boost(&plant, 3., 2., 5);
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
        Tuners::AdaptiveQR aqr(&plant2, 10, 0.1, 10.);
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

    // --- the Riccati and Stein solvers (Kalman/Algebra/Riccati.h) ----------------------
    {
        // Scalar DARE: p = q + a^2 p - a^2 b^2 p^2 / (r + b^2 p), a quadratic in p.
        const double a = 1.1, b = 0.5, q = 2., r = 3.;
        const double A2 = b * b, B2 = r - a * a * r - q * b * b, C2 = -q * r;   // A2 p^2 + B2 p + C2 = 0
        const double p = (-B2 + std::sqrt(B2 * B2 - 4. * A2 * C2)) / (2. * A2);
        const Matrix P = Algebra::SolveDARE(Matrix(1, 1, a), Matrix(1, 1, b), Matrix(1, 1, q), Matrix(1, 1, r));
        CHECK_CLOSE(P(0, 0), p, 1e-12 * p, "Riccati: the scalar DARE's positive root");

        // A pole at 0.999: the plain recursion stopped 6e-8 short; Newton reaches rounding.
        Matrix As(2, 2, 0.); As(0, 0) = 0.999; As(0, 1) = 0.01; As(1, 1) = 0.999;
        Matrix Bs(2, 1, 0.); Bs(1, 0) = 0.01;
        const Matrix Rs(1, 1, 1e3);
        const Matrix Ps = Algebra::SolveDARE(As, Bs, Matrix::Identity(2), Rs);
        Matrix next = As.Transpose() * Ps * As;
        next -= (As.Transpose() * Ps * Bs) * Algebra::RiccatiGain(As, Bs, Rs, Ps);
        next += Matrix::Identity(2);
        double resid = 0., size = 0.;
        for (std::size_t i = 0; i < 4; ++i)
        {
            resid = std::max(resid, std::abs(next.Data()[i] - Ps.Data()[i]));
            size = std::max(size, std::abs(Ps.Data()[i]));
        }
        CHECK(resid < 1e-14 * size, "Riccati: the residual is at rounding for a slow pole");

        // Stein: X = A^T X A + W, checked by substitution; a divergent sum is refused.
        Matrix W = Matrix::Identity(2);
        const Matrix X = Algebra::SolveStein(As, W);
        Matrix back = As.Transpose() * X * As;
        back += W;
        double e = 0.;
        for (std::size_t i = 0; i < 4; ++i) e = std::max(e, std::abs(back.Data()[i] - X.Data()[i]));
        CHECK(e < 1e-12 * X(0, 0), "Stein: the doubling sum solves the equation");
        bool threw = false;
        try { Algebra::SolveStein(Matrix(1, 1, 1.5), Matrix(1, 1, 1.)); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Stein: an unstable A is refused");

        Matrix Au(2, 2, 0.); Au(0, 0) = 1.2; Au(1, 1) = 0.5;
        Matrix Bu(2, 1, 0.); Bu(1, 0) = 1.;
        threw = false;
        try { Algebra::SolveDARE(Au, Bu, Matrix::Identity(2), Matrix::Identity(1)); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Riccati: an unstabilisable pair is refused, not returned as NaN");
    }

    // --- a callback model's actuator limit ---------------------------------------------
    {
        struct Plant
        {
            static int Step(void *, const double *x, const double *u, double, double *out)
            { out[0] = x[0] + (u ? u[0] : 0.); return 0; }
            static int Measure(void *, const double *x, double, double *z) { z[0] = x[0]; return 0; }
            static int Clip(void *, const double *u, double *out) { out[0] = std::max(-0.5, std::min(0.5, u[0])); return 0; }
        };
        Models::Callback plant("clipped", 1, 1, 1, &Plant::Step, &Plant::Measure);
        CHECK_CLOSE(plant.LimitControl(Vector(1, 3.))(0), 3., 0., "callback: no clip function, no limit");
        plant.SetLimit(&Plant::Clip);
        CHECK_CLOSE(plant.LimitControl(Vector(1, 3.))(0), 0.5, 0., "callback: SetLimit() is the actuator limit");
        Controllers::Proportional p(&plant, Matrix(1, 1, 4.));
        p(Vector(1, 1.), Matrix::Identity(1), 0.);
        CHECK_CLOSE(p.CostLog().back().second, 1. + 0.25, 1e-15, "callback: the cost is charged on the clipped command");
    }

    // --- a box on the command: Model::SetControlBounds -------------------------------------
    {
        Models::Callback plant("boxed", 1, 1, 1, [](void *, const double *x, const double *u, double, double *out)
                               { out[0] = x[0] + (u ? u[0] : 0.); return 0; },
                               [](void *, const double *x, double, double *z) { z[0] = x[0]; return 0; });
        plant.SetControlBounds(Vector(1, -0.5), Vector(1, 0.25));
        CHECK_CLOSE(plant.LimitControl(Vector(1, 3.))(0), 0.25, 0., "box: a callback model with no clip function clips to its box");
        CHECK_CLOSE(plant.LimitControl(Vector(1, -3.))(0), -0.5, 0., "box: the lower bound");
        CHECK_CLOSE(plant.LimitControl(Vector(1, 0.1))(0), 0.1, 0., "box: inside, the command is unchanged");
        Controllers::Proportional p(&plant, Matrix(1, 1, 4.));
        p(Vector(1, 1.), Matrix::Identity(1), 0.);
        CHECK_CLOSE(p.CostLog().back().second, 1. + 0.25, 1e-15, "box: the cost is charged on the clipped command");
        bool threw = false;
        try { plant.LimitControl(Vector(2, 0.)); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "box: a command of another size is refused");
        threw = false;
        try { plant.SetControlBounds(Vector(1, 1.), Vector(1, 0.)); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "box: a lower bound above the upper one is refused");
        plant.SetControlBounds(Vector(), Vector());
        CHECK_CLOSE(plant.LimitControl(Vector(1, 3.))(0), 3., 0., "box: removed, no limit");
    }

    // --- a tuner is its own role; a filter takes a controller and a tuner -------------------------
    {
        Fixture::ControlledVelocity plant(fs);
        Vector x0(2); x0(0) = 1.; x0(1) = 0.;
        plant.SetInitialState(x0);
        Tuners::AdaptiveQR tune(&plant, 10);
        CHECK(dynamic_cast<Tuner *>(&tune) != nullptr && dynamic_cast<Controller *>(&tune) != nullptr,
              "tuner: a Tuner, which is a Controller that commands nothing");
        CHECK(tune.Policy(x0, Matrix::Identity(2), 0., nullptr)(0) == 0., "tuner: its command is zero");

        Controllers::LQR lqr(&plant);
        lqr.Fit();
        std::vector<Vector> zk(80, Vector(1, 0.5));
        Filter filter(&plant);
        const Record both = filter.Forward(zk, lqr, tune);
        CHECK(both.NSteps() == 80 && lqr.ControlLog().size() == 80, "Forward(zk, controller, tuner): the controller commands every step");
        CHECK(!tune.ScaleLog().empty(), "Forward(zk, controller, tuner): the tuner retuned");
        Fixture::ControlledVelocity other(fs);
        Tuners::AdaptiveQR foreign(&other, 10);
        bool threw = false;
        try { filter.Forward(zk, lqr, foreign); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Forward(zk, controller, tuner): a tuner built for another model is refused");
    }

    return Check::Summary("13_controllers");
}
