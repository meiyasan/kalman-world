/**
 * \file 22_joint.cc
 * \brief Models::Joint: a declared parameter of another model, estimated with its state.
 */
#include "Kalman/Kalman.h"
#include "Check.h"

#include <cmath>
#include <memory>
#include <stdexcept>

using namespace Kalman;

namespace
{
    /// A damped oscillator, stepped by explicit Euler, whose damping is a declared
    /// parameter. Only the position is measured. Test-local: it carries no physics
    /// worth sharing, only a parameter that leaves a trace in the data.
    class Oscillator : public Models::Extended
    {
        private:
            double fOmega = 2.;
            double fDamping = 0.;

        protected:
            void ParametersChanged() override
            {
                fOmega = Parameter("omega");
                fDamping = Parameter("damping");
            }

        public:
            Oscillator(double omega, double damping, double fs) : Models::Extended("oscillator", 2, 1)
            {
                SetSampleRate(fs);
                SetDifferentiator(DifferentiatorKind::Forward, 1e-6);
                DeclareParameter("omega", omega);
                DeclareParameter("damping", damping);
                ParametersChanged();
                DeclareState({{"position", "m"}, {"velocity", "m/s"}});
                DeclareMeasurement({{"position", "m"}});

                Matrix Q(2, 2, 0.); Q(0, 0) = 1e-6; Q(1, 1) = 1e-4;
                SetQ(Q);
                Matrix R(1, 1, 0.); R(0, 0) = 1e-2;
                SetR(R);
                Matrix H(1, 2, 0.); H(0, 0) = 1.;
                SetH(H);
            }

            Vector f(const Vector &x, const Vector &, double = 0.) const override
            {
                const double dt = DeltaT();
                Vector out(2);
                out(0) = x(0) + dt * x(1);
                out(1) = x(1) + dt * (-fOmega * fOmega * x(0) - fDamping * x(1));
                return out;
            }
    };

    template <class Fn>
    bool Throws(Fn fn)
    {
        try { fn(); }
        catch (const std::exception &) { return true; }
        return false;
    }
}

int main()
{
    const double fs = 50., damping = 0.3;

    // --- construction ------------------------------------------------------------
    auto inner = std::make_shared<Oscillator>(2., 0., fs);       // the filter's guess: no damping
    Vector s0(2); s0(0) = 1.; s0(1) = 0.;
    inner->SetInitialState(s0);
    inner->SetInitialCovariance(Matrix::Identity(2));

    Models::Joint model(inner, {{"damping", 0.5, 1e-4}});
    CHECK(model.NState() == 3 && model.NMeasure() == 1, "joint: n = 2 + 1, m unchanged");
    CHECK(model.NInner() == 2, "joint: the parameters start after the inner state");
    CHECK(model.DefaultEngine() == EngineKind::Extended, "joint: asks for the extended engine");
    CHECK(model.StateComponents()[2].name == "damping", "joint: the parameter is a named component");
    CHECK(model.StateComponents()[0].name == "position", "joint: the inner components keep their names");
    CHECK(model.X0()(0) == 1. && model.X0()(2) == 0., "joint: x0 = inner x0, then the parameter's value");
    CHECK_CLOSE(model.P0()(2, 2), 0.25, 1e-15, "joint: P0 gets sigma^2 for the parameter");
    CHECK(model.P0()(0, 2) == 0., "joint: and no initial correlation");
    model.CheckShapes();
    CHECK(true, "joint: CheckShapes passes");

    // --- refusals ------------------------------------------------------------------
    CHECK(Throws([&] { Models::Joint bad(inner, {{"mass", 1., 0.}}); }), "refused: an undeclared parameter");
    CHECK(Throws([&] { Models::Joint bad(inner, {}); }), "refused: no parameter");
    CHECK(Throws([&] { Models::Joint bad(inner, {{"damping", 0., 0.}}); }), "refused: sigma = 0");
    CHECK(Throws([&] { Models::Joint bad(inner, {{"damping", 1., -1.}}); }), "refused: a negative drift");
    CHECK(Throws([&] { Models::Joint bad(inner, {{"damping", 1., 0.}, {"damping", 1., 0.}}); }),
          "refused: a parameter named twice");
    CHECK(Throws([&] { Models::Joint bad(nullptr, {{"damping", 1., 0.}}); }), "refused: a null model");
    {
        // A declared parameter that nothing reads: the filter could never move it.
        class Idle : public Oscillator
        {
            public:
                Idle() : Oscillator(2., 0., 50.) { DeclareParameter("unused", 1.); }
        };
        auto idle = std::make_shared<Idle>();
        CHECK(Throws([&] { Models::Joint bad(idle, {{"unused", 1., 0.}}); }), "refused: a parameter that acts on nothing");
        CHECK(idle->Parameter("unused") == 1., "refused: and the inner parameter is put back");
    }

    // --- the maps --------------------------------------------------------------------
    Vector x(3); x(0) = 0.4; x(1) = -0.7; x(2) = 0.3;
    const Vector next = model.f(x, Vector(), 0.);
    CHECK(inner->Parameter("damping") == 0., "f: leaves the inner model's parameter as it was");
    {
        Oscillator reference(2., 0.3, fs);
        Vector s(2); s(0) = 0.4; s(1) = -0.7;
        const Vector want = reference.f(s, Vector(), 0.);
        CHECK(next(0) == want(0) && next(1) == want(1), "f: steps the motion with the carried parameter");
        CHECK(next(2) == 0.3, "f: the parameter itself does not move");

        const Matrix J = model.Jacobian(x, Vector(), 0.);
        const Matrix Js = reference.Jacobian(s, Vector(), 0.);
        CHECK(J(0, 0) == Js(0, 0) && J(1, 0) == Js(1, 0) && J(1, 1) == Js(1, 1),
              "Jacobian: the motion block is the inner model's own");
        CHECK_CLOSE(J(1, 2), -(1. / fs) * x(1), 1e-8, "Jacobian: d v_next / d damping = -dt v");
        CHECK(J(0, 2) == 0., "Jacobian: the position does not feel the damping in one step");
        CHECK(J(2, 2) == 1. && J(2, 0) == 0. && J(2, 1) == 0., "Jacobian: the parameter row is [0 0 1]");
        CHECK(inner->Parameter("damping") == 0., "Jacobian: leaves the inner parameter as it was");

        // The inner model differences h too, so dh/ds is 1 to the scheme's accuracy.
        const Matrix H = model.JacobianH(x, 0.);
        CHECK_CLOSE(H(0, 0), 1., 1e-8, "JacobianH: the sensor reads the position");
        CHECK(H(0, 1) == 0. && H(0, 2) == 0., "JacobianH: and does not see the damping");
    }
    {
        const Matrix Q = model.Q(0.);
        CHECK(Q(0, 0) == 1e-6 && Q(1, 1) == 1e-4, "Q: the inner block");
        CHECK_CLOSE(Q(2, 2), 1e-8, 1e-20, "Q: drift^2 for the parameter");
        CHECK(Q(0, 2) == 0. && Q(2, 1) == 0., "Q: block-diagonal");
        CHECK(model.R(0.)(0, 0) == 1e-2, "R: the inner one");
    }
    {
        Vector w(3); w(0) = 0.1; w(1) = 0.2; w(2) = 0.05;
        const Vector noisy = model.f(x, Vector(), w, 0.);
        CHECK_CLOSE(noisy(2), 0.35, 1e-15, "f(x, u, w): the parameter takes its share of the noise");
        CHECK_CLOSE(noisy(0), next(0) + 0.1, 1e-15, "f(x, u, w): the motion takes the rest");
    }

    // --- the sample rate reaches the inner model -------------------------------------------
    {
        auto other = std::make_shared<Oscillator>(2., 0., fs);
        Models::Joint resampled(other, {{"damping", 0.5, 0.}});
        resampled.SetSampleRate(2. * fs);
        resampled.f(x, Vector(), 0.);
        CHECK(other->SampleRate() == 2. * fs, "sample rate: passed on to the inner model before a step");
    }

    // --- the damping is recovered from the positions alone ----------------------------------------
    Oscillator truth(2., damping, fs);
    Sampler sampler(&truth, 20261116);
    const std::size_t N = 3000;          // 60 s, about 19 periods
    sampler.Generate(N, s0);

    Filters::EKF ekf(&model);
    const Record rec = ekf.Forward(sampler.Measurements());
    const double estimate = rec.State(N - 1)(2);
    const double sigma = std::sqrt(rec.Covariance(N - 1)(2, 2));
    std::printf("  damping: estimate %.4f +/- %.4f, truth %.2f; mean NIS %.3f\n", estimate, sigma, damping, ekf.MeanNIS());
    CHECK(std::fabs(estimate - damping) < 3. * sigma, "EKF: the estimate is within 3 sigma of the truth");
    CHECK(sigma < 0.05, "EKF: and the data pin it down (sigma < 0.05 from 0.5)");
    const double nis = ekf.MeanNIS();
    CHECK(nis > 0.7 && nis < 1.4, "EKF: mean NIS consistent with m = 1");
    CHECK(inner->Parameter("damping") == 0., "EKF: the inner model is left as it was found");

    // The same run with the damping known: the joint filter tracks as well.
    Oscillator known(2., damping, fs);
    known.SetInitialState(s0);
    known.SetInitialCovariance(Matrix::Identity(2));
    Filters::EKF reference(&known);
    const Record best = reference.Forward(sampler.Measurements());
    const double e_aug = rec.RMSE(0, sampler.Truth()), e_known = best.RMSE(0, sampler.Truth());
    std::printf("  position RMSE: joint %.4f, damping known %.4f\n", e_aug, e_known);
    CHECK(e_aug < 1.2 * e_known, "EKF: within 20% of the filter that knew the damping");

    // --- the method borrows, the function shares ---------------------------------------------
    {
        Oscillator onStack(2., 0., fs);
        const std::unique_ptr<Model> joint = onStack.Joint({{"damping", 0.5, 1e-4}});
        CHECK(joint->NState() == 3, "model.Joint({...}): borrows a model on the stack");
        CHECK(Kalman::Joint(std::make_shared<Oscillator>(2., 0., fs), {{"damping", 0.5, 1e-4}})->NState() == 3,
              "Joint(model, {...}): shares the model");
    }

    return Check::Summary("22_joint");
}
