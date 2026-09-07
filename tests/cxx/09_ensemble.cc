/**
 * \file 09_ensemble.cc
 * \brief Ensemble engine: members, inflation, localization, tracking, and no smoother.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <stdexcept>

using namespace Kalman;

namespace
{
    // RMSE of the raw measurement against the truth, over the steps where one arrived.
    double RawRMSE(const Sampler &s, std::size_t channel = 0)
    {
        double e = 0.;
        std::size_t count = 0;
        for (std::size_t k = 0; k < s.NSteps(); ++k)
        {
            const double d = s.Measurements()[k](channel) - s.Truth()[k](channel);
            if (!std::isfinite(d)) continue;              // a dropped channel is no measurement
            e += d * d;
            ++count;
        }
        return std::sqrt(e / static_cast<double>(count));
    }

    double MaxStateDiff(const Record &a, const Record &b)
    {
        double d = 0.;
        for (std::size_t k = 0; k < a.NSteps(); ++k)
            for (std::size_t i = 0; i < a.NState(); ++i)
                d = std::max(d, std::abs(a.State(k)(i) - b.State(k)(i)));
        return d;
    }
}

int main()
{
    // --- construction and the members ------------------------------------------
    {
        bool threw = false;
        try { Engines::Ensemble(1); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "engine: fewer than two members is rejected");
        threw = false;
        try { Engines::Ensemble(10, 0.); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "engine: a non-positive inflation is rejected");

        Fixture::ConstantVelocity cv(1, 100.);
        Vector x0(2); x0(0) = 3.; x0(1) = 1.;
        cv.SetInitialState(x0);
        Matrix P0 = Matrix::Identity(2); P0 *= 0.04;
        cv.SetInitialCovariance(P0);

        Engines::Ensemble enkf(400, 1., 11);
        CHECK(enkf.Members().empty(), "engine: no members before Reset()");
        enkf.Reset(cv);
        CHECK(enkf.Members().size() == 400, "reset: N members");

        Vector mean; Matrix cov;
        enkf.Estimate(mean, cov);
        CHECK_CLOSE(mean(0), 3., 0.03, "reset: ensemble mean near x0");
        CHECK_CLOSE(cov(0, 0), 0.04, 0.01, "reset: sample covariance near P0");
        CHECK_CLOSE(enkf.Spread(), 0.2, 0.03, "reset: spread is the per-state sigma");
        CHECK(!enkf.HasSmoother(), "engine: declares no smoother");
    }

    // --- tracking, against the KF it approximates -----------------------------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        cv.SetProcessNoise(0.5);
        cv.SetMeasurementNoise(0.2);

        Sampler s(&cv, 2024);
        s.Generate(400);
        s.DropChannel(0, 0.1);

        Filters::EnKF enkf(&cv, 200, 1., 7);
        const Record rec = enkf.Forward(s.Measurements());
        CHECK(rec.NSteps() == 400, "EnKF: one estimate per step");

        bool finite = true;
        for (std::size_t k = 0; k < 400 && finite; ++k)
            finite = std::isfinite(rec.State(k)(0)) && std::isfinite(rec.Covariance(k)(0, 0));
        CHECK(finite, "EnKF: finite through the dropped measurements");

        const double eraw = RawRMSE(s);
        const double eest = rec.RMSE(0, s.Truth());
        CHECK(eest < 0.7 * eraw, "EnKF: beats the raw measurement error");

        Filters::KF kf(&cv);
        const Record ref = kf.Forward(s.Measurements());
        CHECK(eest < 1.4 * ref.RMSE(0, s.Truth()), "EnKF (N = 200): within 40% of the KF on a linear model");
        CHECK(enkf.Members().Spread() > 0., "EnKF: the ensemble keeps a spread");

        bool threw = false;
        try { enkf.Backward(); } catch (const std::runtime_error &e)
        {
            threw = std::string(e.what()).find("smoother") != std::string::npos;
        }
        CHECK(threw, "EnKF: Backward() refuses, naming the missing smoother");
    }

    // --- inflation and localization are real, and the seed is honoured --------
    {
        Fixture::ConstantVelocity cv(1, 50.);
        cv.SetProcessNoise(0.5);
        cv.SetMeasurementNoise(0.2);
        Sampler s(&cv, 5);
        s.Generate(150);

        Filters::EnKF a(&cv, 40, 1., 99);
        Filters::EnKF b(&cv, 40, 1., 99);
        Filters::EnKF c(&cv, 40, 1.05, 99);
        const Record ra = a.Forward(s.Measurements());
        const Record rb = b.Forward(s.Measurements());
        const Record rc = c.Forward(s.Measurements());
        CHECK(MaxStateDiff(ra, rb) == 0., "EnKF: the same seed gives the same run");
        CHECK(MaxStateDiff(ra, rc) > 0., "EnKF: inflation changes the analysis");

        // All-ones tapers are a no-op; a zero cross taper kills the update entirely.
        Filters::EnKF d(&cv, 40, 1., 99);
        d.Members().SetLocalization(Matrix(2, 1, 1.), Matrix(1, 1, 1.));
        const Record rd = d.Forward(s.Measurements());
        CHECK(MaxStateDiff(ra, rd) < 1e-12, "EnKF: unit localization tapers change nothing");

        Filters::EnKF e(&cv, 40, 1., 99);
        e.Members().SetLocalization(Matrix(2, 1, 0.), Matrix(1, 1, 1.));
        const Record re = e.Forward(s.Measurements());
        CHECK(re.RMSE(0, s.Truth()) > ra.RMSE(0, s.Truth()), "EnKF: a zero P H^T taper removes the correction");

        Filters::EnKF f(&cv, 40, 1., 99);
        f.Members().SetLocalization(Matrix(3, 1, 1.), Matrix());
        bool threw = false;
        try { f.Forward(s.Measurements()); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "EnKF: a mis-shaped taper is refused");
    }

    // --- a nonlinear model: the pendulum -------------------------------------
    {
        Fixture::Pendulum pendulum(1.0, 0.15, 100.);
        pendulum.SetNoise(1e-2, 5e-2);
        Vector x0(2); x0(0) = 0.6; x0(1) = 0.;
        pendulum.SetInitialState(x0);
        Matrix P0 = Matrix::Identity(2); P0 *= 0.05;
        pendulum.SetInitialCovariance(P0);

        Sampler sampler(&pendulum, 4242);
        sampler.Generate(600, x0);

        Filters::EnKF enkf(&pendulum, 100, 1.02, 3);
        const Record rec = enkf.Forward(sampler.Measurements());
        CHECK(rec.RMSE(0, sampler.Truth()) < 0.7 * RawRMSE(sampler), "EnKF pendulum: reduces the angle error");

        double evel = 0.;
        for (std::size_t k = 200; k < 600; ++k)
            evel += std::pow(rec.State(k)(1) - sampler.Truth()[k](1), 2);
        CHECK(std::sqrt(evel / 400.) < 0.5, "EnKF pendulum: the unobserved velocity is tracked");
    }

    return Check::Summary("09_ensemble");
}
