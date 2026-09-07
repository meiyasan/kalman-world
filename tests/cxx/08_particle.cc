/**
 * \file 08_particle.cc
 * \brief Particle engine: the cloud, resampling, tracking, determinism, and no smoother.
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
}

int main()
{
    // --- construction and the cloud -------------------------------------------
    {
        bool threw = false;
        try { Engines::Particle(1); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "engine: fewer than two particles is rejected");
        threw = false;
        try { Engines::Particle(10, 1.5); } catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "engine: a threshold outside [0, 1] is rejected");

        Fixture::ConstantVelocity cv(1, 100.);
        Vector x0(2); x0(0) = 3.; x0(1) = 1.;
        cv.SetInitialState(x0);
        Matrix P0 = Matrix::Identity(2); P0 *= 0.01;
        cv.SetInitialCovariance(P0);

        Engines::Particle pf(2000, 0.5, 11);
        CHECK(pf.Particles().empty(), "engine: no cloud before Reset()");
        pf.Reset(cv);
        CHECK(pf.Particles().size() == 2000 && pf.Weights().size() == 2000, "reset: N particles, N weights");
        CHECK_CLOSE(pf.EffectiveSampleSize(), 2000., 1e-9, "reset: uniform weights, ESS = N");

        Vector mean; Matrix cov;
        pf.Estimate(mean, cov);
        CHECK_CLOSE(mean(0), 3., 0.01, "reset: cloud mean near x0 (position)");
        CHECK_CLOSE(mean(1), 1., 0.01, "reset: cloud mean near x0 (velocity)");
        CHECK_CLOSE(cov(0, 0), 0.01, 0.002, "reset: cloud covariance near P0");
        CHECK(pf.HasSmoother() == false, "engine: declares no smoother");
    }

    // --- tracking a constant-velocity target ----------------------------------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        cv.SetProcessNoise(0.5);
        cv.SetMeasurementNoise(0.2);

        Sampler s(&cv, 2024);
        s.Generate(400);
        s.DropChannel(0, 0.1);

        Filters::PF pf(&cv, 1000, 0.5, 7);
        const Record rec = pf.Forward(s.Measurements());
        CHECK(rec.NSteps() == 400, "PF: one estimate per step");

        bool finite = true;
        for (std::size_t k = 0; k < 400 && finite; ++k)
            finite = std::isfinite(rec.State(k)(0)) && std::isfinite(rec.Covariance(k)(0, 0));
        CHECK(finite, "PF: finite through the dropped measurements");

        const double eraw = RawRMSE(s);
        const double eest = rec.RMSE(0, s.Truth());
        CHECK(eest < 0.7 * eraw, "PF: beats the raw measurement error");

        Filters::KF kf(&cv);
        const Record ref = kf.Forward(s.Measurements());
        CHECK(eest < 1.6 * ref.RMSE(0, s.Truth()), "PF: within 60% of the optimal KF on a linear model");

        CHECK(pf.Cloud().Resamples() > 0, "PF: the cloud was resampled at least once");
        CHECK(pf.Cloud().EffectiveSampleSize() > 0., "PF: a positive effective sample size");

        bool threw = false;
        try { pf.Backward(); } catch (const std::runtime_error &e)
        {
            threw = std::string(e.what()).find("smoother") != std::string::npos;
        }
        CHECK(threw, "PF: Backward() refuses, naming the missing smoother");
    }

    // --- the seed makes a run reproducible -------------------------------------
    {
        Fixture::ConstantVelocity cv(1, 50.);
        Sampler s(&cv, 5);
        s.Generate(100);

        Filters::PF a(&cv, 300, 0.5, 99);
        Filters::PF b(&cv, 300, 0.5, 99);
        Filters::PF c(&cv, 300, 0.5, 100);
        const Record ra = a.Forward(s.Measurements());
        const Record rb = b.Forward(s.Measurements());
        const Record rc = c.Forward(s.Measurements());

        double same = 0., diff = 0.;
        for (std::size_t k = 0; k < 100; ++k)
        {
            same = std::max(same, std::abs(ra.State(k)(0) - rb.State(k)(0)));
            diff = std::max(diff, std::abs(ra.State(k)(0) - rc.State(k)(0)));
        }
        CHECK(same == 0., "PF: the same seed gives the same run");
        CHECK(diff > 0., "PF: a different seed gives a different run");

        // A second Forward() on the same filter reseeds the cloud from x0 / P0 (the
        // random stream carries on, so the draws differ, but the run is complete
        // and its bookkeeping starts over).
        const std::size_t before = a.Cloud().Resamples();
        const Record ra2 = a.Forward(s.Measurements());
        CHECK(ra2.NSteps() == 100 && a.Cloud().Particles().size() == 300, "PF: Forward(reset) rebuilds the cloud");
        CHECK(a.Cloud().Resamples() <= before, "PF: Forward(reset) restarts the resample count");
        CHECK(std::isfinite(ra2.RMSE(0, s.Truth())), "PF: the second run is a valid run");
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

        Filters::PF pf(&pendulum, 2000, 0.5, 3);
        const Record rec = pf.Forward(sampler.Measurements());
        const double eraw = RawRMSE(sampler);
        CHECK(rec.RMSE(0, sampler.Truth()) < 0.7 * eraw, "PF pendulum: reduces the angle error");

        double evel = 0.;
        for (std::size_t k = 200; k < 600; ++k)
            evel += std::pow(rec.State(k)(1) - sampler.Truth()[k](1), 2);
        CHECK(std::sqrt(evel / 400.) < 0.5, "PF pendulum: the unobserved velocity is tracked");
    }

    return Check::Summary("08_particle");
}
