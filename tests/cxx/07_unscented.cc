/**
 * \file 07_unscented.cc
 * \brief Unscented engine: KF equivalence on a linear model, nonlinear tracking, gaps, smoother.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <stdexcept>

using namespace Kalman;

int main()
{
    // --- the transform itself ------------------------------------------------
    {
        Engines::Unscented ut(1e-3, 2., 0.);
        double gamma;
        std::vector<double> wm, wc;
        ut.Weights(2, gamma, wm, wc);
        CHECK(wm.size() == 5 && wc.size() == 5, "weights: 2n + 1 of each");
        double sum = 0.;
        for (double w : wm) sum += w;
        CHECK_CLOSE(sum, 1., 1e-8, "weights: mean weights sum to one (1e6-sized terms cancel)");
        CHECK_CLOSE(wc[0] - wm[0], 1. - 1e-6 + 2., 1e-8, "weights: Wc0 - Wm0 = 1 - alpha^2 + beta");

        Vector mean(2); mean(0) = 1.; mean(1) = -2.;
        Matrix P = Matrix::Identity(2); P(0, 0) = 4.;
        const std::vector<Vector> pts = Engines::Unscented::SigmaPoints(mean, P, gamma);
        CHECK(pts.size() == 5, "sigma points: 2n + 1");
        CHECK(pts[0](0) == 1. && pts[0](1) == -2., "sigma points: the first is the mean");
        CHECK_CLOSE(pts[1](0) - mean(0), gamma * 2., 1e-12, "sigma points: +gamma sqrt(P) along axis 0");
        CHECK_CLOSE(pts[3](0) - mean(0), -gamma * 2., 1e-12, "sigma points: -gamma sqrt(P) along axis 0");
        CHECK_CLOSE(pts[2](1) - mean(1), gamma * 1., 1e-12, "sigma points: +gamma along axis 1");

        bool threw = false;
        try { Engines::Unscented(0., 2., 0.); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "engine: alpha = 0 is rejected");
    }

    // --- a linear model gives the KF answer ----------------------------------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        cv.SetProcessNoise(0.5);
        cv.SetMeasurementNoise(0.2);

        Sampler s(&cv, 777);
        s.Generate(500);
        s.DropChannel(0, 0.1);

        Filters::KF kf(&cv);
        Filters::UKF ukf(&cv);
        const Record a = kf.Forward(s.Measurements());
        const Record b = ukf.Forward(s.Measurements());

        double maxdiff = 0., maxPdiff = 0.;
        for (std::size_t k = 0; k < 500; ++k)
            for (std::size_t i = 0; i < 2; ++i)
            {
                maxdiff = std::max(maxdiff, std::abs(a.State(k)(i) - b.State(k)(i)));
                maxPdiff = std::max(maxPdiff, std::abs(a.Covariance(k)(i, i) - b.Covariance(k)(i, i)));
            }
        CHECK(maxdiff < 1e-6, "UKF reduces to KF on a linear model (states)");
        CHECK(maxPdiff < 1e-6, "UKF reduces to KF on a linear model (covariances)");
        CHECK(ukf.GetEngine()->Kind() == EngineKind::Unscented, "UKF: names its engine");
        CHECK_CLOSE(ukf.MeanNIS(), kf.MeanNIS(), 1e-4, "UKF: same NIS as the KF, dropped channels included");

        // The RTS pass runs with the identity transition the engine reports.
        const Record sa = kf.Backward();
        const Record sb = ukf.Backward();
        CHECK(sb.Flavor() == StateFlavor::Smoothed, "UKF: Backward() produces a smoothed record");
        CHECK(sb.RMSE(0, s.Truth()) <= b.RMSE(0, s.Truth()), "UKF: smoothing does not worsen the estimate");
        CHECK(sa.RMSE(0, s.Truth()) < 1.5 * sb.RMSE(0, s.Truth()) + 1e-9,
              "UKF: identity-transition smoother stays close to the KF's");
    }

    // --- the pendulum, where sin(x) is not x ---------------------------------
    {
        Fixture::Pendulum pendulum(1.0, 0.15, 100.);
        pendulum.SetNoise(1e-2, 5e-2);
        Vector x0(2); x0(0) = 0.6; x0(1) = 0.;
        pendulum.SetInitialState(x0);

        Sampler sampler(&pendulum, 4242);
        sampler.Generate(1500, x0);

        Filters::UKF ukf(&pendulum);
        const Record rec = ukf.Forward(sampler.Measurements());

        const double nis = ukf.MeanNIS();
        CHECK(nis > 0.6 && nis < 1.8, "UKF pendulum: mean NIS consistent with m = 1");

        double eraw = 0.;
        for (std::size_t k = 0; k < 1500; ++k)
            eraw += std::pow(sampler.Measurements()[k](0) - sampler.Truth()[k](0), 2);
        eraw = std::sqrt(eraw / 1500.);
        CHECK(rec.RMSE(0, sampler.Truth()) < 0.5 * eraw, "UKF pendulum: at least halves the angle error");

        double evel = 0.;
        for (std::size_t k = 500; k < 1500; ++k)
            evel += std::pow(rec.State(k)(1) - sampler.Truth()[k](1), 2);
        CHECK(std::sqrt(evel / 1000.) < 0.3, "UKF pendulum: recovers the unobserved angular velocity");

        // Against the EKF on the same data: same ballpark, no Jacobian involved.
        Filters::EKF ekf(&pendulum);
        const Record ref = ekf.Forward(sampler.Measurements());
        const double r1 = rec.RMSE(0, sampler.Truth()), r2 = ref.RMSE(0, sampler.Truth());
        CHECK(std::abs(r1 - r2) < 0.25 * r2, "UKF pendulum: RMSE within 25% of the EKF's");
    }

    // --- resolution through the model's declared default ----------------------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        cv.SetDefaultEngine(EngineKind::Unscented);
        Filter f(&cv);
        CHECK(f.GetEngine()->Kind() == EngineKind::Unscented, "resolution: a model may ask for the unscented engine");
        CHECK(std::string(ToString(EngineKind::Combinatorial)) == "Combinatorial", "kinds: Combinatorial is nameable");

        cv.SetDefaultEngine(EngineKind::Combinatorial);
        Filter g(&cv);
        CHECK(g.GetEngine()->Kind() == EngineKind::Combinatorial, "resolution: the Combinatorial kind builds its engine too");
    }

    return Check::Summary("07_unscented");
}
