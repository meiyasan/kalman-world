/**
 * \file 01_linear.cc
 * \brief Linear filter + RTS smoother: consistency, dropouts, multi-axis.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <limits>

using namespace Kalman;

int main()
{
    Fixture::ConstantVelocity model(1, 100.);
    model.SetProcessNoise(0.5);
    model.SetMeasurementNoise(0.2);

    Vector x0(2); x0(0) = 0.; x0(1) = 1.;
    model.SetInitialState(x0);

    Sampler sampler(&model, 12345);
    sampler.Generate(2000, x0);

    Filters::KF filter(&model);
    const Record fwd = filter.Forward(sampler.Measurements());

    CHECK(fwd.NSteps() == 2000, "forward: one record entry per measurement");
    CHECK(fwd.NState() == 2 && fwd.NMeasure() == 1, "forward: record carries (n, m) = (2, 1)");
    CHECK(fwd.Flavor() == StateFlavor::Posterior, "forward: record tagged Posterior");

    // A consistent filter has NIS ~ m and NEES ~ n. These are the checks that
    // actually catch a wrong Q: the estimates can look fine on a plot while the
    // covariance is badly scaled.
    const double nis = filter.MeanNIS();
    CHECK(nis > 0.6 && nis < 1.6, "forward: mean NIS consistent with m = 1");

    const double nees = filter.MeanNEES(sampler.Truth());
    CHECK(nees > 1.2 && nees < 3.2, "forward: mean NEES consistent with n = 2");

    // Filtering must beat the raw measurement.
    double eraw = 0.;
    for (std::size_t k = 0; k < 2000; ++k)
        eraw += std::pow(sampler.Measurements()[k](0) - sampler.Truth()[k](0), 2);
    eraw = std::sqrt(eraw / 2000.);
    const double efilt = fwd.RMSE(0, sampler.Truth());
    CHECK(efilt < eraw, "forward: filtered position beats the raw measurement");

    // --- RTS smoother -------------------------------------------------------
    const Record sm = filter.Backward();
    CHECK(sm.Flavor() == StateFlavor::Smoothed, "backward: record tagged Smoothed");

    const double esm = sm.RMSE(0, sampler.Truth());
    CHECK(esm < efilt, "backward: smoothing strictly improves on filtering");

    bool tighter = true;
    for (std::size_t k = 0; k < 2000; ++k)
        if (sm.Covariance(k)(0, 0) > fwd.Covariance(k)(0, 0) * (1. + 1e-9)) tighter = false;
    CHECK(tighter, "backward: smoothed covariance never exceeds the filtered one");

    CHECK_CLOSE(sm.State(1999)(0), fwd.State(1999)(0), 1e-12,
                "backward: last smoothed state equals the last posterior");

    // --- missing measurements ----------------------------------------------
    {
        Sampler gappy(&model, 999);
        gappy.Generate(500, x0);
        gappy.DropChannel(0, 0.3);

        Filters::KF kf(&model);
        const Record rec = kf.Forward(gappy.Measurements());

        bool finite = true;
        for (std::size_t k = 0; k < rec.NSteps(); ++k)
            if (!std::isfinite(rec.State(k)(0)) || !std::isfinite(rec.Covariance(k)(0, 0)))
                finite = false;
        CHECK(finite, "dropout: 30% missing leaves every estimate finite");
    }
    {
        // An entirely missing measurement is a pure prediction, so uncertainty grows.
        std::vector<Vector> blind;
        for (int k = 0; k < 20; ++k)
        {
            Vector z(1);
            z(0) = std::numeric_limits<double>::quiet_NaN();
            blind.push_back(z);
        }
        Filters::KF kf(&model);
        const Record rec = kf.Forward(blind);
        CHECK(rec.Covariance(19)(0, 0) > rec.Covariance(0)(0, 0),
              "dropout: an all-NaN run grows the covariance");
        CHECK(std::isnan(kf.NIS()[0]), "dropout: an unmeasured step reports NaN NIS");
        CHECK(std::isnan(kf.MeanNIS()), "dropout: a fully blind run has no mean NIS");
    }

    // --- 2-D model -----------------------------------------------------------
    {
        Fixture::ConstantVelocity plane(2, 50.);
        plane.SetProcessNoise(0.3);
        plane.SetMeasurementNoise(0.1);
        CHECK(plane.NState() == 4 && plane.NMeasure() == 2, "2-D: state (px,vx,py,vy)");

        Sampler s2(&plane, 5);
        s2.Generate(300);

        Filters::KF kf(&plane);
        const Record rec = kf.Forward(s2.Measurements());
        CHECK(rec.NSteps() == 300, "2-D: runs");
        CHECK(kf.MeanNIS() > 1.2 && kf.MeanNIS() < 3.2, "2-D: mean NIS consistent with m = 2");
    }

    // --- the record's derived series ----------------------------------------
    {
        const std::vector<double> series = fwd.Series(0);
        const std::vector<double> sigma = fwd.Sigma(0);
        CHECK(series.size() == 2000 && sigma.size() == 2000, "record: series span the run");
        CHECK(sigma[1999] > 0., "record: sigma is positive");
        CHECK(sigma[1999] < sigma[0], "record: uncertainty shrinks as evidence accumulates");
        CHECK_CLOSE(fwd.Time(0), 0., 1e-12, "record: starts at t0");
        CHECK_CLOSE(fwd.Time(100), 1.0, 1e-9, "record: time advances at 1/fs");
    }

    return Check::Summary("linear");
}
