/**
 * \file 02_extended.cc
 * \brief Extended engine: nonlinear tracking, numerical Jacobians, KF equivalence.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>

using namespace Kalman;

int main()
{
    // --- the extended engine on a genuinely nonlinear model -----------------
    Fixture::Pendulum pendulum(1.0, 0.15, 100.);
    pendulum.SetNoise(1e-2, 5e-2);

    Vector x0(2); x0(0) = 0.6; x0(1) = 0.;   // ~34 degrees: sin(x) is not x
    pendulum.SetInitialState(x0);

    CHECK(pendulum.DefaultEngine() == EngineKind::Extended, "pendulum: asks for the extended engine");
    CHECK(!pendulum.HasAnalyticJacobian(), "pendulum: declares a numerical Jacobian");

    Sampler sampler(&pendulum, 4242);
    sampler.Generate(1500, x0);

    Filters::EKF ekf(&pendulum);
    const Record rec = ekf.Forward(sampler.Measurements());

    const double nis = ekf.MeanNIS();
    CHECK(nis > 0.6 && nis < 1.8, "EKF: mean NIS consistent with m = 1");

    double eraw = 0.;
    for (std::size_t k = 0; k < 1500; ++k)
        eraw += std::pow(sampler.Measurements()[k](0) - sampler.Truth()[k](0), 2);
    eraw = std::sqrt(eraw / 1500.);
    const double eest = rec.RMSE(0, sampler.Truth());
    CHECK(eest < 0.5 * eraw, "EKF: filtering at least halves the angle error");

    // The velocity is never measured; the filter has to infer it from the dynamics.
    double evel = 0.;
    for (std::size_t k = 500; k < 1500; ++k)
        evel += std::pow(rec.State(k)(1) - sampler.Truth()[k](1), 2);
    CHECK(std::sqrt(evel / 1000.) < 0.3, "EKF: recovers the unobserved angular velocity");

    // --- the Jacobian itself -------------------------------------------------
    {
        Vector x(2); x(0) = 0.4; x(1) = -0.3;
        const Vector u;
        const Matrix J = pendulum.Jacobian(x, u, 0.);

        CHECK(J.Rows() == 2 && J.Cols() == 2, "Jacobian: shape (2, 2)");

        // One RK4 step of dt = 0.01 is I + dt A + O(dt^2), with
        // A = [[0, 1], [-(g/l) cos(x0), -gamma]].
        const double dt = pendulum.DeltaT();
        CHECK_CLOSE(J(0, 0), 1., 5e-3, "Jacobian: (0,0) ~ 1");
        CHECK_CLOSE(J(0, 1), dt, 5e-4, "Jacobian: (0,1) ~ dt");
        CHECK_CLOSE(J(1, 0), -dt * (pendulum.Gravity() / pendulum.Length()) * std::cos(0.4),
                    5e-3, "Jacobian: (1,0) ~ -dt (g/l) cos(theta)");
        CHECK_CLOSE(J(1, 1), 1. - dt * pendulum.Damping(), 5e-3, "Jacobian: (1,1) ~ 1 - dt gamma");
    }

    // --- a linear model must give the same answer under either engine --------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        cv.SetProcessNoise(0.5);
        cv.SetMeasurementNoise(0.2);

        Sampler s(&cv, 777);
        s.Generate(500);

        Filters::KF kf(&cv);
        Filters::EKF ekf_lin(&cv);

        const Record a = kf.Forward(s.Measurements());
        const Record b = ekf_lin.Forward(s.Measurements());

        double maxdiff = 0.;
        for (std::size_t k = 0; k < 500; ++k)
            for (std::size_t i = 0; i < 2; ++i)
                maxdiff = std::max(maxdiff, std::abs(a.State(k)(i) - b.State(k)(i)));
        CHECK(maxdiff < 1e-9, "EKF reduces to KF on a linear model");
    }

    // --- an engine can be swapped on a plain Filter ---------------------------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        Engines::Linear linear;
        Filter generic(&cv, &linear);
        CHECK(generic.GetEngine()->Kind() == EngineKind::Linear, "generic filter: honours the given engine");

        Engines::Extended extended;
        generic.SetEngine(&extended);
        CHECK(generic.GetEngine()->Kind() == EngineKind::Extended, "generic filter: SetEngine swaps it");
    }

    // --- a model asking for an unimplemented kind must fail loudly ----------
    {
        Fixture::ConstantVelocity cv(1, 100.);
        cv.SetDefaultEngine(EngineKind::Unscented);

        bool threw = false;
        try { Filter f(&cv); }
        catch (const std::exception &) { threw = true; }
        CHECK(threw, "resolution: an unimplemented engine kind throws rather than substituting");
    }

    return Check::Summary("extended");
}
