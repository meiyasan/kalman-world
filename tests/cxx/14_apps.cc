/**
 * \file 14_apps.cc
 * \brief libKalmanApps: the application models, their parameters and their filters.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Apps/Ballistics/Ballistic.h"
#include "Kalman/Apps/Weather/Storm.h"
#include "Check.h"

#include <cmath>
#include <stdexcept>

using namespace Kalman;

int main()
{
    // --- registration: linking the library is enough to name its models -------
    {
        const Factory &factory = Factory::Instance();
        CHECK(factory.Has("storm") && factory.Has("storm-track-only"), "apps: the storm models are registered");
        CHECK(factory.Has("ballistic") && factory.Has("ballistic-order2"), "apps: the ballistic models are registered");
        const Factory::Entry entry = factory.Describe("storm");
        CHECK(entry.n == 5 && entry.m == 3 && entry.engine == EngineKind::Extended,
              "apps: storm is (n, m) = (5, 3) on the extended engine");
        CHECK(factory.Describe("storm-track-only").m == 2, "apps: the track-only storm observes the centre alone");
        CHECK(factory.Describe("ballistic-order2").n == 6, "apps: the order-2 ballistic model has six states");
    }

    // --- named parameters ------------------------------------------------------
    {
        Apps::Weather::Storm storm;
        CHECK(storm.ParameterNames().size() == 4, "parameters: storm declares four");
        CHECK(storm.HasParameter("omega") && !storm.HasParameter("basin"), "parameters: known and unknown names");
        CHECK_CLOSE(storm.Parameter("beta_drift"), 5., 0., "parameters: defaults are readable");
        storm.SetParameter("omega", -0.022);
        CHECK_CLOSE(storm.Parameter("omega"), -0.022, 0., "parameters: a set value reads back");

        bool threw = false;
        try { storm.SetParameter("basin", 1.); }
        catch (const std::out_of_range &) { threw = true; }
        CHECK(threw, "parameters: an unknown name throws");
    }

    // --- storm dynamics ---------------------------------------------------------
    {
        Apps::Weather::Storm north;
        Vector x(5);
        x(0) = 135.; x(1) = 15.; x(2) = -20.; x(3) = 6.; x(4) = 80.;
        const Vector straight = north.f(x, Vector(), 0.);
        CHECK(straight(0) < x(0) && straight(1) > x(1), "storm: a straight NH run heads west and poleward");
        CHECK_CLOSE(straight(2), x(2), 0., "storm: a straight run keeps its eastward velocity");
        CHECK_CLOSE(straight(4), x(4), 0., "storm: the intensity persists");

        // One step by hand: 6 h of (v + beta) on the sphere.
        const double rad = 3.14159265358979323846 / 180.;
        const double be = -5. * std::cos(20. * rad), bn = 5. * std::sin(20. * rad);
        CHECK_CLOSE(straight(1), 15. + (6. + bn) * 6. / 110.574, 1e-12, "storm: latitude step matches the formula");
        CHECK_CLOSE(straight(0), 135. + (-20. + be) * 6. / (111.320 * std::cos(15. * rad)), 1e-12,
                    "storm: longitude step matches the formula");

        Apps::Weather::Storm south;
        south.SetParameter("hemisphere", -1.);
        Vector xs(x); xs(1) = -15.; xs(3) = -6.;
        const Vector mirrored = south.f(xs, Vector(), 0.);
        CHECK_CLOSE(mirrored(1), -straight(1), 1e-12, "storm: the southern hemisphere mirrors the latitude");
        CHECK_CLOSE(mirrored(0), straight(0), 1e-12, "storm: and keeps the longitude");

        north.SetParameter("omega", -0.022);
        const Vector turned = north.f(x, Vector(), 0.);
        const double speed0 = std::hypot(x(2), x(3)), speed1 = std::hypot(turned(2), turned(3));
        CHECK_CLOSE(speed1, speed0, 1e-12, "storm: a turn rotates the velocity without changing the speed");
        CHECK(turned(3) > x(3), "storm: NH recurvature turns the motion poleward");

        Apps::Weather::Storm track(false);
        CHECK(track.h(x, 0.).Size() == 2, "storm: the track-only model returns the centre fix");
        CHECK(north.h(x, 0.)(2) == 80., "storm: the full model also observes the intensity");
    }

    // --- storm: the filter beats the raw fixes and stays consistent ----------------
    {
        Apps::Weather::Storm storm;
        Vector x0(5);
        x0(0) = 138.; x0(1) = 12.; x0(2) = -20.; x0(3) = 7.; x0(4) = 90.;
        storm.SetInitialState(x0);
        storm.SetInitialCovariance(Matrix::Identity(5));

        Sampler sampler(&storm, 2026);
        sampler.Generate(300, x0);

        Filters::EKF ekf(&storm);
        const Record rec = ekf.Forward(sampler.Measurements());

        double eraw = 0.;
        for (std::size_t k = 0; k < 300; ++k)
            eraw += std::pow(sampler.Measurements()[k](1) - sampler.Truth()[k](1), 2);
        eraw = std::sqrt(eraw / 300.);
        CHECK(rec.RMSE(1, sampler.Truth()) < eraw, "storm EKF: the latitude estimate beats the raw fix");
        const double nis = ekf.MeanNIS();
        CHECK(nis > 2.0 && nis < 4.2, "storm EKF: mean NIS consistent with m = 3");
    }

    // --- ballistic dynamics -------------------------------------------------------
    {
        Apps::Ballistics::Ballistic ball;
        ball.SetSampleRate(100.);
        Vector x(9);
        x(1) = 30.; x(4) = 10.; x(7) = 120.;                       // velocities, from the origin
        const Vector next = ball.f(x, Vector(), 0.);
        const double dt = 0.01;
        CHECK_CLOSE(next(0), 30. * dt, 1e-12, "ballistic: x advances by v dt without drag");
        CHECK_CLOSE(next(6), 120. * dt - 0.5 * 9.81 * dt * dt, 1e-12, "ballistic: z falls by g dt^2 / 2");
        CHECK_CLOSE(next(7), 120. - 9.81 * dt, 1e-12, "ballistic: v_z loses g dt");
        CHECK_CLOSE(next(8), -9.81, 0., "ballistic: the acceleration state is the modelled one");

        ball.SetParameter("c_drag", 0.01);
        const Vector dragged = ball.f(x, Vector(), 0.);
        CHECK(dragged(1) < next(1) && dragged(7) < next(7), "ballistic: drag slows every moving axis");

        // Orbital mode: a circular orbit keeps its radius over a short step.
        ball.SetParameter("c_drag", 0.);
        ball.SetParameter("surface_gravity", 0.);
        const double r0 = 6.771e6, v0 = std::sqrt(6.674e-11 * 5.972e24 / r0);
        Vector orbit(9);
        orbit(0) = r0; orbit(4) = v0;
        const Vector moved = ball.f(orbit, Vector(), 0.);
        const double r1 = std::hypot(moved(0), moved(3));
        CHECK_CLOSE(r1, r0, 1e-3, "ballistic: one step of a circular orbit keeps the radius to a millimetre");
        CHECK_CLOSE(moved(2), -6.674e-11 * 5.972e24 / (r0 * r0), 1e-9, "ballistic: orbital gravity is G M / r^2 inward");

        Apps::Ballistics::Ballistic two(2);
        CHECK(two.NState() == 6 && two.h(Vector(6), 0.).Size() == 3, "ballistic: order 2 has six states, three observed");

        bool threw = false;
        try { Apps::Ballistics::Ballistic bad(4); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "ballistic: an order other than 2 or 3 throws");
    }

    // --- the Jacobian is the forward difference the Python framework takes ----------
    {
        Apps::Weather::Storm storm;
        storm.SetParameter("omega", -0.022);
        Vector x(5);
        x(0) = 135.; x(1) = 15.; x(2) = -20.; x(3) = 6.; x(4) = 80.;
        const Matrix J = storm.Jacobian(x, Vector(), 0.);
        const Vector f0 = storm.f(x, Vector(), 0.);
        Vector xp(x);
        const double step = 1e-6 * 15.;
        xp(1) += step;
        const Vector f1 = storm.f(xp, Vector(), 0.);
        CHECK_CLOSE(J(0, 1), (f1(0) - f0(0)) / step, 0., "jacobian: forward difference with step 1e-6 max(1, |x|)");
        CHECK_CLOSE(J(4, 4), 1., 1e-9, "jacobian: the intensity carries over");
        const Matrix H = storm.JacobianH(x, 0.);
        CHECK(H.Rows() == 3 && H.Cols() == 5, "jacobian: dh/dx is (m, n)");
        CHECK_CLOSE(H(2, 4), 1., 1e-9, "jacobian: the intensity is observed directly");
    }

    return Check::Summary("14_apps");
}
