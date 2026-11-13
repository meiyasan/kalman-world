/**
 * \file 14_apps.cc
 * \brief libKalmanApps: the application models, their parameters and their filters.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Apps/Astro/RocketAscent.h"
#include "Kalman/Apps/Astro/RocketDescent.h"
#include "Kalman/Apps/Astro/RocketLanding.h"
#include "Kalman/Apps/Astro/RocketLiftoff.h"
#include "Kalman/Apps/Ballistics/Ballistic.h"
#include "Kalman/Apps/Vehicles/CarLane.h"
#include "Kalman/Apps/Vehicles/CarSpeed.h"
#include "Kalman/Apps/Weather/Storm.h"
#include "Kalman/Apps/Weather/Tornado.h"
#include "Check.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

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
        CHECK(factory.Has("tornado-environment") && factory.Has("car-speed"),
              "apps: the tornado and car-speed models are registered");
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

    // --- tornado environment: relaxation toward the regime mean ------------------------
    {
        Apps::Weather::Tornado tornado;
        tornado.SetParameter("mu", 3.);
        tornado.SetParameter("persistence", 0.5);
        Vector x(1);
        x(0) = 1.;
        CHECK_CLOSE(tornado.f(x, Vector(), 0.)(0), 2., 1e-15, "tornado: one step closes half the gap to mu");
        x(0) = 3.;
        CHECK_CLOSE(tornado.f(x, Vector(), 0.)(0), 3., 0., "tornado: the equilibrium is a fixed point");
        CHECK_CLOSE(tornado.h(x, 0.)(0), 3., 0., "tornado: the latent is observed directly");
    }

    // --- car speed: grip-limited launch, power-limited run, a top speed ----------------
    {
        Apps::Vehicles::CarSpeed car;
        CHECK_CLOSE(car.Acceleration(0.), 1. * 9.81 - 0.015 * 9.81, 1e-12,
                    "car: off the line the grip sets the acceleration (mu g minus rolling)");
        CHECK(car.Acceleration(40.) < car.Acceleration(10.), "car: power-limited acceleration falls with speed");
        const double hand = 150000. / (1. * 1500. * 9.81);          // P / (mu m g)
        CHECK(car.Acceleration(hand - 1.) > car.Acceleration(hand + 1.), "car: the hand-off sits at P / (mu m g)");

        Vector x(2);
        for (int k = 0; k < 20 * 600; ++k) x = car.f(x, Vector(), 0.);   // ten minutes at 20 Hz
        const double top = x(1);
        CHECK(std::fabs(car.Acceleration(top)) < 1e-6, "car: the run ends at the speed where the forces balance");
        CHECK(top > 60. && top < 80., "car: a 150 kW car tops out between 216 and 288 km/h");

        Vector back(2);
        back(1) = 0.;
        car.SetParameter("grade", 1.2);                              // a wall of a hill,
        car.SetParameter("mu", 0.2);                                 // and too little grip to climb it
        CHECK_CLOSE(car.f(back, Vector(), 0.)(1), 0., 0., "car: the speed is clamped at zero, never negative");
    }

    // --- car lane and the rocket family: registered, named, continuous, finite ---------
    {
        const Factory &factory = Factory::Instance();
        struct Expected { const char *name; std::size_t n, m, l; std::vector<std::string> state, measured; };
        const std::vector<Expected> expected = {
            {"car-lane", 3, 1, 1, {"lateral", "heading", "speed"}, {"lateral"}},
            {"rocket-liftoff", 3, 1, 0, {"altitude", "velocity", "mass"}, {"altitude"}},
            {"rocket-ascent", 4, 2, 2, {"x", "y", "vx", "vy"}, {"x", "y"}},
            {"rocket-descent", 7, 3, 0, {"x", "y", "vx", "vy", "theta", "omega", "mass"}, {"x", "y", "theta"}},
            {"rocket-descent-control", 7, 3, 2, {"x", "y", "vx", "vy", "theta", "omega", "mass"}, {"x", "y", "theta"}},
            {"rocket-landing", 3, 1, 0, {"altitude", "velocity", "mass"}, {"altitude"}},
        };
        for (const Expected &e : expected)
        {
            const std::string what = std::string(e.name) + ": ";
            CHECK(factory.Has(e.name), (what + "registered").c_str());
            if (!factory.Has(e.name)) continue;
            const std::unique_ptr<Model> model = factory.Create(e.name);
            CHECK(model->NState() == e.n && model->NMeasure() == e.m && model->NControl() == e.l,
                  (what + "(n, m, l) as the Python class").c_str());
            CHECK(model->Dynamics() == DynamicsKind::Continuous, (what + "continuous dynamics").c_str());
            const auto state = model->StateComponents(), measured = model->MeasurementComponents();
            bool names = state.size() == e.state.size() && measured.size() == e.measured.size();
            for (std::size_t i = 0; names && i < state.size(); ++i) names = state[i].name == e.state[i];
            for (std::size_t i = 0; names && i < measured.size(); ++i) names = measured[i].name == e.measured[i];
            CHECK(names, (what + "component names as the Python enums").c_str());

            const Vector x = model->X0();
            Vector u(e.l, 0.);
            const Vector next = model->f(x, u, 0.);
            bool finite = next.Size() == e.n;
            for (std::size_t i = 0; finite && i < e.n; ++i) finite = std::isfinite(next(i));
            CHECK(finite, (what + "one step from the initial state is finite").c_str());
        }
    }

    // --- car lane: the steering is clipped, Euler steps by default ---------------------
    {
        Apps::Vehicles::CarLane lane;
        Vector x(3);
        x(1) = 0.01; x(2) = 15.;
        Vector u(1);
        u(0) = 1.;                                                   // far past the 0.1 rad limit
        const Vector d = lane.Derivative(x, u, 0.);
        CHECK_CLOSE(d(1), 15. * std::tan(0.1) / 2.7, 1e-15, "lane: the steering command is clipped to steer_limit");
        const Vector next = lane.f(x, u, 0.);
        CHECK_CLOSE(next(0), 15. * std::sin(0.01) / 50., 1e-15, "lane: one explicit Euler step per sample");
    }

    // --- liftoff: thrust over weight, then burnout -----------------------------------
    {
        Apps::Astro::RocketLiftoff rocket;
        Vector x = rocket.X0();
        CHECK_CLOSE(rocket.Derivative(x, Vector(), 0.)(1), 2500. * 8. / 1000. - 9.80665, 1e-12,
                    "liftoff: on the pad the acceleration is T / m - g");
        for (int k = 0; k < 10 * 100; ++k) x = rocket.f(x, Vector(), 0.);   // 100 s at 10 Hz
        CHECK_CLOSE(x(2), 400., 0., "liftoff: after the burn the mass is the dry mass");
        CHECK(rocket.Derivative(x, Vector(), 0.)(1) < -9., "liftoff: after burnout the rocket coasts");
    }

    // --- landing: free fall above the profile, a burn below it, at rest on the pad --
    {
        Apps::Astro::RocketLanding stage;
        Vector x(3);
        x(0) = 3000.; x(1) = -10.; x(2) = 1000.;
        CHECK_CLOSE(stage.Guidance(x), 0., 0., "landing: above the profile the engine is off");
        x(1) = -300.;
        CHECK(stage.Guidance(x) > 0., "landing: below the profile it fires");
        Vector pad(3);
        pad(0) = 0.; pad(1) = -3.; pad(2) = 800.;
        const Vector landed = stage.f(pad, Vector(), 0.);
        CHECK(landed(0) == 0. && landed(1) == 0., "landing: a stage on the pad stays there at rest");
    }

    // --- descent: the guidance and the control input fly the same booster ------------
    {
        Apps::Astro::RocketDescent guided;
        Apps::Astro::RocketDescent driven(true);
        Vector x(7);
        x(0) = 120.; x(1) = 1500.; x(2) = -4.; x(3) = -50.; x(4) = 0.02; x(5) = 0.; x(6) = 1000.;
        double T, delta;
        guided.Guidance(x, T, delta);
        Vector u(2);
        u(0) = T / 40000.; u(1) = delta;
        const Vector a = guided.Derivative(x, Vector(), 0.), b = driven.Derivative(x, u, 0.);
        double diff = 0.;
        for (std::size_t i = 0; i < 7; ++i) diff = std::max(diff, std::fabs(a(i) - b(i)));
        CHECK(diff < 1e-12, "descent: u = [T / T_max, delta] reproduces the guidance");
        CHECK(T > 0. && T <= 40000., "descent: the throttle is within the engine's range");
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
