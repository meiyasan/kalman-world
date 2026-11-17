/**
 * \file 28_sensors.cc
 * \brief Kalman::Sensors: a position reading, a Doppler reading, a camera; their settings as
 *        parameters of the model; the sensor factory.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace Kalman;

namespace
{
    template <class Fn>
    bool Throws(Fn fn)
    {
        try { fn(); }
        catch (const std::exception &) { return true; }
        return false;
    }

    /// The largest difference between a sensor's Jacobian and central differences of its reading.
    double JacobianError(const Sensor &sensor, const Vector &x)
    {
        const Matrix J = sensor.jacobian(x, 0.);
        double worst = 0.;
        for (std::size_t j = 0; j < x.Size(); ++j)
        {
            Vector up = x, down = x;
            const double step = 1e-6 * (1. + std::abs(x(j)));
            up(j) += step;
            down(j) -= step;
            const Vector zu = sensor.g(up, 0.), zd = sensor.g(down, 0.);
            for (std::size_t i = 0; i < zu.Size(); ++i)
                worst = std::max(worst, std::abs(J(i, j) - (zu(i) - zd(i)) / (2. * step)));
        }
        return worst;
    }

    // State of the fixture with three axes: (px, vx, py, vy, pz, vz).
    const std::vector<std::size_t> P = {0, 2, 4}, V = {1, 3, 5};

    std::shared_ptr<Fixture::ConstantVelocity> Target()
    {
        auto target = std::make_shared<Fixture::ConstantVelocity>(3, 20.);
        target->SetProcessNoise(1.);
        target->SetMeasurementNoise(5.);
        Vector x0(6);
        x0(0) = 100.; x0(1) = 30.; x0(2) = 50.; x0(3) = -10.; x0(4) = 200.; x0(5) = 5.;
        target->SetInitialState(x0);
        target->SetInitialCovariance(25. * Matrix::Identity(6));
        return target;
    }
}

int main()
{
    Vector x(6);
    x(0) = 3.; x(1) = 1.; x(2) = 4.; x(3) = -2.; x(4) = 12.; x(5) = 0.5;

    // --- Position --------------------------------------------------------------------------
    {
        const Sensors::Position gps("gps", 6, {0, 2}, 3.);
        CHECK(gps.IsLinear() && gps.Channels() == 2 && gps.name == "gps", "position: a matrix sensor, one channel per axis");
        CHECK(gps.G(0, 0) == 1. && gps.G(1, 2) == 1. && gps.G(0, 2) == 0. && gps.G(1, 1) == 0., "position: G picks the axes");
        CHECK(gps.R(0, 0) == 9. && gps.R(1, 1) == 9. && gps.R(0, 1) == 0., "position: R = sigma^2 on each channel");
        const Vector z = gps.Read(x);
        CHECK(z(0) == 3. && z(1) == 4., "position: reads the components");

        Matrix turn(2, 2, 0.);                      // a quarter turn: the sensor's first axis is the state's second
        turn(0, 1) = 1.; turn(1, 0) = -1.;
        const Sensors::Position turned("turned", 6, {0, 2}, 3., turn);
        const Vector zt = turned.Read(x);
        CHECK(zt(0) == 4. && zt(1) == -3., "position: in the sensor's own axes when rotated");

        CHECK(Throws([] { Sensors::Position("bad", 6, {7}, 1.); }), "refused: a component the state does not have");
        CHECK(Throws([] { Sensors::Position("bad", 6, {0}, 0.); }), "refused: sigma = 0");
        CHECK(Throws([] { Sensors::Position("bad", 6, {}, 1.); }), "refused: no axis");
        CHECK(Throws([&] { Sensors::Position("bad", 6, {0}, 1., turn); }), "refused: a rotation of another size");

        auto target = Target();
        const auto both = AddSensor(target, Sensors::Position("second", 6, P, 1.));
        CHECK(both->NMeasure() == 6 && both->Structure() == target->Structure() && both->DefaultEngine() == EngineKind::Linear,
              "position: a model of matrices that takes it stays one");
    }

    // --- Doppler -----------------------------------------------------------------------------
    {
        const Sensors::Doppler reading("doppler", 6, P, V, 0.5);
        CHECK(reading.name == "doppler" && reading.Channels() == 1, "doppler: one channel");
        CHECK(!reading.IsLinear() && reading.R(0, 0) == 0.25 && bool(reading.jacobian), "doppler: a function, with its Jacobian");
        // p = (3, 4, 12), |p| = 13, v = (1, -2, 0.5): v . p / |p| = (3 - 8 + 6) / 13
        CHECK_CLOSE(reading.g(x, 0.)(0), 1. / 13., 1e-15, "doppler: the velocity along the line of sight");
        CHECK(JacobianError(reading, x) < 1e-8, "doppler: the Jacobian is the derivative of the reading");

        Vector at(3);
        at(0) = 3.; at(1) = 4.; at(2) = 0.;         // straight below the target
        Sensors::Doppler below("below", 6, P, V, 0.5, at);
        CHECK_CLOSE(below.g(x, 0.)(0), 0.5, 1e-15, "doppler: seen from where the sensor is");
        CHECK(JacobianError(below, x) < 1e-8, "doppler: and its Jacobian there");
        Vector on = x;
        on(4) = 0.;                                 // the target at the sensor
        CHECK(below.g(on, 0.)(0) == 0. && below.jacobian(on, 0.)(0, 5) == 0., "doppler: at the sensor itself the reading is 0");

        const std::vector<std::string> names = below.SettingNames();
        CHECK(names.size() == 3 && names[0] == "x" && names[2] == "z" && below.Setting("y") == 4.,
              "doppler: where it is are its settings x, y, z");
        below.Set("x", 0.); below.Set("y", 0.);
        CHECK_CLOSE(below.g(x, 0.)(0), 1. / 13., 1e-15, "doppler: a setting changed moves the sensor");
        CHECK(Throws([&] { below.Set("focal", 1.); }) && !below.HasSetting("focal"), "refused: a setting it does not have");

        const Sensors::Doppler flat("flat", 6, {0, 2}, {1, 3}, 0.5);
        CHECK_CLOSE(flat.g(x, 0.)(0), (3. - 8.) / 5., 1e-15, "doppler: in two dimensions");
        CHECK(flat.SettingNames().size() == 2, "doppler: as many settings as dimensions");

        CHECK(Throws([] { Sensors::Doppler("bad", 6, {0, 2, 4}, {1, 3}, 1.); }), "refused: position and velocity of different sizes");
        CHECK(Throws([&] { Vector two(2); Sensors::Doppler("bad", 6, P, V, 1., two); }), "refused: a position of another size");
    }

    // --- Camera -------------------------------------------------------------------------------------
    {
        Sensors::Camera image("camera", 6, P, 800., 1.5);
        CHECK(image.Channels() == 2 && !image.IsLinear(), "camera: two channels");
        const Vector z = image.g(x, 0.);
        CHECK_CLOSE(z(0), 800. * 3. / 12., 1e-12, "camera: u = focal x / z");
        CHECK_CLOSE(z(1), 800. * 4. / 12., 1e-12, "camera: v = focal y / z");
        CHECK(image.R(0, 0) == 2.25 && image.R(1, 1) == 2.25, "camera: R = sigma^2 on each pixel coordinate");
        CHECK(JacobianError(image, x) < 1e-5, "camera: the Jacobian is the derivative of the projection");

        // Looking along the world's x axis from (-5, 0, 0): camera axes (y, z, x).
        Matrix look(3, 3, 0.);
        look(0, 1) = 1.; look(1, 2) = 1.; look(2, 0) = 1.;
        Vector at(3);
        at(0) = -5.;
        const Sensors::Camera side("side", 6, P, 100., 1., at, look);
        const Vector zs = side.g(x, 0.);
        CHECK_CLOSE(zs(0), 100. * 4. / 8., 1e-12, "camera: placed and turned, u");
        CHECK_CLOSE(zs(1), 100. * 12. / 8., 1e-12, "camera: placed and turned, v");
        CHECK(JacobianError(side, x) < 1e-6, "camera: and its Jacobian");

        Vector behind = x;
        behind(4) = -1.;
        const Vector zb = image.g(behind, 0.);
        CHECK(std::isfinite(zb(0)) && std::isfinite(zb(1)), "camera: a point behind it still gives finite numbers");

        CHECK(image.SettingNames().size() == 4 && image.Setting("focal") == 800. && image.Setting("z") == 0.,
              "camera: its settings are focal, x, y, z");
        image.Set("focal", 400.);
        CHECK_CLOSE(image.g(x, 0.)(0), 400. * 3. / 12., 1e-12, "camera: the focal length changed");
        CHECK(JacobianError(image, x) < 1e-5, "camera: and the Jacobian follows");

        CHECK(Throws([] { Sensors::Camera("bad", 6, {0, 2}, 800., 1.); }), "refused: a point of two coordinates");
        CHECK(Throws([] { Sensors::Camera("bad", 6, {0, 2, 4}, 0., 1.); }), "refused: focal = 0");
    }

    // --- by name --------------------------------------------------------------------------------------
    {
        const SensorFactory &factory = SensorFactory::Instance();
        CHECK(factory.Has("position") && factory.Has("doppler") && factory.Has("camera") && !factory.Has("sonar"),
              "factory: the three kinds are registered");
        const Sensor doppler = factory.Create("doppler", "radar", 6, {3, 0, 2, 4, 1, 3, 5, 0.5});
        CHECK(doppler.name == "radar" && doppler.g(x, 0.)(0) == Sensors::Doppler("radar", 6, P, V, 0.5).g(x, 0.)(0)
              && doppler.HasSetting("x"), "factory: 'doppler' is Sensors::Doppler, settings included");
        const Sensor gps = factory.Create("position", "gps", 6, {2, 0, 2, 3.});
        CHECK(gps.IsLinear() && gps.G(1, 2) == 1. && gps.R(0, 0) == 9., "factory: 'position' is Sensors::Position");
        const Sensor camera = factory.Create("camera", "cam", 6, {0, 2, 4, 800., 1.5});
        CHECK_CLOSE(camera.g(x, 0.)(1), 800. * 4. / 12., 1e-12, "factory: 'camera' is Sensors::Camera");
        CHECK(Throws([&] { factory.Create("sonar", "s", 6, {}); }), "refused: a kind that is not registered");
        CHECK(Throws([&] { factory.Create("doppler", "s", 6, {3, 0, 2, 4}); }), "refused: too few settings");
        CHECK(Throws([&] { factory.Create("camera", "s", 6, {0, 2, 4, 800., 1.5, 9.}); }), "refused: settings that do not add up");
        CHECK(Throws([&] { factory.Create("position", "s", 6, {1, 0.5, 1.}); }), "refused: an index that is not a whole number");
    }

    // --- in a filter ----------------------------------------------------------------------------------
    {
        const std::size_t N = 600;
        const auto seen = AddSensor(Target(), Sensors::Doppler("doppler", 6, P, V, 0.2));
        CHECK(seen->NMeasure() == 4 && seen->DefaultEngine() == EngineKind::Extended, "AddSensor: one more channel, a model of code");
        Sampler sampler(seen.get(), 20261117);
        sampler.Generate(N, seen->X0());

        Filters::EKF both(seen.get());
        const Record with = both.Forward(sampler.Measurements());
        auto target = Target();
        std::vector<Vector> positions;
        for (const Vector &z : sampler.Measurements())
        {
            Vector p(3);
            for (std::size_t i = 0; i < 3; ++i) p(i) = z(i);
            positions.push_back(p);
        }
        Filters::KF alone(target.get());
        const Record without = alone.Forward(positions);
        const double v_with = with.RMSE(1, sampler.Truth()), v_without = without.RMSE(1, sampler.Truth());
        std::printf("  velocity error along x: %.3f with the Doppler reading, %.3f from positions alone\n", v_with, v_without);
        CHECK(v_with < 0.8 * v_without, "the Doppler reading improves the velocity");
        CHECK(both.MeanNIS() > 3.2 && both.MeanNIS() < 4.8, "EKF: mean NIS consistent with m = 4");
    }

    // --- settings are parameters of the model: Joint estimates them --------------------------------------
    {
        // An adapter is one piece of hardware: an instrument of generic sensors.
        const Instrument radar = Compose({Sensors::Doppler("doppler", 6, P, V, 0.2)}, "radar").Calibrate(1., 0.);
        const auto tracked = AddInstrument(Target(), radar);
        CHECK(tracked->HasParameter("radar.offset") && tracked->HasParameter("radar.x") && tracked->HasParameter("radar.z"),
              "an instrument's calibration and its sensors' settings are parameters of the model");
        CHECK(!tracked->HasParameter("radar.focal"), "and only those");
        tracked->SetParameter("radar.x", 7.);
        CHECK(tracked->Parameter("radar.x") == 7., "SetParameter reaches the sensor");

        Vector at(3);
        at(2) = -50.;
        const std::size_t N = 1500;
        const auto real = AddSensor(Target(), Sensors::Camera("camera", 6, P, 800., 1.5, at));
        Sampler sampler(real.get(), 20261118);
        sampler.Generate(N, real->X0());

        // The filter is told 700 px; the focal length is estimated with the state.
        const auto assumed = AddSensor(Target(), Sensors::Camera("camera", 6, P, 700., 1.5, at));
        CHECK(assumed->Parameter("camera.focal") == 700., "camera.focal is the focal length the sensor was built with");
        const auto model = Joint(assumed, {{"camera.focal", 100., 0.}});
        Filters::EKF ekf(model.get());
        const Record record = ekf.Forward(sampler.Measurements());
        const double focal = record.State(N - 1)(6), sigma = record.Sigma(6)[N - 1];
        std::printf("  camera.focal = %.1f +/- %.1f px (truth 800, started at 700)\n", focal, sigma);
        CHECK(std::abs(focal - 800.) < 4. * sigma && sigma < 10., "Joint finds the focal length");
    }

    return Check::Summary("sensors");
}
