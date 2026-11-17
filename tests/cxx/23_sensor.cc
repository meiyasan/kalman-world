/**
 * \file 23_sensor.cc
 * \brief Kalman::Sensor and Models::Fusion: one more sensor on the state of another model.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>
#include <memory>
#include <stdexcept>

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

    double VelocityRMSE(const Record &rec, const std::vector<Vector> &truth, std::size_t from)
    {
        double sum = 0.;
        for (std::size_t k = from; k < truth.size(); ++k) sum += std::pow(rec.State(k)(1) - truth[k](1), 2);
        return std::sqrt(sum / static_cast<double>(truth.size() - from));
    }
}

int main()
{
    Matrix R1(1, 1, 0.); R1(0, 0) = 0.04;          // the new sensor: 0.2 in standard deviation

    // --- a function: the pendulum's angular velocity, read by a second instrument ---
    auto pendulum = std::make_shared<Fixture::Pendulum>(1.0, 0.15, 100.);
    pendulum->SetNoise(1e-2, 5e-2);
    Vector x0(2); x0(0) = 0.6; x0(1) = 0.;
    pendulum->SetInitialState(x0);

    const Sensor::Measure rate = [](const Vector &s, double) { Vector g(1); g(0) = s(1); return g; };
    Models::Fusion gyro(pendulum, Sensor("gyro", rate, R1, nullptr, {{"rate", "rad/s"}}));
    CHECK(gyro.NState() == 2 && gyro.NMeasure() == 2 && gyro.NExtra() == 1, "function: n unchanged, m = 1 + 1");
    CHECK(gyro.DefaultEngine() == EngineKind::Extended, "function: the extended engine");
    CHECK(gyro.Observables()[1].name == "rate" && gyro.Observables()[1].unit == "rad/s", "function: its observable is named");
    {
        const auto sensors = gyro.Sensors();
        CHECK(sensors.size() == 2, "sensors: the built-in one, then the added one");
        CHECK(sensors[0].name == "pendulum" && sensors[0].Channels() == 1, "sensors: the built-in one is the model's own h and R");
        CHECK(sensors[1].name == "gyro" && sensors[1].Channels() == 1, "sensors: the added one");
        Vector at(2); at(0) = 0.2; at(1) = 0.7;
        CHECK(sensors[0].g(at, 0.)(0) == pendulum->h(at, 0.)(0), "sensors: the built-in one reads h");
        CHECK(pendulum->Sensors().size() == 1, "sensors: the inner model is unchanged");
        CHECK(Sensor("gyro", rate, R1).Observables()[0].name == "gyro", "observables: named after the sensor by default");
        Matrix R2 = Matrix::Identity(2);
        CHECK(Sensor("imu", [](const Vector &, double) { return Vector(2); }, R2).Observables()[1].name == "imu1",
              "observables: numbered when there are several");
    }
    CHECK(gyro.X0()(0) == 0.6, "function: the initial state is the inner model's");
    gyro.CheckShapes();

    Vector x(2); x(0) = 0.4; x(1) = -0.3;
    {
        const Vector z = gyro.h(x, 0.);
        CHECK(z(0) == pendulum->h(x, 0.)(0) && z(1) == -0.3, "function: z = [h_inner; g]");
        const Matrix H = gyro.JacobianH(x, 0.);
        const Matrix Hs = pendulum->JacobianH(x, 0.);
        CHECK(H(0, 0) == Hs(0, 0) && H(0, 1) == Hs(0, 1), "function: the inner rows of the Jacobian are the inner model's");
        CHECK_CLOSE(H(1, 0), 0., 1e-8, "function: dg/dx differenced, d rate / d angle = 0");
        CHECK_CLOSE(H(1, 1), 1., 1e-8, "function: d rate / d rate = 1");
        const Matrix R = gyro.R(0.);
        CHECK(R(0, 0) == pendulum->R(0.)(0, 0) && R(1, 1) == 0.04 && R(0, 1) == 0., "function: R = diag(R_inner, R_g)");
        const Vector fx = gyro.f(x, Vector(), 0.), want = pendulum->f(x, Vector(), 0.);
        CHECK(fx(0) == want(0) && fx(1) == want(1), "function: the dynamics are the inner model's");
        Vector v(2); v(0) = 0.01; v(1) = 0.5;
        CHECK_CLOSE(gyro.MeasureWithNoise(x, v, 0.)(1), 0.2, 1e-15, "function: the new channel takes its noise");
    }
    {
        Models::Fusion analytic(pendulum, Sensor("gyro", rate, R1,
                                                 [](const Vector &, double) { Matrix J(1, 2, 0.); J(0, 1) = 1.; return J; }));
        CHECK(analytic.JacobianH(x, 0.)(1, 1) == 1., "function: a given Jacobian is used as it is");
    }

    // --- refusals --------------------------------------------------------------------------
    CHECK(Throws([&] { Models::Fusion bad(nullptr, Sensor("gyro", rate, R1)); }), "refused: a null model");
    CHECK(Throws([&] { Models::Fusion bad(pendulum, Sensor("", rate, R1)); }), "refused: a sensor with no name");
    CHECK(Throws([&] { Models::Fusion bad(pendulum, Sensor("gyro", rate, Matrix())); }), "refused: no channel");
    CHECK(Throws([&] { Sensor bad("gyro", Sensor::Measure(), R1); }), "refused: a null function");
    CHECK(Throws([&] { Models::Fusion bad(pendulum, Sensor("gyro", rate, Matrix(1, 2, 0.))); }), "refused: an R that is not square");
    CHECK(Throws([&] { Models::Fusion bad(pendulum, Sensor("gyro", Matrix(1, 3, 0.), R1)); }), "refused: a G with the wrong columns");
    CHECK(Throws([&] { Models::Fusion bad(pendulum, Sensor("gyro", rate, R1, nullptr, {{"a", ""}, {"b", ""}})); }),
          "refused: more names than channels");
    {
        Models::Fusion wrong(pendulum, Sensor("gyro", [](const Vector &, double) { return Vector(3); }, R1));
        CHECK(Throws([&] { wrong.h(x, 0.); }), "refused: a function that returns the wrong length");
    }

    // --- the second sensor helps: the velocity is no longer inferred from the angle alone ---
    // Over several records: std::normal_distribution draws another stream with each
    // standard library (libstdc++, libc++), so one record is not the same everywhere.
    {
        const std::size_t N = 1500;
        double nis = 0., with = 0., without = 0.;
        const std::vector<std::uint64_t> seeds = {99, 1, 2, 3, 4, 5, 6, 7};
        for (const std::uint64_t seed : seeds)
        {
            Sampler sampler(&gyro, seed);
            sampler.Generate(N, x0);
            Filters::EKF both(&gyro);
            const Record rec = both.Forward(sampler.Measurements());
            nis += both.MeanNIS() / seeds.size();

            std::vector<Vector> angles;
            for (const Vector &z : sampler.Measurements()) { Vector a(1); a(0) = z(0); angles.push_back(a); }
            Filters::EKF alone(pendulum.get());
            const Record ref = alone.Forward(angles);
            const double w = VelocityRMSE(rec, sampler.Truth(), 300), wo = VelocityRMSE(ref, sampler.Truth(), 300);
            std::printf("  seed %2llu: velocity RMSE angle only %.4f, angle + rate %.4f\n",
                        static_cast<unsigned long long>(seed), wo, w);
            with += w / seeds.size();
            without += wo / seeds.size();
        }
        std::printf("  mean NIS with two channels %.3f; mean velocity RMSE: angle only %.4f, angle + rate %.4f\n",
                    nis, without, with);
        CHECK(nis > 1.7 && nis < 2.3, "EKF: mean NIS consistent with m = 2");
        CHECK(with < without, "EKF: the rate channel lowers the velocity error");
    }

    // --- a matrix on a model of matrices is still a model of matrices ---------------------------
    {
        auto cv = std::make_shared<Fixture::ConstantVelocity>(1, 10.);
        Matrix G(1, 2, 0.); G(0, 1) = 1.;                 // reads the velocity
        Models::Fusion radar(cv, Sensor("radar", G, R1));
        CHECK(radar.DefaultEngine() == EngineKind::Linear, "matrix: the linear engine is kept");
        CHECK(radar.HasAnalyticJacobian(), "matrix: the Jacobians are still the matrices");
        CHECK(radar.Structure() == cv->Structure(), "matrix: the structure is the inner model's");
        const Matrix H = radar.H(0.);
        CHECK(H.Rows() == 2 && H(0, 0) == cv->H(0.)(0, 0) && H(1, 1) == 1. && H(1, 0) == 0., "matrix: H = [H_inner; G]");
        radar.CheckShapes();

        Vector s0(2); s0(0) = 0.; s0(1) = 1.;
        Sampler sampler(&radar, 7);
        sampler.Generate(800, s0);
        Filters::KF kf(&radar);
        kf.Forward(sampler.Measurements());
        const double nis = kf.MeanNIS();
        CHECK(nis > 1.6 && nis < 2.5, "KF: mean NIS consistent with m = 2");
    }

    // --- the method borrows, the function shares ---------------------------------------------
    {
        Fixture::Pendulum onStack(1.0, 0.15, 100.);
        const std::unique_ptr<Model> sensed = onStack.AddSensor(Sensor("gyro", rate, R1));
        CHECK(sensed->NMeasure() == 2 && sensed->Sensors().size() == 2, "model.AddSensor(sensor): borrows a model on the stack");
        const std::unique_ptr<Model> twice = sensed->AddSensor(Sensor("gyro2", rate, R1));
        CHECK(twice->NMeasure() == 3 && twice->Sensors().size() == 3 && twice->Sensors()[2].name == "gyro2",
              "AddSensor twice: one flat list of sensors");

        auto shared = std::make_shared<Fixture::Pendulum>(1.0, 0.15, 100.);
        CHECK(AddSensor(shared, Sensor("gyro", rate, R1))->NMeasure() == 2, "AddSensor(model, sensor): shares the model");
        Matrix G(1, 2, 0.); G(0, 1) = 1.;
        CHECK(AddSensor(shared, Sensor("gyro", G, R1))->Sensors()[1].IsLinear(), "AddSensor: a matrix sensor");
    }

    return Check::Summary("23_sensor");
}
