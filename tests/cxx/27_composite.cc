/**
 * \file 27_composite.cc
 * \brief Models::Composite: several models side by side, stacked or summed.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
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

    std::shared_ptr<Fixture::ConstantVelocity> Cart(double fs, double v0)
    {
        auto cart = std::make_shared<Fixture::ConstantVelocity>(1, fs);
        cart->SetProcessNoise(1.);
        cart->SetMeasurementNoise(0.5);
        Vector x0(2); x0(0) = 0.; x0(1) = v0;
        cart->SetInitialState(x0);
        cart->SetInitialCovariance(Matrix::Identity(2));
        return cart;
    }

    std::shared_ptr<Fixture::Pendulum> Swing(double fs)
    {
        auto pendulum = std::make_shared<Fixture::Pendulum>(1.0, 0.15, fs);
        pendulum->SetNoise(1e-2, 5e-2);
        Vector x0(2); x0(0) = 0.6; x0(1) = 0.;
        pendulum->SetInitialState(x0);
        return pendulum;
    }
}

int main()
{
    const double fs = 10.;

    // --- two models of matrices, stacked -----------------------------------------------------
    auto a = Cart(fs, 1.), b = Cart(fs, -2.);
    Models::Composite pair({a, b});
    CHECK(pair.NState() == 4 && pair.NMeasure() == 2 && pair.NParts() == 2, "stacked: n and m add up");
    CHECK(pair.Structure() == StructureKind::Matrices && pair.DefaultEngine() == EngineKind::Linear,
          "matrices side by side are matrices: the linear engine");
    CHECK(pair.PartName(0) == "constant-velocity" && pair.PartName(1) == "constant-velocity2",
          "parts: called by their model's name, numbered from the second");
    CHECK(pair.X0()(1) == 1. && pair.X0()(3) == -2. && pair.P0()(2, 2) == 1. && pair.P0()(0, 2) == 0.,
          "x0 one after the other, P0 in blocks");
    {
        const auto names = pair.StateComponents();
        const std::string first = a->StateComponents()[0].name;
        CHECK(names[0].name == "constant-velocity_" + first && names[2].name == "constant-velocity2_" + first,
              "a component two parts share carries its part's name");
        Vector x(4); x(0) = 1.; x(1) = 2.; x(2) = 10.; x(3) = -4.;
        const Vector next = pair.f(x, Vector(), 0.);
        CHECK_CLOSE(next(0), 1.2, 1e-15, "f: each part steps its own state");
        CHECK_CLOSE(next(2), 9.6, 1e-15, "f: the second too");
        const Vector z = pair.h(x, 0.);
        CHECK(z(0) == 1. && z(1) == 10., "h: one measurement per part");
        const Matrix F = pair.F(0.), H = pair.H(0.), Q = pair.Q(0.), R = pair.R(0.);
        CHECK(F(0, 1) == 0.1 && F(2, 3) == 0.1 && F(0, 2) == 0. && F(2, 0) == 0., "F: block-diagonal");
        CHECK(H(0, 0) == 1. && H(1, 2) == 1. && H(0, 2) == 0., "H: block-diagonal");
        CHECK(Q(3, 3) == a->Q(0.)(1, 1) && Q(1, 3) == 0. && R(1, 1) == 0.25 && R(0, 1) == 0., "Q and R: block-diagonal");
        const Matrix J = pair.Jacobian(x, Vector(), 0.), JH = pair.JacobianH(x, 0.);
        CHECK(J(2, 3) == 0.1 && J(0, 3) == 0. && JH(1, 2) == 1., "the Jacobians are the parts' own, in blocks");
        const auto sensors = pair.Sensors();
        CHECK(sensors.size() == 2 && sensors[0].name == "constant-velocity_constant-velocity"
              && sensors[1].Read(x)(0) == 10., "Sensors: each part brings its own, reading its part of the state");
        pair.CheckShapes();
        CHECK(true, "CheckShapes passes");
    }

    // The composite filters as its parts do, each on its own.
    {
        Sampler sampler(&pair, 20261117);
        const std::size_t N = 500;
        sampler.Generate(N, pair.X0());
        Filters::KF both(&pair), one(a.get());
        const Record rec = both.Forward(sampler.Measurements());
        std::vector<Vector> first;
        for (const Vector &z : sampler.Measurements()) { Vector z0(1); z0(0) = z(0); first.push_back(z0); }
        const Record alone = one.Forward(first);
        double worst = 0.;
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t i = 0; i < 2; ++i) worst = std::max(worst, std::fabs(rec.State(k)(i) - alone.State(k)(i)));
        CHECK(worst < 1e-10, "KF: the first part is filtered as it would be alone");
        CHECK(both.MeanNIS() > 1.7 && both.MeanNIS() < 2.3, "KF: mean NIS consistent with m = 2");
    }

    // --- summed: two signals on one channel ----------------------------------------------------------
    {
        Models::Composite sum({a, b}, Readout::Summed, {"slow", "fast"});
        CHECK(sum.NState() == 4 && sum.NMeasure() == 1, "summed: one measurement");
        Vector x(4); x(0) = 1.; x(1) = 2.; x(2) = 10.; x(3) = -4.;
        CHECK(sum.h(x, 0.)(0) == 11., "h: the parts add up");
        const Matrix H = sum.H(0.), JH = sum.JacobianH(x, 0.);
        CHECK(H(0, 0) == 1. && H(0, 2) == 1. && H(0, 1) == 0. && JH(0, 2) == 1., "H: the parts' rows side by side");
        CHECK(sum.R(0.).Rows() == 1 && sum.R(0.)(0, 0) == 0.25, "R: the noise of the one channel");
        Vector v(1); v(0) = 0.5;
        CHECK(sum.MeasureWithNoise(x, v, 0.)(0) == 11.5, "the noise is added once");
        CHECK(sum.Sensors().size() == 1 && sum.StateComponents()[2].name.rfind("fast_", 0) == 0,
              "one sensor; the parts go by the names given");
        CHECK(sum.Structure() == StructureKind::Matrices, "summed matrices are matrices");
        auto wide = std::make_shared<Fixture::ConstantVelocity>(2, fs);
        CHECK(Throws([&] { Models::Composite bad({a, wide}, Readout::Summed); }),
              "refused: summed parts with different measurements");
    }

    // --- a part that is code -----------------------------------------------------------------------
    {
        auto swing = Swing(fs);
        Models::Composite mixed({a, swing});
        CHECK(mixed.Structure() == StructureKind::Maps && mixed.DefaultEngine() == EngineKind::Extended,
              "one part of code makes it code: the extended engine");
        Vector x(4); x(0) = 1.; x(1) = 2.; x(2) = 0.3; x(3) = -0.2;
        Vector s(2); s(0) = 0.3; s(1) = -0.2;
        const Vector next = mixed.f(x, Vector(), 0.), want = swing->f(s, Vector(), 0.);
        CHECK(next(2) == want(0) && next(3) == want(1), "f: the part of code steps itself");
        const Matrix J = mixed.Jacobian(x, Vector(), 0.), Js = swing->Jacobian(s, Vector(), 0.);
        CHECK(J(3, 2) == Js(1, 0) && J(2, 3) == Js(0, 1) && J(0, 1) == 0.1, "Jacobian: each part linearises itself");
        Filters::EKF ekf(&mixed);
        Sampler sampler(&mixed, 7);
        sampler.Generate(300, mixed.X0());
        ekf.Forward(sampler.Measurements());
        CHECK(ekf.MeanNIS() > 1.5 && ekf.MeanNIS() < 2.6, "EKF: mean NIS consistent with m = 2");
    }

    // --- parameters, refusals, and the ways in -----------------------------------------------------------
    {
        auto station = AddInstrument(a, Compose({Sensor("probe", a->H(0.), a->R(0.))}, "thermo").Calibrate(1., 0.5));
        Models::Composite named({station, b}, Readout::Stacked, {"left", "right"});
        CHECK(named.HasParameter("left.thermo.offset") && named.Parameter("left.thermo.offset") == 0.5,
              "parameters: <part>.<name>");
        named.SetParameter("left.thermo.offset", 2.);
        CHECK(station->Parameter("thermo.offset") == 2., "SetParameter reaches the part");
        CHECK(!named.HasParameter("right.thermo.offset") && Throws([&] { named.Parameter("thermo.offset"); }),
              "refused: a name without its part, or on the wrong one");
        const auto joint = Joint(std::make_shared<Models::Composite>(
            std::vector<std::shared_ptr<Model>>{station, b}, Readout::Stacked, std::vector<std::string>{"left", "right"}),
            {{"left.thermo.offset", 1., 0.}});
        CHECK(joint->NState() == 5, "joint on a composite: a part's parameter estimated with the state");

        CHECK(Throws([&] { Models::Composite bad({}); }), "refused: no part");
        CHECK(Throws([&] { Models::Composite bad({a, nullptr}); }), "refused: a null part");
        CHECK(Throws([&] { Models::Composite bad({a, Cart(20., 0.)}); }), "refused: parts at different sample rates");
        CHECK(Throws([&] { Models::Composite bad({a, b}, Readout::Stacked, {"same", "same"}); }), "refused: two parts of one name");

        Fixture::ConstantVelocity onStack(1, fs), other(1, fs);
        CHECK(onStack.Compose(other)->NState() == 4, "model.Compose(other): borrows both");
        const auto shared = Compose({a, b}, Readout::Summed);
        CHECK(shared->NMeasure() == 1, "Compose({a, b}, readout): shares them");
        shared->SetSampleRate(20.);
        shared->f(shared->X0(), Vector(), 0.);
        CHECK(a->SampleRate() == 20. && b->SampleRate() == 20., "sample rate: passed on to the parts");
    }

    return Check::Summary("27_composite");
}
