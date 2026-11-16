/**
 * \file 26_lag.cc
 * \brief Models::Lag: a model that carries its last states, for late sensors and fixed-lag smoothing.
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

    std::shared_ptr<Fixture::ConstantVelocity> Cart(double fs)
    {
        auto cart = std::make_shared<Fixture::ConstantVelocity>(1, fs);
        cart->SetProcessNoise(1.);
        cart->SetMeasurementNoise(0.5);
        Vector x0(2); x0(0) = 0.; x0(1) = 1.;
        cart->SetInitialState(x0);
        cart->SetInitialCovariance(Matrix::Identity(2));
        return cart;
    }

    /// A precise position sensor that reports \p delay late.
    Sensor Late(double delay)
    {
        Matrix G(1, 2, 0.); G(0, 0) = 1.;
        Matrix R(1, 1, 0.); R(0, 0) = 0.01;
        Sensor late("late", G, R, {{"position_late", "m"}});
        late.delay = delay;
        return late;
    }
}

int main()
{
    const double fs = 10.;

    // --- construction ------------------------------------------------------------------
    auto cart = Cart(fs);
    Models::Lag lag(cart, 3);
    CHECK(lag.NState() == 8 && lag.NMeasure() == 1 && lag.Depth() == 3 && lag.NInner() == 2, "lag: n = 2 * (3 + 1), m unchanged");
    CHECK(lag.DefaultEngine() == EngineKind::Linear && lag.Structure() == cart->Structure(), "lag: a model of matrices stays one");
    CHECK(lag.StateComponents()[0].name == cart->StateComponents()[0].name, "lag: the present state keeps its names");
    CHECK(lag.StateComponents()[2].name == cart->StateComponents()[0].name + "@-1"
          && lag.StateComponents()[7].name == cart->StateComponents()[1].name + "@-3", "lag: the state j steps ago carries @-j");
    CHECK(lag.X0()(1) == 1. && lag.X0()(7) == 1., "lag: x0 is the inner one in every block");
    CHECK(lag.P0()(6, 6) == 1. && lag.P0()(0, 2) == 0., "lag: P0 on each block, none between");
    lag.CheckShapes();
    CHECK(true, "lag: CheckShapes passes");
    CHECK(Throws([&] { Models::Lag bad(cart, 0); }), "refused: depth = 0");
    CHECK(Throws([&] { Models::Lag bad(nullptr, 2); }), "refused: a null model");

    // --- the maps --------------------------------------------------------------------------
    {
        Vector x(8);
        for (std::size_t i = 0; i < 8; ++i) x(i) = 1. + static_cast<double>(i);
        const Vector next = lag.f(x, Vector(), 0.);
        CHECK_CLOSE(next(0), 1. + 2. / fs, 1e-15, "f: the present state steps");
        CHECK(next(2) == 1. && next(3) == 2. && next(6) == 5. && next(7) == 6., "f: the others shift down");
        CHECK(lag.Block(x, 2)(0) == 5. && lag.Block(x, 2)(1) == 6., "Block: the state two steps ago");
        CHECK(Throws([&] { lag.Block(x, 4); }), "Block: refused beyond the depth");
        const Matrix F = lag.F(0.), J = lag.Jacobian(x, Vector(), 0.);
        bool same = true;
        for (std::size_t i = 0; i < 8; ++i)
            for (std::size_t j = 0; j < 8; ++j) same = same && F(i, j) == J(i, j);
        CHECK(same && F(0, 1) == 1. / fs && F(2, 0) == 1. && F(7, 5) == 1. && F(2, 2) == 0., "F: the inner one, and the shift");
        CHECK(lag.h(x, 0.)(0) == 1. && lag.H(0.)(0, 0) == 1. && lag.H(0.)(0, 2) == 0., "h: the present state, with no late sensor");
        const Matrix Q = lag.Q(0.);
        CHECK(Q(0, 0) == cart->Q(0.)(0, 0) && Q(2, 2) == 0. && Q(7, 7) == 0., "Q: the past states take no noise");
    }

    // --- fixed-lag smoothing ---------------------------------------------------------------------
    {
        Sampler sampler(&lag, 20261117);
        const std::size_t N = 2000;
        sampler.Generate(N, lag.X0());
        Filters::KF kf(&lag);
        const Record rec = kf.Forward(sampler.Measurements());
        const double now = rec.RMSE(0, sampler.Truth()), then = rec.RMSE(6, sampler.Truth());
        const double v_now = rec.RMSE(1, sampler.Truth()), v_then = rec.RMSE(7, sampler.Truth());
        std::printf("  position RMSE: filtered %.4f, three steps later %.4f; velocity %.4f -> %.4f\n", now, then, v_now, v_then);
        CHECK(then < 0.9 * now, "block 3 is the state three steps ago, improved by what came after");
        CHECK(v_then < 0.9 * v_now, "and so is its velocity");
        CHECK(kf.MeanNIS() > 0.8 && kf.MeanNIS() < 1.2, "KF: mean NIS consistent with m = 1");
    }

    // --- a sensor that reports late ------------------------------------------------------------------
    const std::shared_ptr<Model> both = AddSensor(Cart(fs), Late(0.3));          // 3 steps late
    CHECK(Throws([&] { both->CheckShapes(); }), "refused: a late sensor on a model that carries no past state");
    CHECK(Throws([&] { Models::Lag bad(both, 2); }), "refused: late by more steps than the model carries");
    const std::shared_ptr<Models::Lag> model = Lag(both, 3);
    model->CheckShapes();
    CHECK(model->Delay(0) == 0 && model->Delay(1) == 3, "Delay: each row of z and the block it reads");
    {
        Vector x(8);
        for (std::size_t i = 0; i < 8; ++i) x(i) = 1. + static_cast<double>(i);
        const Vector z = model->h(x, 0.);
        CHECK(z(0) == 1. && z(1) == 7., "h: the late sensor reads the position three steps ago");
        const Matrix H = model->H(0.), JH = model->JacobianH(x, 0.);
        CHECK(H(0, 0) == 1. && H(1, 6) == 1. && H(1, 0) == 0. && JH(1, 6) == 1. && JH(1, 0) == 0.,
              "H: its row sits on that block");
        const auto sensors = model->Sensors();
        CHECK(sensors.size() == 2 && sensors[1].delay == 0. && sensors[1].Read(x)(0) == 7.,
              "Sensors: each reads its own block of the lagged state");
        CHECK(Throws([&] { AddSensor(model, Late(0.1))->CheckShapes(); }), "refused: a late sensor added after the lag");
        const std::shared_ptr<Models::Lag> twice = Lag(model, 1);
        Vector xx(16, 0.); xx(6) = 7.;
        CHECK(twice->NState() == 16 && twice->h(xx, 0.)(1) == 7., "they nest: the delay is applied once");
    }
    {
        Sampler sampler(model.get(), 20261118);
        const std::size_t N = 2000;
        sampler.Generate(N, model->X0());
        Filters::KF kf(model.get());
        const Record rec = kf.Forward(sampler.Measurements());

        // The same readings, the delay ignored: the late sensor is believed to read the present.
        Sensor mistaken = Late(0.);
        const std::shared_ptr<Model> naive = AddSensor(Cart(fs), mistaken);
        Filters::KF wrong(naive.get());
        const Record bad = wrong.Forward(sampler.Measurements());

        // Without the late sensor at all.
        auto cart_alone = Cart(fs);
        std::vector<Vector> first;
        for (const Vector &z : sampler.Measurements()) { Vector one(1); one(0) = z(0); first.push_back(one); }
        Filters::KF alone(cart_alone.get());
        const Record base = alone.Forward(first);

        double e_lag = 0., e_naive = 0., e_alone = 0.;
        for (std::size_t k = 0; k < N; ++k)
        {
            const double truth = sampler.Truth()[k](0);
            e_lag += std::pow(rec.State(k)(0) - truth, 2);
            e_naive += std::pow(bad.State(k)(0) - truth, 2);
            e_alone += std::pow(base.State(k)(0) - truth, 2);
        }
        e_lag = std::sqrt(e_lag / N); e_naive = std::sqrt(e_naive / N); e_alone = std::sqrt(e_alone / N);
        std::printf("  position RMSE: lagged %.4f, delay ignored %.4f, late sensor left out %.4f; NIS %.3f (ignored: %.2f)\n",
                    e_lag, e_naive, e_alone, kf.MeanNIS(), wrong.MeanNIS());
        CHECK(e_lag < 0.8 * e_alone, "the late sensor still helps the present state");
        CHECK(e_lag < 0.8 * e_naive, "and ignoring its delay is worse");
        CHECK(kf.MeanNIS() > 1.7 && kf.MeanNIS() < 2.3, "KF: mean NIS consistent with m = 2");
        CHECK(wrong.MeanNIS() > 3., "the filter that ignores the delay is not consistent");
    }

    // --- under joint, and the method that borrows ----------------------------------------------------
    {
        Fixture::ConstantVelocity onStack(1, fs);
        const std::unique_ptr<Model> lagged = onStack.Lag(2);
        CHECK(lagged->NState() == 6, "model.Lag(L): borrows a model on the stack");
        lagged->SetSampleRate(20.);
        lagged->f(lagged->X0(), Vector(), 0.);
        CHECK(onStack.SampleRate() == 20., "sample rate: passed on to the inner model");
    }

    return Check::Summary("26_lag");
}
