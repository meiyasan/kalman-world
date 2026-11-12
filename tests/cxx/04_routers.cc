/**
 * \file 04_routers.cc
 * \brief The decision layer: normalisation, the rule-based routers, and the blend.
 */
#include "Kalman/Kalman.h"
#include "Check.h"

#include <cmath>
#include <limits>

using namespace Kalman;
using namespace Kalman::Routers;

/// A NIS history in which model \p good sits at m and the others sit far above it.
static Matrix MakeNis(std::size_t rows, std::size_t M, std::size_t good, double m)
{
    Matrix nis(rows, M, 10. * m);
    for (std::size_t r = 0; r < rows; ++r) nis(r, good) = m;
    return nis;
}

int main()
{
    // --- Normalise ------------------------------------------------------------
    {
        Vector mu{2., 2., 4.};
        const Vector n = Router::Normalise(mu);
        CHECK_CLOSE(n(0) + n(1) + n(2), 1., 1e-12, "Normalise: sums to 1");
        CHECK_CLOSE(n(2), 0.5, 1e-12, "Normalise: preserves the ratios");

        // A mode driven to zero must stay recoverable, so it is floored, not dropped.
        Vector dead{1., 0., 0.};
        const Vector d = Router::Normalise(dead);
        CHECK(d(1) > 0., "Normalise: a zeroed mode keeps a positive floor");
        CHECK(d(1) < 1e-10, "Normalise: the floor is negligible, not a real weight");
        CHECK_CLOSE(d(0) + d(1) + d(2), 1., 1e-12, "Normalise: still a probability vector");

        Vector nan{std::numeric_limits<double>::quiet_NaN(),
                   std::numeric_limits<double>::quiet_NaN()};
        const Vector u = Router::Normalise(nan);
        CHECK_CLOSE(u(0), 0.5, 1e-12, "Normalise: all-NaN degrades to uniform, not garbage");
    }

    // --- Bayesian -------------------------------------------------------------
    {
        Bayesian router;
        const Matrix nis(3, 2, 1.);
        Vector c{0.5, 0.5};
        Vector L{3.0, 1.0};

        const Vector mu = router.Route(nis, c, L);
        CHECK_CLOSE(mu(0) + mu(1), 1., 1e-12, "Bayesian: normalised");
        CHECK_CLOSE(mu(0), 0.75, 1e-12, "Bayesian: mu proportional to c * likelihood");

        // With an uninformative likelihood the prior must survive untouched.
        Vector cSkew{0.9, 0.1};
        Vector flat{1.0, 1.0};
        const Vector keep = router.Route(nis, cSkew, flat);
        CHECK_CLOSE(keep(0), 0.9, 1e-12, "Bayesian: a flat likelihood leaves the prior alone");
    }

    // --- Threshold ------------------------------------------------------------
    {
        const double m = 2.;
        const Matrix nis = MakeNis(10, 3, /* good */ 1, m);
        Vector c(3, 1. / 3.);
        Vector L(3, 1.);

        Threshold hard(2, 5, /* soft */ false);
        const Vector mh = hard.Route(nis, c, L);
        CHECK_CLOSE(mh(1), 1., 1e-12, "Threshold(hard): all weight on the model whose NIS is m");
        CHECK_CLOSE(mh(0), 0., 1e-12, "Threshold(hard): nothing on the others");

        Threshold soft(2, 5, /* soft */ true);
        const Vector ms = soft.Route(nis, c, L);
        CHECK_CLOSE(ms(0) + ms(1) + ms(2), 1., 1e-12, "Threshold(soft): normalised");
        CHECK(ms(1) > ms(0) && ms(1) > ms(2), "Threshold(soft): still prefers the best match");
        CHECK(ms(0) > 0., "Threshold(soft): keeps the others alive");

        // A short history must not read as a wrong answer: front-padding keeps the
        // most recent step in the last slot whatever the length.
        const Matrix shortNis = MakeNis(2, 3, 1, m);
        const Vector me = hard.Route(shortNis, c, L);
        CHECK_CLOSE(me(1) + me(0) + me(2), 1., 1e-12, "Threshold: normalised on a short history");

        const Matrix empty(0, 3, 0.);
        const Vector mu0 = hard.Route(empty, c, L);
        CHECK_CLOSE(mu0(0), 1. / 3., 1e-12, "Threshold: uniform before anything is observed");
    }

    // --- Hybrid ---------------------------------------------------------------
    {
        Bayesian a;
        Threshold b(2, 5, true);
        const Matrix nis = MakeNis(10, 2, /* good */ 0, 2.);
        Vector c{0.5, 0.5};
        Vector L{1.0, 9.0};                 // the likelihood favours model 1 ...
                                            // ... while the NIS favours model 0

        Hybrid pureA(&a, &b, 0.0);
        Hybrid pureB(&a, &b, 1.0);
        Hybrid mixed(&a, &b, 0.5);

        const Vector ma = pureA.Route(nis, c, L);
        const Vector mb = pureB.Route(nis, c, L);
        const Vector mm = mixed.Route(nis, c, L);

        CHECK_CLOSE(ma(0), a.Route(nis, c, L)(0), 1e-9, "Hybrid: alpha=0 is exactly router a");
        CHECK_CLOSE(mb(0), Router::Normalise(b.Route(nis, c, L))(0), 1e-9,
                    "Hybrid: alpha=1 is exactly router b");
        CHECK(mm(0) > ma(0) && mm(0) < mb(0),
              "Hybrid: the blend sits between its two children");
        CHECK_CLOSE(mm(0) + mm(1), 1., 1e-12, "Hybrid: normalised");

        bool threw = false;
        try { Hybrid bad(&a, &b, 1.5); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Hybrid: an out-of-range alpha throws");

        threw = false;
        try { Hybrid bad(nullptr, &b); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Hybrid: a null child throws");
    }

    // --- the learned path: router owns features, model owns the forward pass --
    {
        // No model attached: degrades to Bayesian rather than failing. A missing
        // optional dependency must cost the policy, never the filter.
        Neural idle;
        CHECK(!idle.IsReady(), "Neural: holds no model until one is attached");

        const Matrix nis(3, 2, 1.);
        Vector c{0.5, 0.5};
        Vector L{3.0, 1.0};
        CHECK_CLOSE(idle.Route(nis, c, L)(0), 0.75, 1e-12,
                    "Neural: falls back to the Bayesian update with no model");

        // Feature layout must match routers/neural.py: window rows front-padded
        // with 1.0, flattened row-major. This is the router's job, not the model's.
        Neural probe;
        Matrix hist(2, 2, 0.);
        hist(0, 0) = 5.; hist(0, 1) = 6.;
        hist(1, 0) = 7.; hist(1, 1) = 8.;
        const Vector feat = probe.Features(hist);
        CHECK(feat.Size() == 5 * 2, "Neural: feature length is window * models");
        CHECK_CLOSE(feat(0), 1., 1e-12, "Neural: short history front-padded with 1.0");
        CHECK_CLOSE(feat(6), 5., 1e-12, "Neural: row-major, oldest real row first");
        CHECK_CLOSE(feat(9), 8., 1e-12, "Neural: most recent step in the last slot");

        Vector mean(10, 1.), scale(10, 2.);
        probe.SetNormalisation(mean, scale);
        CHECK_CLOSE(probe.Features(hist)(9), (8. - 1.) / 2., 1e-12,
                    "Neural: applies the training normalisation");

        Vector zero(10, 0.);
        probe.SetNormalisation(mean, zero);
        CHECK_CLOSE(probe.Scale()(0), 1., 1e-12, "Neural: zero-variance feature not divided by 0");
    }

    // --- NN::MLP — the backend that needs no optional dependency at all -------
    {
        // Two modes, ten features. Layer picks whichever of the two feature groups
        // is larger, so the expected answer is known without training anything.
        NN::MLP mlp;
        Matrix W(2, 10, 0.);
        for (std::size_t i = 0; i < 10; ++i) W(i % 2, i) = 1.0;
        Vector b(2, 0.);
        mlp.AddLayer(W, b);

        CHECK(mlp.IsReady(), "NN::MLP: ready once a layer is added");
        CHECK(mlp.NInputs() == 10 && mlp.NOutputs() == 2, "NN::MLP: reports its shape");

        Vector feat(10, 0.);
        for (std::size_t i = 0; i < 10; i += 2) feat(i) = 3.0;      // favours mode 0
        const Vector probs = mlp(feat);
        CHECK_CLOSE(probs(0) + probs(1), 1., 1e-12, "NN::MLP: outputs a probability vector");
        CHECK(probs(0) > probs(1), "NN::MLP: the excited output wins");

        // Large logits must not overflow to NaN — the softmax shift earns its keep.
        Vector big(10, 0.);
        for (std::size_t i = 0; i < 10; i += 2) big(i) = 400.0;
        const Vector stable = mlp(big);
        CHECK(std::isfinite(stable(0)) && std::isfinite(stable(1)),
              "NN::MLP: softmax is shift-stabilised against overflow");
        CHECK_CLOSE(stable(0), 1., 1e-9, "NN::MLP: saturates to a confident answer");

        // Shape mistakes are caught when the layer is added, not at inference.
        bool threw = false;
        try { NN::MLP bad; bad.AddLayer(Matrix(2, 3, 0.), Vector(5, 0.)); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "NN::MLP: a weight/bias mismatch throws on AddLayer");

        threw = false;
        try { NN::MLP bad; bad.AddLayer(Matrix(2, 3, 0.), Vector(2, 0.));
              bad.AddLayer(Matrix(2, 9, 0.), Vector(2, 0.)); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "NN::MLP: a layer that does not fit the previous one throws");

        // And it drives the router end to end, with no optional dependency present.
        Neural router(&mlp, 5);
        CHECK(router.IsReady(), "Neural: ready once a model is attached");
        Matrix nis(5, 2, 0.);
        for (std::size_t r = 0; r < 5; ++r) { nis(r, 0) = 3.0; nis(r, 1) = 0.0; }
        Vector c{0.5, 0.5}, L{1.0, 1.0};
        const Vector mu = router.Route(nis, c, L);
        CHECK_CLOSE(mu(0) + mu(1), 1., 1e-12, "Neural + NN::MLP: normalised");
        CHECK(mu(0) > mu(1), "Neural + NN::MLP: routes on the learned policy");

        // A window that disagrees with the policy is a shape error, not a guess.
        Neural wrong(&mlp, 7);
        threw = false;
        try { wrong.Route(nis, c, L); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Neural: a window mismatching the policy throws");
    }

    return Check::Summary("routers");
}
