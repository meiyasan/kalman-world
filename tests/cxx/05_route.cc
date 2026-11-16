/**
 * \file 05_route.cc
 * \brief Models::Router on the interacting engine: does it find the regime, and switch when it changes?
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

using namespace Kalman;

namespace
{
    /// A router model over borrowed modes, its filter, and the engine that holds the mode history.
    struct Bank
    {
        std::unique_ptr<Model> model;
        std::unique_ptr<Filter> filter;
        Bank(const std::vector<Model *> &modes, const Matrix &Pi, const Routers::Router *decision = nullptr)
            : model(Model::Route(modes, Pi))
        {
            if (decision) Modes().SetDecision(decision);
            filter = std::make_unique<Filter>(model.get());
        }
        Models::Router &Modes() { return dynamic_cast<Models::Router &>(*model); }
        const Engines::Interacting &Engine() const { return dynamic_cast<const Engines::Interacting &>(*filter->GetEngine()); }
        Record Forward(const std::vector<Vector> &zk) { return filter->Forward(zk); }
        Matrix ModeHistory() const { return Engine().ModeHistory(); }
        Matrix NisHistory() const { return Engine().NisHistory(); }
        std::size_t DominantMode(std::size_t k) const { return Engine().DominantMode(k); }
    };
}

int main()
{
    // Two modes of the same 1-D system, differing in how much unmodelled
    // acceleration they expect: a quiet cruise and a manoeuvre.
    //
    // The contrast has to be large enough to *show up in the data*. With a
    // measurement noise comparable to the process noise, both modes explain the
    // track equally well (measured: innovation covariances 8% apart, both with
    // mean NIS ~ 1.09) and a correct Bayesian router then returns ~0.5 — not
    // because the filter is broken but because there is genuinely nothing to
    // choose between. Hence a sharp sensor and a violent manoeuvre.
    const double sigmaZ = 0.05;

    Fixture::ConstantVelocity quiet(1, 50.);
    quiet.SetProcessNoise(0.05);
    quiet.SetMeasurementNoise(sigmaZ);

    Fixture::ConstantVelocity agile(1, 50.);
    agile.SetProcessNoise(50.0);
    agile.SetMeasurementNoise(sigmaZ);

    Matrix Pi(2, 2, 0.);
    Pi(0, 0) = 0.95; Pi(0, 1) = 0.05;
    Pi(1, 0) = 0.05; Pi(1, 1) = 0.95;

    // --- construction is validated ------------------------------------------
    {
        bool threw = false;
        try { Matrix bad(3, 3, 1.); Bank ikf({&quiet, &agile}, bad); }
        catch (const std::exception &) { threw = true; }
        CHECK(threw, "route: a Pi of the wrong shape throws");

        threw = false;
        try { Matrix bad(2, 2, 0.7); Bank ikf({&quiet, &agile}, bad); }
        catch (const std::exception &) { threw = true; }
        CHECK(threw, "route: a Pi whose rows do not sum to 1 throws");

        threw = false;
        try { Bank ikf({&quiet}, Matrix::Identity(1)); }
        catch (const std::exception &) { threw = true; }
        CHECK(threw, "route: fewer than two modes throws");

        threw = false;
        try
        {
            Fixture::ConstantVelocity plane(2, 50.);       // n = 4, not 2
            Bank ikf({&quiet, &plane}, Pi);
        }
        catch (const std::exception &) { threw = true; }
        CHECK(threw, "route: modes with different dimensions throw");
    }

    // --- a quiet track: the filter should settle on the quiet mode ---------------
    {
        Vector x0(2); x0(0) = 0.; x0(1) = 2.;
        quiet.SetInitialState(x0);
        agile.SetInitialState(x0);

        Sampler sampler(&quiet, 4242);
        sampler.Generate(400, x0);

        Routers::Bayesian router;
        Bank ikf({&quiet, &agile}, Pi, &router);
        const Record rec = ikf.Forward(sampler.Measurements());

        CHECK(rec.NSteps() == 400, "route: one record entry per measurement");
        CHECK(rec.NState() == 2, "route: the combined estimate has the shared state size");

        const Matrix modes = ikf.ModeHistory();
        CHECK(modes.Rows() == 400 && modes.Cols() == 2, "route: mode history is (T, M)");

        bool normalised = true;
        for (std::size_t k = 0; k < modes.Rows(); ++k)
            if (std::abs(modes(k, 0) + modes(k, 1) - 1.) > 1e-9) normalised = false;
        CHECK(normalised, "route: mode probabilities sum to 1 at every step");

        // Average over the second half, once the filter has had time to decide.
        double quietShare = 0.;
        for (std::size_t k = 200; k < 400; ++k) quietShare += modes(k, 0);
        quietShare /= 200.;
        std::printf("    quiet-mode probability on a quiet track: %.3f\n", quietShare);
        CHECK(quietShare > 0.6, "route: settles on the mode that generated the data");

        // The estimate must still be a good estimate, not just a good classifier.
        double eraw = 0.;
        for (std::size_t k = 0; k < 400; ++k)
            eraw += std::pow(sampler.Measurements()[k](0) - sampler.Truth()[k](0), 2);
        eraw = std::sqrt(eraw / 400.);
        CHECK(rec.RMSE(0, sampler.Truth()) < eraw, "route: the combined estimate beats the raw measurement");
    }

    // --- a switching track: the filter should follow the switch ------------------
    {
        // Cruise for 200 steps, then manoeuvre hard for 200. Built by hand so the
        // regime boundary is known exactly.
        //
        // One track is a coin toss: of its 100 manoeuvring steps the agile mode
        // wins anywhere from ~45 to ~100 (300 seeds: mean 76, sd 11.5), so a
        // threshold on one draw passes or fails with the random stream -- which
        // libstdc++ and libc++ produce differently from the same seed. The checks
        // are on the mean over 20 tracks, which moves by a few steps, not tens.
        const std::size_t nTracks = 20;
        const double dt = 1. / 50.;
        double before = 0., after = 0., agileWins = 0.;
        bool nisShape = true, finite = true;
        for (std::uint64_t seed = 0; seed < nTracks; ++seed)
        {
            std::mt19937_64 rng(seed);
            std::normal_distribution<double> gauss(0., 1.);

            std::vector<Vector> zk;
            double p = 0., v = 2.;
            for (std::size_t k = 0; k < 400; ++k)
            {
                const double sigmaA = (k < 200) ? 0.05 : 50.0;   // the regime change
                const double a = sigmaA * gauss(rng);
                p += dt * v + 0.5 * a * dt * dt;
                v += dt * a;

                Vector z(1);
                z(0) = p + sigmaZ * gauss(rng);
                zk.push_back(z);
            }

            Vector x0(2); x0(0) = 0.; x0(1) = 2.;
            quiet.SetInitialState(x0);
            agile.SetInitialState(x0);

            Routers::Bayesian router;
            Bank ikf({&quiet, &agile}, Pi, &router);
            ikf.Forward(zk);

            const Matrix modes = ikf.ModeHistory();
            for (std::size_t k = 100; k < 200; ++k) before += modes(k, 1) / 100.;   // agile share, cruising
            for (std::size_t k = 300; k < 400; ++k) after += modes(k, 1) / 100.;    // agile share, manoeuvring

            // A single step can go either way on noise; the question is which mode
            // wins across the manoeuvre.
            for (std::size_t k = 300; k < 400; ++k)
                if (ikf.DominantMode(k) == 1) ++agileWins;

            const Matrix nis = ikf.NisHistory();
            nisShape = nisShape && nis.Rows() == 400 && nis.Cols() == 2;
            for (std::size_t k = 0; k < nis.Rows(); ++k)
                for (std::size_t j = 0; j < nis.Cols(); ++j)
                    if (!std::isfinite(nis(k, j))) finite = false;
        }
        before /= nTracks; after /= nTracks; agileWins /= nTracks;

        std::printf("    agile-mode probability: %.3f while cruising -> %.3f while manoeuvring"
                    " (mean of %zu tracks)\n", before, after, nTracks);
        std::printf("    agile mode dominant on %.1f of 100 manoeuvring steps\n", agileWins);
        CHECK(after > before + 0.2, "route: the agile mode gains sharply once the target manoeuvres");
        CHECK(before < 0.45, "route: the quiet mode is preferred while cruising");
        CHECK(after > 0.55, "route: the agile mode dominates during the manoeuvre");
        CHECK(agileWins > 65, "route: DominantMode reports the manoeuvring mode most of the time");
        CHECK(nisShape, "route: NIS history is (T, M)");
        CHECK(finite, "route: every NIS entry is finite");
    }

    // --- the router really is pluggable, and dropouts are handled -------------
    {
        Vector x0(2); x0(0) = 0.; x0(1) = 2.;
        quiet.SetInitialState(x0);
        agile.SetInitialState(x0);

        Sampler sampler(&agile, 99);
        sampler.Generate(300, x0);
        sampler.DropChannel(0, 0.25);

        Routers::Bayesian bayes;
        Routers::Threshold thresh(1, 5, true);
        Routers::Hybrid hybrid(&bayes, &thresh, 0.5);

        for (const Routers::Router *r :
             {static_cast<const Routers::Router *>(&bayes),
              static_cast<const Routers::Router *>(&thresh),
              static_cast<const Routers::Router *>(&hybrid)})
        {
            Bank ikf({&quiet, &agile}, Pi, r);
            const Record rec = ikf.Forward(sampler.Measurements());

            bool ok = true;
            for (std::size_t k = 0; k < rec.NSteps(); ++k)
                if (!std::isfinite(rec.State(k)(0)) || !std::isfinite(rec.Covariance(k)(0, 0)))
                    ok = false;
            CHECK(ok, (std::string("route + ") + r->Title() + ": finite through 25% dropouts").c_str());
        }

        // Swapping the decision layer must not require rebuilding the bank.
        Bank ikf({&quiet, &agile}, Pi, &bayes);
        CHECK(&ikf.Modes().Decision() == &bayes, "route: holds the decision layer it was given");
        ikf.Modes().SetDecision(&thresh);
        CHECK(&ikf.Modes().Decision() == &thresh, "route: SetDecision swaps the decision layer");

        // No router at all means the classical update, not a crash.
        Bank plain({&quiet, &agile}, Pi);
        CHECK(dynamic_cast<const Routers::Bayesian *>(&plain.Modes().Decision()) != nullptr, "route: defaults to the Bayesian decision");
        CHECK(plain.model->DefaultEngine() == EngineKind::Interacting && !plain.filter->GetEngine()->HasSmoother(),
              "route: the interacting engine, which has no smoother");
        {
            Fixture::ConstantVelocity alone(1, 50.);
            bool threw = false;
            try { Engines::Interacting engine; Filter wrong(&alone, &engine); wrong.Forward(sampler.Measurements()); }
            catch (const std::exception &) { threw = true; }
            CHECK(threw, "route: the interacting engine refuses a model that has no modes");
        }
        CHECK(plain.Forward(sampler.Measurements()).NSteps() == 300, "route: the default runs");
    }

    return Check::Summary("05_route");
}
