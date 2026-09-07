/**
 * *********************************************
 *
 * \file Combinatorial.cc
 * \brief Implementation of the Kalman::Engines::Combinatorial class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace Kalman::Engines
{
    using Tracking::Candidate;
    using Tracking::Event;
    using Tracking::GateContext;
    using Tracking::HitId;

    namespace
    {
        bool AllFinite(const Vector &v)
        {
            for (std::size_t i = 0; i < v.Size(); ++i)
                if (!std::isfinite(v(i))) return false;
            return true;
        }

        bool HasNaN(const Tracking::Layer &layer)
        {
            for (const Vector &z : layer)
                if (!Utils::IsComplete(z)) return true;
            return false;
        }
    }

    Combinatorial::Combinatorial(double chi2Cut, std::size_t maxCandidates, std::size_t maxHoles,
                                 std::size_t minHits, double sharedHitFrac, double holePenalty)
        : fChi2Cut(chi2Cut), fHolePenalty(holePenalty), fMaxCandidates(maxCandidates),
          fMaxHoles(maxHoles), fMinHits(minHits), fSharedHitFrac(sharedHitFrac)
    {
        if (maxCandidates == 0) throw std::invalid_argument("Engines::Combinatorial: the beam needs at least one candidate.");
        if (minHits == 0) throw std::invalid_argument("Engines::Combinatorial: a track needs at least one hit.");
        if (sharedHitFrac < 0. || sharedHitFrac > 1.)
            throw std::invalid_argument("Engines::Combinatorial: sharedHitFrac is a fraction in [0, 1].");
    }

    double Combinatorial::Chi2Quantile99(std::size_t dof)
    {
        // scipy.stats.chi2.ppf(0.99, dof) for dof = 1..10; beyond that the
        // Wilson-Hilferty approximation, which is within 0.1% at these sizes.
        static const double table[] = {6.6349, 9.2103, 11.3449, 13.2767, 15.0863,
                                       16.8119, 18.4753, 20.0902, 21.6660, 23.2093};
        if (dof == 0) return 9.2103;
        if (dof <= 10) return table[dof - 1];
        const double k = static_cast<double>(dof), z = 2.3263;
        return k * std::pow(1. - 2. / (9. * k) + z * std::sqrt(2. / (9. * k)), 3);
    }

    double Combinatorial::GateThreshold(const Model &model) const
    {
        return std::isnan(fChi2Cut) ? Chi2Quantile99(model.NMeasure()) : fChi2Cut;
    }

    double Combinatorial::HoleCost(const Model &model) const
    {
        return std::isnan(fHolePenalty) ? GateThreshold(model) : fHolePenalty;
    }

    const Tracking::Detector &Combinatorial::DetectorOf(const Model &model)
    {
        const auto *detector = dynamic_cast<const Tracking::Detector *>(&model);
        if (!detector)
            throw std::runtime_error("Engines::Combinatorial: '" + model.Name() + "' does not implement "
                                     "Kalman::Tracking::Detector (Radii / SeedState); pass radii and "
                                     "seeds explicitly, or make the model a detector.");
        return *detector;
    }

    // --- the gate ------------------------------------------------------------------

    GateContext Combinatorial::Context(const Model &model, const Matrix &Pk, const Vector &zpred,
                                       const Matrix &H, const Matrix &R) const
    {
        GateContext ctx;
        ctx.zpred = zpred;
        ctx.H = H;
        ctx.R = R;
        Matrix S = Utils::Similarity(H, Pk);
        S += R;
        ctx.Sinv = Utils::Inverse(S);
        ctx.K = Pk * H.Transpose() * ctx.Sinv;
        ctx.P = Utils::JosephForm(ctx.K, H, Pk, R);
        (void)model;
        return ctx;
    }

    GateContext Combinatorial::MakeGateContext(const Model &model, const Vector &xk,
                                               const Matrix &Pk, double t) const
    {
        Vector zpred;
        Matrix H;
        Measure(model, xk, t, zpred, H);
        return Context(model, Pk, zpred, H, model.R(t));
    }

    double Combinatorial::Gate(const Model &model, const Vector &xk, const Matrix &Pk,
                               const Tracking::Hit &z, double t, Vector &xOut, Matrix &POut) const
    {
        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        if (keep.empty())
        {
            xOut = xk;
            POut = Pk;
            return std::numeric_limits<double>::infinity();
        }

        Vector zpred;
        Matrix H;
        Measure(model, xk, t, zpred, H);

        const GateContext ctx = Context(model, Pk, Utils::Compress(zpred, keep),
                                        Utils::CompressRows(H, keep), Utils::CompressBoth(model.R(t), keep));
        const double chi2 = ctx.Score(xk, Utils::Compress(z, keep), xOut);
        POut = ctx.P;
        return chi2;
    }

    // --- the search ------------------------------------------------------------------

    std::vector<Candidate> Combinatorial::Seed(const Model &model, const Event &hits,
                                               const std::vector<double> &radii) const
    {
        const Tracking::Detector &detector = DetectorOf(model);
        const Matrix I = Matrix::Identity(model.NState());

        std::vector<Candidate> seeds;
        for (std::size_t l = 0; l < hits.size(); ++l)
        {
            if (hits[l].empty()) continue;
            const double r = radii.at(l);
            for (std::size_t j = 0; j < hits[l].size(); ++j)
            {
                Candidate c;
                detector.SeedState(hits[l][j], c.x, c.P);
                c.hitIds.push_back(HitId{l, j});
                c.radii.push_back(r);
                c.xk.push_back(c.x);
                c.Pk.push_back(c.P);
                c.zk.push_back(hits[l][j]);
                c.priorX.push_back(c.x);
                c.priorP.push_back(c.P);
                c.Fc.push_back(I);
                c.Faccum = I;
                seeds.push_back(std::move(c));
            }
            break;                                  // seed only from the first populated layer
        }
        return seeds;
    }

    std::vector<Candidate> Combinatorial::Find(const Model &model, const Event &hits,
                                               const std::vector<double> &radiiIn,
                                               const std::vector<Candidate> *seedsIn) const
    {
        const std::vector<double> radii = radiiIn.empty() ? DetectorOf(model).Radii() : radiiIn;
        if (radii.size() < hits.size())
            throw std::runtime_error("Engines::Combinatorial: fewer radii than layers.");
        for (std::size_t l = 0; l < hits.size(); ++l)
            for (const Vector &z : hits[l])
                if (z.Size() != model.NMeasure())
                    throw std::runtime_error("Engines::Combinatorial: a hit on layer " + std::to_string(l)
                                             + " has " + std::to_string(z.Size()) + " channels, expected "
                                             + std::to_string(model.NMeasure()) + ".");

        const std::size_t L = hits.size();
        const Matrix I = Matrix::Identity(model.NState());
        const double cut = GateThreshold(model);
        const double penalty = HoleCost(model);
        const std::vector<Candidate> seeds = seedsIn ? *seedsIn : Seed(model, hits, radii);

        std::vector<Candidate> seedTracks;
        for (const Candidate &seed : seeds)
        {
            std::vector<Candidate> cands{seed};

            // Follow outward from the seed's own layer, whichever that is — Seed()
            // picks the innermost *populated* layer, and a caller may pass seeds of
            // its own. Starting at a fixed layer 1 would re-scan the seeding layer.
            for (std::size_t l = static_cast<std::size_t>(seed.SeedLayer() + 1); l < L; ++l)
            {
                const double r = radii[l];
                const Tracking::Layer &layer = hits[l];
                // NaN channels need the per-hit compression in Gate(); without them the
                // whole layer shares one prepared gate.
                const bool masked = !layer.empty() && HasNaN(layer);

                std::vector<Candidate> grown;
                for (const Candidate &c : cands)
                {
                    Vector x;
                    Matrix P, F;
                    Predict(model, c.x, c.P, nullptr, r, x, P, F);
                    if (!AllFinite(x)) continue;                      // track left the detector

                    const Matrix Faccum = F * c.Faccum;
                    GateContext ctx;
                    if (!masked && !layer.empty()) ctx = MakeGateContext(model, x, P, r);

                    for (std::size_t j = 0; j < layer.size(); ++j)
                    {
                        Vector xk;
                        Matrix Pk;
                        double chi2;
                        if (masked) chi2 = Gate(model, x, P, layer[j], r, xk, Pk);
                        else { chi2 = ctx.Score(x, layer[j], xk); Pk = ctx.P; }
                        if (chi2 < cut)
                        {
                            Candidate nc = c;
                            nc.x = xk; nc.P = Pk;
                            nc.chi2 += chi2; nc.score += chi2;
                            nc.hitIds.push_back(HitId{l, j});
                            nc.radii.push_back(r);
                            nc.xk.push_back(xk); nc.Pk.push_back(Pk);
                            nc.zk.push_back(layer[j]);
                            nc.priorX.push_back(x); nc.priorP.push_back(P);
                            nc.Fc.push_back(Faccum);
                            nc.Faccum = I;
                            grown.push_back(std::move(nc));
                        }
                    }
                    // The hole branch: the track survives a missed layer.
                    if (c.holes < fMaxHoles)
                    {
                        Candidate nc = c;
                        nc.x = x; nc.P = P;
                        ++nc.holes; nc.score += penalty;
                        nc.Faccum = Faccum;
                        grown.push_back(std::move(nc));
                    }
                }
                if (grown.empty()) break;
                std::stable_sort(grown.begin(), grown.end(),
                                 [](const Candidate &a, const Candidate &b) { return a.score < b.score; });
                if (grown.size() > fMaxCandidates) grown.resize(fMaxCandidates);
                cands.swap(grown);
            }

            const Candidate *best = nullptr;
            for (const Candidate &c : cands)
                if (c.NHits() >= fMinHits && (!best || c.Quality() < best->Quality())) best = &c;
            if (best) seedTracks.push_back(*best);
        }

        return Arbitrate(std::move(seedTracks));
    }

    std::vector<Candidate> Combinatorial::Arbitrate(std::vector<Candidate> tracks) const
    {
        std::stable_sort(tracks.begin(), tracks.end(),
                         [](const Candidate &a, const Candidate &b) { return a.Quality() < b.Quality(); });
        std::vector<Candidate> accepted;
        std::set<HitId> used;
        for (Candidate &c : tracks)
        {
            std::set<HitId> ids(c.hitIds.begin(), c.hitIds.end());
            std::size_t shared = 0;
            for (const HitId &id : ids) shared += used.count(id);
            if (static_cast<double>(shared) <= fSharedHitFrac * static_cast<double>(ids.size()))
            {
                used.insert(ids.begin(), ids.end());
                accepted.push_back(std::move(c));
            }
        }
        return accepted;
    }

    std::vector<Candidate> Combinatorial::SmoothTracks(const Model &, const std::vector<Candidate> &tracks) const
    {
        std::vector<Candidate> smoothed;
        smoothed.reserve(tracks.size());
        for (const Candidate &c : tracks)
        {
            const std::size_t N = c.NHits();
            Candidate out = c;
            if (N == 0) { smoothed.push_back(out); continue; }
            for (std::size_t k = N - 1; k-- > 0;)
            {
                Vector xs;
                Matrix Ps;
                Smooth(c.xk[k], c.Pk[k], c.priorX[k + 1], c.priorP[k + 1],
                       out.xk[k + 1], out.Pk[k + 1], c.Fc[k + 1], xs, Ps);
                out.xk[k] = xs;
                out.Pk[k] = Ps;
            }
            smoothed.push_back(std::move(out));
        }
        return smoothed;
    }
}
