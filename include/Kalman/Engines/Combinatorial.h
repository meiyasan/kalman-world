/**
 * *********************************************
 *
 * \file Combinatorial.h
 * \brief Header of the Kalman::Engines::Combinatorial class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Combinatorial
 * \brief Track *finding* by track following — the Belle II CKF in miniature.
 *
 * \details Where the other engines estimate one trajectory from a time series of
 * single measurements, this one reconstructs **several tracks** from an event
 * whose every layer holds **many hits**. For each seed it follows the track
 * outward layer by layer; at each layer it opens a **branch** per hit that
 * passes a chi-square gate (plus a **hole** branch for detector inefficiency),
 * **prunes** to a bounded beam of best candidates, and finally **arbitrates**
 * shared hits so the reported tracks are disjoint.
 *
 * A single predict/update step is not the algorithm here — the *search* is — so
 * this engine offers Find() and SmoothTracks() over a whole event, and
 * Kalman::Filters::CKF is the thin leaf that names it. The per-step arithmetic is
 * inherited unchanged from Engines::Extended: nonlinear propagation to the next
 * layer radius (`f(x, u, t = r)`), the observation map and its Jacobian, and the
 * RTS step. The additions are the gate and the search around it.
 *
 * Hyper-parameters, the counterpart of the UKF's alpha/beta/kappa:
 *
 * \code{.unparsed}
 *     chi2Cut         per-hit chi-square gate (default: 99% of chi2(m))
 *     maxCandidates   beam width kept per seed after each layer (6)
 *     maxHoles        missed layers allowed on a track (2)
 *     minHits         minimum assigned hits to accept a track (3)
 *     sharedHitFrac   max fraction of a track's hits shared with an accepted one (0.3)
 *     holePenalty     score added per hole (default: one gate's worth)
 * \endcode
 *
 * Mirrors `modules/framework/engines/combinatorial.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_COMBINATORIALENGINE_H
#define KALMAN_COMBINATORIALENGINE_H

#include "Kalman/Engines/Extended.h"
#include "Kalman/Tracking.h"

#include <limits>
#include <vector>

namespace Kalman::Engines
{
    class Combinatorial : public Extended
    {
        private:
            double fChi2Cut = std::numeric_limits<double>::quiet_NaN();      // NaN => from m
            double fHolePenalty = std::numeric_limits<double>::quiet_NaN();  // NaN => one gate
            std::size_t fMaxCandidates = 6;
            std::size_t fMaxHoles = 2;
            std::size_t fMinHits = 3;
            double fSharedHitFrac = 0.3;

        public:
            Combinatorial() = default;
            Combinatorial(double chi2Cut, std::size_t maxCandidates, std::size_t maxHoles,
                          std::size_t minHits, double sharedHitFrac,
                          double holePenalty = std::numeric_limits<double>::quiet_NaN());
            ~Combinatorial() override = default;

            EngineKind Kind() const override { return EngineKind::Combinatorial; }
            const char *Title() const override { return "Combinatorial Kalman Engine"; }
            const char *Description() const override
            { return "EKF track-following with a chi-square gate, beam pruning and hit arbitration."; }
            const char *Version() const override { return "2.0"; }

            // --- hyper-parameters --------------------------------------------------------
            /// The 99% quantile of chi2 with \p dof degrees of freedom (tabulated up to 10).
            static double Chi2Quantile99(std::size_t dof);
            /// The chi-square cut in force for \p model — explicit if given, else 99% of chi2(m).
            double GateThreshold(const Model &model) const;
            /// Score charged for a missed layer — explicit if given, else one gate's worth.
            double HoleCost(const Model &model) const;
            std::size_t MaxCandidates() const { return fMaxCandidates; }
            std::size_t MaxHoles() const { return fMaxHoles; }
            std::size_t MinHits() const { return fMinHits; }
            double SharedHitFrac() const { return fSharedHitFrac; }

            // --- the gate ----------------------------------------------------------------
            /// The hit-independent half of the gate for the prediction (xk, Pk); assumes complete hits.
            Tracking::GateContext MakeGateContext(const Model &model, const Vector &xk,
                                                  const Matrix &Pk, double t) const;
            /**
             * \brief Score and apply one candidate hit; NaN channels are skipped.
             * \return the innovation chi-square, or +infinity when every channel is missing.
             */
            double Gate(const Model &model, const Vector &xk, const Matrix &Pk, const Tracking::Hit &z,
                        double t, Vector &xOut, Matrix &POut) const;

            // --- the search --------------------------------------------------------------
            /**
             * \brief Find tracks in one event.
             * \param model  the system; must also implement Kalman::Tracking::Detector
             *               unless \p seeds are given.
             * \param hits   the layers, innermost first.
             * \param radii  the layer radii; empty to take them from the detector.
             * \param seeds  explicit seeds, or nullptr for one per hit on the innermost populated layer.
             * \return the accepted, arbitrated tracks.
             */
            std::vector<Tracking::Candidate> Find(const Model &model, const Tracking::Event &hits,
                                                  const std::vector<double> &radii = {},
                                                  const std::vector<Tracking::Candidate> *seeds = nullptr) const;

            /// RTS-smooth every track from Find() into new, smoothed candidates.
            std::vector<Tracking::Candidate> SmoothTracks(const Model &model,
                                                          const std::vector<Tracking::Candidate> &tracks) const;

            /// One seed per hit on the innermost non-empty layer.
            std::vector<Tracking::Candidate> Seed(const Model &model, const Tracking::Event &hits,
                                                  const std::vector<double> &radii) const;

            /// Greedy hit competition: accept the best tracks, reject those overlapping too much.
            std::vector<Tracking::Candidate> Arbitrate(std::vector<Tracking::Candidate> tracks) const;

        private:
            Tracking::GateContext Context(const Model &model, const Matrix &Pk, const Vector &zpred,
                                          const Matrix &H, const Matrix &R) const;
            static const Tracking::Detector &DetectorOf(const Model &model);
    };
}

#endif // KALMAN_COMBINATORIALENGINE_H
