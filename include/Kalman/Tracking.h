/**
 * *********************************************
 *
 * \file Tracking.h
 * \brief The vocabulary of track *finding*: hits, layers, candidates, and the detector contract.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Every other engine estimates **one** trajectory from a time series of
 * single measurements. Track finding starts from an *event*: a stack of detector
 * layers, each holding **many** hits (signal and background), and asks which
 * hits belong together. That is a different shape of input and output, so it
 * gets its own types here rather than being forced through Kalman::Record —
 * although each found track ends up as one, indexed by layer radius rather than
 * by time.
 *
 * A model that can be tracked implements Kalman::Tracking::Detector next to
 * Kalman::Model: the layer radii, and how to seed a track from one inner hit.
 * The physics — a helix in a field, a straight line, a drift chamber's
 * r–phi — stays in the application, as always.
 *
 * Mirrors the `TrackCandidate` / `GateContext` of
 * `modules/framework/engines/combinatorial.py` and the `radii` / `seed_state`
 * contract of `modules/apps/tracking/driftchamber.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_TRACKING_H
#define KALMAN_TRACKING_H

#include "Kalman/Constants.h"
#include "Kalman/IO/Record.h"
#include "Kalman/Matrix.h"
#include "Kalman/Model.h"

#include <cstddef>
#include <utility>
#include <vector>

namespace Kalman::Tracking
{
    using Hit = Vector;                   ///< one measurement (m) on a layer
    using Layer = std::vector<Hit>;       ///< every hit on one layer
    using Event = std::vector<Layer>;     ///< the layers, innermost first

    /// Which hit, on which layer.
    struct HitId
    {
        std::size_t layer = 0;
        std::size_t index = 0;

        bool operator==(const HitId &o) const { return layer == o.layer && index == o.index; }
        bool operator<(const HitId &o) const { return layer < o.layer || (layer == o.layer && index < o.index); }
    };

    /**
     * \brief The contract a trackable model adds to Kalman::Model.
     *
     * The engine asks for the layer radii once, and for a seed per hit on the
     * innermost populated layer. Everything else it needs — `f(x, u, t = r)` to
     * propagate to a radius, `h`, `R` — is the ordinary model contract.
     */
    class Detector
    {
        public:
            virtual ~Detector() = default;

            /// Layer radii, innermost first; the station tag `t` handed to `f()` is one of these.
            virtual std::vector<double> Radii() const = 0;

            /// Initial `(x0, P0)` for a track that produced \p hit on an inner layer.
            virtual void SeedState(const Hit &hit, Vector &x0, Matrix &P0) const = 0;
    };

    /**
     * \brief One track hypothesis under construction — a node of the search tree.
     *
     * Carries the current posterior, the accumulated chi-square and score, and,
     * per assigned hit, everything the smoother later needs: the posterior, the
     * prediction it corrected, the hit, the radius, and the transition product
     * since the previous assigned hit (holes fold into it).
     */
    struct Candidate
    {
        Vector x;                         ///< current posterior mean
        Matrix P;                         ///< current posterior covariance
        double chi2 = 0.;                 ///< accumulated measurement chi-square (ranking)
        double score = 0.;                ///< chi2 plus hole penalties (pruning)
        std::size_t holes = 0;

        std::vector<HitId> hitIds;        ///< per assigned hit
        std::vector<double> radii;
        std::vector<Vector> xk;
        std::vector<Matrix> Pk;
        std::vector<Hit> zk;
        std::vector<Vector> priorX;
        std::vector<Matrix> priorP;
        std::vector<Matrix> Fc;           ///< transition from the previous posterior
        Matrix Faccum;                    ///< transition product since the last assigned hit

        std::size_t NHits() const { return zk.size(); }

        /// The layer the search continues *after*: the last one this candidate visited, or -1.
        long SeedLayer() const { return hitIds.empty() ? -1L : static_cast<long>(hitIds.back().layer); }

        /// Sort key for the best track: most hits first, then lowest chi2 per hit.
        std::pair<long, double> Quality() const
        {
            return {-static_cast<long>(NHits()), chi2 / static_cast<double>(NHits() ? NHits() : 1)};
        }

        /// This track as a record, indexed by layer radius rather than time.
        Record ToRecord(const Model &model, StateFlavor flavor = StateFlavor::Posterior) const;
    };

    /**
     * \brief Everything about a gate that depends on the **prediction only**, not on the hit.
     *
     * At one layer a candidate is tested against every hit. The predicted
     * measurement, its Jacobian, the innovation covariance, the gain and — since
     * the Joseph form never mentions `z` — the posterior covariance itself are
     * identical for all of them, so they are computed once per (candidate, layer).
     */
    struct GateContext
    {
        Vector zpred;   ///< predicted measurement h(x)
        Matrix H;       ///< observation Jacobian
        Matrix R;       ///< measurement noise
        Matrix Sinv;    ///< inverse innovation covariance
        Matrix K;       ///< Kalman gain
        Matrix P;       ///< posterior covariance, shared by every hit

        /// Score one complete hit: returns chi2 and writes the updated mean.
        double Score(const Vector &x, const Hit &z, Vector &xPost) const;
    };
}

#endif // KALMAN_TRACKING_H
