/**
 * *********************************************
 *
 * \file CKF.h
 * \brief Header of the Kalman::Filters::CKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::CKF
 * \brief The combinatorial Kalman filter — track finding over an event.
 *
 * \details Not a Kalman::Filter: its input is an event of many hits per layer,
 * not a time series, and its output is a collection of tracks, not one record.
 * The search — seeding, gating, branching, pruning, arbitration, and the
 * smoother over the found tracks — *is* the algorithm, so it lives in
 * Kalman::Engines::Combinatorial; this class names that engine, hands it the
 * search hyper-parameters, and turns what it returns into records indexed by
 * layer radius.
 *
 * \code{.cpp}
 * DriftChamber chamber(8, 0.05);            // a Model that is also a Tracking::Detector
 * auto event = chamber.GenerateEvent(3, 4, 0);
 * Kalman::Filters::CKF ckf(&chamber);
 * Kalman::RecordCollection tracks = ckf.Find(event.hits);   // one record per track
 * tracks = ckf.Smooth();                                     // RTS over each
 * \endcode
 *
 * Mirrors `modules/framework/filters/ckf.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_CKF_H
#define KALMAN_FILTERS_CKF_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/IO/Collection.h"
#include "Kalman/Model.h"
#include "Kalman/Tracking.h"

#include <limits>
#include <vector>

namespace Kalman::Filters
{
    class CKF : public Abstract::Metadata
    {
        private:
            Model *fModel = nullptr;                          // not owned
            Engines::Combinatorial fEngine;
            std::vector<Tracking::Candidate> fTracks;         // the search result
            bool fSmoothed = false;

        public:
            /**
             * \param model the system, which must also implement Kalman::Tracking::Detector
             *              unless radii and seeds are passed to Find(); must outlive the filter.
             * \param chi2Cut, maxCandidates, maxHoles, minHits, sharedHitFrac, holePenalty
             *              the search hyper-parameters (NaN = the engine's default).
             */
            explicit CKF(Model *model,
                         double chi2Cut = std::numeric_limits<double>::quiet_NaN(),
                         std::size_t maxCandidates = 6, std::size_t maxHoles = 2, std::size_t minHits = 3,
                         double sharedHitFrac = 0.3,
                         double holePenalty = std::numeric_limits<double>::quiet_NaN());
            ~CKF() override = default;

            const char *Title() const override { return "Combinatorial Kalman Filter"; }
            const char *Description() const override
            { return "Layer-by-layer track following with branching, pruning and hit arbitration."; }

            Model *GetModel() const { return fModel; }
            Engines::Combinatorial &GetEngine() { return fEngine; }
            const Engines::Combinatorial &GetEngine() const { return fEngine; }

            /**
             * \brief Find tracks in \p hits; returns them as records (posterior).
             * \param hits  the event: the hits of each layer, innermost first.
             * \param radii layer radii, or empty to take them from the detector.
             */
            RecordCollection Find(const Tracking::Event &hits, const std::vector<double> &radii = {});

            /// RTS-smooth every found track; returns the smoothed records.
            RecordCollection Smooth();

            /// The tracks as found (or smoothed), with the search bookkeeping the records drop.
            const std::vector<Tracking::Candidate> &Tracks() const { return fTracks; }
            std::size_t NTracks() const { return fTracks.size(); }

            // --- the resolved hyper-parameters --------------------------------------
            double Chi2Cut() const { return fEngine.GateThreshold(*fModel); }
            double HolePenalty() const { return fEngine.HoleCost(*fModel); }

        private:
            RecordCollection Records(StateFlavor flavor) const;
    };
}

#endif // KALMAN_FILTERS_CKF_H
