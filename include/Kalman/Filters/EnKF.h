/**
 * *********************************************
 *
 * \file EnKF.h
 * \brief Header of the Kalman::Filters::EnKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::EnKF
 * \brief The ensemble Kalman filter — Kalman::Filter driven by Engines::Ensemble.
 *
 * \details Forward-only: Backward() throws. Members() reaches the engine for
 * the ensemble, the inflation and the localization tapers. Mirrors
 * `modules/framework/filters/enkf.py`.
 *
 * Forward() records what every filter records, the mean and the covariance of
 * each step. A field of `n` cells makes that covariance `n x n`, which a data-
 * assimilation run never wants: Cycle() runs the same forecast and analysis
 * steps and keeps the means and the spread only (the loop of the Python
 * `TwinExperiment`).
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_ENKF_H
#define KALMAN_FILTERS_ENKF_H

#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Filter.h"

#include <cstdint>
#include <vector>

namespace Kalman::Filters
{
    class EnKF : public Filter
    {
        public:
            explicit EnKF(Model *model, std::size_t nMembers = 50, double inflation = 1.,
                          std::uint64_t seed = 0)
                : Filter(model, new Engines::Ensemble(nMembers, inflation, seed), true) {}

            const char *Title() const override { return "Ensemble Kalman Filter"; }
            const char *Description() const override
            { return "Forward-only workflow over a model driven by the ensemble engine."; }

            /// Over an engine the caller owns (its members, draws and tapers already set).
            EnKF(Model *model, Engines::Ensemble *engine) : Filter(model, engine, false) {}

            /// What Cycle() keeps of each step.
            struct Cycles
            {
                std::vector<Vector> forecast;    ///< the mean after the forecast
                std::vector<Vector> analysis;    ///< the mean after the analysis (the forecast's when none)
                std::vector<double> spread;      ///< Engines::Ensemble::Spread() after the step
            };

            /**
             * \brief One forecast per measurement, then its analysis when \p analyse,
             * from a fresh Reset(); no covariance is formed.
             * \param zk      the measurements (NaN marks a silent channel).
             * \param analyse false for a free run (forecasts only).
             */
            Cycles Cycle(const std::vector<Vector> &zk, bool analyse = true);

            /// The engine, for the members, inflation and localization.
            Engines::Ensemble &Members() { return *static_cast<Engines::Ensemble *>(GetEngine()); }
            const Engines::Ensemble &Members() const { return *static_cast<const Engines::Ensemble *>(GetEngine()); }
    };
}

#endif // KALMAN_FILTERS_ENKF_H
