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
 * *********************************************
 */

#ifndef KALMAN_FILTERS_ENKF_H
#define KALMAN_FILTERS_ENKF_H

#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Filter.h"

#include <cstdint>

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

            /// The engine, for the members, inflation and localization.
            Engines::Ensemble &Members() { return *static_cast<Engines::Ensemble *>(GetEngine()); }
            const Engines::Ensemble &Members() const { return *static_cast<const Engines::Ensemble *>(GetEngine()); }
    };
}

#endif // KALMAN_FILTERS_ENKF_H
