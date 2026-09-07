/**
 * *********************************************
 *
 * \file PF.h
 * \brief Header of the Kalman::Filters::PF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::PF
 * \brief The particle filter — Kalman::Filter driven by Engines::Particle.
 *
 * \details Forward-only: Backward() throws, because a particle cloud has no RTS
 * smoother. Cloud() reaches the engine for the particles, the weights and the
 * effective sample size. Mirrors `modules/framework/filters/pf.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_PF_H
#define KALMAN_FILTERS_PF_H

#include "Kalman/Engines/Particle.h"
#include "Kalman/Filter.h"

#include <cstdint>

namespace Kalman::Filters
{
    class PF : public Filter
    {
        public:
            explicit PF(Model *model, std::size_t nParticles = 500, double threshold = 0.5,
                        std::uint64_t seed = 0)
                : Filter(model, new Engines::Particle(nParticles, threshold, seed), true) {}

            const char *Title() const override { return "Particle Filter"; }
            const char *Description() const override
            { return "Forward-only workflow over a model driven by the particle engine."; }

            /// The engine, for the cloud itself.
            const Engines::Particle &Cloud() const { return *static_cast<const Engines::Particle *>(GetEngine()); }
    };
}

#endif // KALMAN_FILTERS_PF_H
