/**
 * *********************************************
 *
 * \file Sampler.h
 * \brief Header of the Kalman::Sampler class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Sampler
 * \brief Draws truth and measurements *from a model*, for tests and demos.
 *
 * \details Runs the model forward as a stochastic system rather than as an
 * estimator: the truth is propagated through `f` with process noise drawn from
 * `Q`, and each measurement is `h(x)` plus noise drawn from `R`. Because the same
 * model object supplies both the data and the filter's belief, a filter run
 * against a sampler is a genuine consistency check — the NIS should average the
 * number of measurement channels.
 *
 * \code{.cpp}
 * Kalman::Sampler sampler(&model, 12345);
 * sampler.Generate(1000);
 * Kalman::Filters::KF filter(&model);
 * filter.Forward(sampler.Measurements());
 * \endcode
 *
 * \note Noise is drawn through a Cholesky factor of Q and R, so correlated noise
 * is handled correctly; a covariance that is only positive *semi*-definite — a
 * white-noise-acceleration block is exactly that — is jittered rather than
 * rejected.
 *
 * *********************************************
 */

#ifndef KALMAN_SAMPLER_H
#define KALMAN_SAMPLER_H

#include "Kalman/Matrix.h"
#include "Kalman/Model.h"

#include <cstdint>
#include <random>
#include <vector>

namespace Kalman
{
    class Sampler
    {
        private:
            Model *fModel = nullptr;          // the system (not owned)
            std::mt19937_64 fRandom;          // the noise source

            std::vector<Vector> fTruth;       // (N) the true states
            std::vector<Vector> fMeasure;     // (N) the noisy observations
            std::vector<double> fTime;        // (N) time tags

        public:
            /// \param model the system to sample from; must outlive the sampler.
            /// \param seed  seeds the Mersenne twister, so a run is reproducible.
            explicit Sampler(Model *model, std::uint64_t seed = 0);

            /**
             * \brief Generate \p nsteps of truth and measurements.
             * \param nsteps how many steps to draw.
             * \param x0     starting truth, or an empty vector to start from the model's x0.
             */
            void Generate(std::size_t nsteps, const Vector &x0 = Vector());

            /**
             * \brief Blank out (set to NaN) channel \p channel at random.
             * \param channel     the measurement channel to make unreliable.
             * \param probability chance of dropping it at any given step.
             */
            void DropChannel(std::size_t channel, double probability);

            const std::vector<Vector> &Truth() const { return fTruth; }
            const std::vector<Vector> &Measurements() const { return fMeasure; }
            const std::vector<double> &Times() const { return fTime; }

            std::size_t NSteps() const { return fTruth.size(); }

            std::mt19937_64 &Random() { return fRandom; }

        protected:
            /// A draw from N(0, C) given the Cholesky factor of C.
            Vector Draw(const Matrix &factor);
    };
}

#endif // KALMAN_SAMPLER_H
