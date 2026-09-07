/**
 * *********************************************
 *
 * \file Particle.h
 * \brief Header of the Kalman::Engines::Particle class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Particle
 * \brief Sequential importance resampling (the bootstrap particle filter).
 *
 * \details The one engine that makes no Gaussian assumption at all: the belief is
 * a cloud of `N` weighted samples. Predict() pushes every particle through the
 * true `f` with a draw from `Q`; Update() reweights each by the Gaussian
 * likelihood of the measurement under `h` and `R`, normalises, and resamples
 * systematically when the effective sample size falls below
 * `threshold * N`. Both steps hand back the weighted mean and covariance so the
 * filter's record and diagnostics are unchanged.
 *
 * It is therefore **stateful** — it owns the cloud for the duration of a forward
 * pass — and Kalman::Filter calls Reset() on it when a run starts. It has no RTS
 * smoother (HasSmoother() is false): Filter::Backward() refuses rather than
 * pretending.
 *
 * The noise source is an engine-owned Mersenne twister seeded in the constructor,
 * never a global stream, so adding or removing another stochastic engine in the
 * same process cannot change this one's draws.
 *
 * Mirrors `modules/framework/engines/particle.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_PARTICLEENGINE_H
#define KALMAN_PARTICLEENGINE_H

#include "Kalman/Engine.h"

#include <cstdint>
#include <random>
#include <vector>

namespace Kalman::Engines
{
    class Particle : public Engine
    {
        private:
            std::size_t fNParticles = 500;
            double fThreshold = 0.5;              // resample when ESS < threshold * N
            mutable std::mt19937_64 fRandom;
            mutable std::vector<Vector> fParticles;   // (N) samples of the state
            mutable std::vector<double> fWeights;     // (N) normalised weights
            mutable std::size_t fResamples = 0;       // how often the cloud was resampled

        public:
            /**
             * \param nParticles cloud size.
             * \param threshold  resampling trigger as a fraction of \p nParticles.
             * \param seed       seeds the engine's own random stream.
             */
            explicit Particle(std::size_t nParticles = 500, double threshold = 0.5,
                              std::uint64_t seed = 0);
            ~Particle() override = default;

            EngineKind Kind() const override { return EngineKind::Particle; }
            const char *Title() const override { return "Particle Engine"; }
            const char *Description() const override
            { return "Sequential importance-resampling particle filter (forward-only)."; }

            std::size_t NParticles() const { return fNParticles; }
            double Threshold() const { return fThreshold; }

            /// (Re)seed the cloud from the model's x0 / P0.
            void Reset(const Model &model) override;
            bool HasSmoother() const override { return false; }

            void Propagate(const Model &model, const Vector &x, const Vector *u,
                           double t, Vector &xk, Matrix &F) const override;
            void Measure(const Model &model, const Vector &x, double t,
                         Vector &zpred, Matrix &H) const override;

            /// Every particle through `f` plus a `Q` draw; returns the cloud's moments.
            void Predict(const Model &model, const Vector &x, const Matrix &P,
                         const Vector *u, double t,
                         Vector &xk, Matrix &Pk, Matrix &F) const override;

            /// Reweight by the likelihood of \p z, resample if degenerate; returns the moments.
            std::size_t Update(const Model &model, const Vector &x, const Matrix &P,
                               const Vector &z, double t,
                               Vector &xOut, Matrix &POut) const override;

            /// \throws std::runtime_error always — there is no RTS pass over a cloud.
            void Smooth(const Vector &xPost, const Matrix &PPost,
                        const Vector &xPriorNext, const Matrix &PPriorNext,
                        const Vector &xSmoothNext, const Matrix &PSmoothNext,
                        const Matrix &F,
                        Vector &xSmooth, Matrix &PSmooth) const override;

            // --- the cloud ----------------------------------------------------------
            const std::vector<Vector> &Particles() const { return fParticles; }
            const std::vector<double> &Weights() const { return fWeights; }
            /// `1 / sum w_i^2` — how many particles are effectively carrying the belief.
            double EffectiveSampleSize() const;
            /// Number of resampling events since the last Reset().
            std::size_t Resamples() const { return fResamples; }
            /// Weighted mean and covariance of the current cloud.
            void Estimate(Vector &mean, Matrix &cov) const;

        private:
            void Resample() const;
            Vector Draw(const Matrix &factor) const;
    };
}

#endif // KALMAN_PARTICLEENGINE_H
