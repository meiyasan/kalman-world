/**
 * *********************************************
 *
 * \file Ensemble.h
 * \brief Header of the Kalman::Engines::Ensemble class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Ensemble
 * \brief Perturbed-observation ensemble Kalman filter (Evensen).
 *
 * \details The workhorse of high-dimensional data assimilation. The covariance
 * is never formed during the analysis: `N` members carry it, and the gain is
 * built from their anomalies —
 *
 * \code{.unparsed}
 *     P H^T   = rho_xy o X' Y'^T / (N - 1)        Y' = h(X)'  (observation-space anomalies)
 *     H P H^T = rho_yy o Y' Y'^T / (N - 1)
 *     K       = P H^T (H P H^T + R)^-1
 *     X^a     = X + K (z + eps - h(X)),   eps ~ N(0, R) per member
 * \endcode
 *
 * Multiplicative **inflation** scales the anomalies before each analysis to
 * make up for sampling error; optional **localization** tapers (`rho_xy`,
 * `rho_yy`, e.g. Gaspari–Cohn of the state/observation distances) cut spurious
 * long-range correlations from a small ensemble. Predict() and Update() hand
 * back the ensemble mean and sample covariance so the record and the
 * diagnostics are unchanged; Members() exposes the raw ensemble for workflows
 * that never want the dense `n x n`.
 *
 * Stateful, like Engines::Particle: it holds the ensemble across a forward pass
 * and Kalman::Filter calls Reset() at the start of a run. No RTS smoother.
 *
 * The random stream is engine-owned and seeded in the constructor — the RNG
 * discipline that lets several filter variants run side by side without one
 * changing another's draws.
 *
 * Mirrors `modules/framework/engines/ensemble.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_ENSEMBLEENGINE_H
#define KALMAN_ENSEMBLEENGINE_H

#include "Kalman/Engine.h"

#include <cstdint>
#include <random>
#include <vector>

namespace Kalman::Engines
{
    class Ensemble : public Engine
    {
        private:
            std::size_t fNMembers = 50;
            double fInflation = 1.;
            mutable std::mt19937_64 fRandom;
            mutable std::vector<Vector> fMembers;   // (N) state vectors
            Matrix fRhoXY;                          // (n, m) taper on P H^T, empty = none
            Matrix fRhoYY;                          // (m, m) taper on H P H^T, empty = none

        public:
            /**
             * \param nMembers  ensemble size (typically 10–100).
             * \param inflation multiplicative prior inflation (1 = none).
             * \param seed      seeds the engine's own random stream.
             */
            explicit Ensemble(std::size_t nMembers = 50, double inflation = 1., std::uint64_t seed = 0);
            ~Ensemble() override = default;

            EngineKind Kind() const override { return EngineKind::Ensemble; }
            const char *Title() const override { return "Ensemble Engine"; }
            const char *Description() const override
            { return "Perturbed-observation EnKF: localized, inflated ensemble analysis (forward-only)."; }

            std::size_t NMembers() const { return fNMembers; }
            double Inflation() const { return fInflation; }
            void SetInflation(double a) { fInflation = a; }

            /**
             * \brief Covariance localization tapers.
             * \param rhoXY (n, m) multiplies `P H^T` elementwise.
             * \param rhoYY (m, m) multiplies `H P H^T` elementwise.
             * Pass empty matrices to switch localization off.
             */
            void SetLocalization(const Matrix &rhoXY, const Matrix &rhoYY);

            /// (Re)seed the members from the model's x0 / P0.
            void Reset(const Model &model) override;
            bool HasSmoother() const override { return false; }

            void Propagate(const Model &model, const Vector &x, const Vector *u,
                           double t, Vector &xk, Matrix &F) const override;
            void Measure(const Model &model, const Vector &x, double t,
                         Vector &zpred, Matrix &H) const override;

            /// Forecast(): every member through `f` plus a `Q` draw; returns the moments.
            void Predict(const Model &model, const Vector &x, const Matrix &P,
                         const Vector *u, double t,
                         Vector &xk, Matrix &Pk, Matrix &F) const override;

            /// Analyze(): the perturbed-observation update; returns the moments.
            std::size_t Update(const Model &model, const Vector &x, const Matrix &P,
                               const Vector &z, double t,
                               Vector &xOut, Matrix &POut) const override;

            /// \throws std::runtime_error always — there is no RTS pass over an ensemble.
            void Smooth(const Vector &xPost, const Matrix &PPost,
                        const Vector &xPriorNext, const Matrix &PPriorNext,
                        const Vector &xSmoothNext, const Matrix &PSmoothNext,
                        const Matrix &F,
                        Vector &xSmooth, Matrix &PSmooth) const override;

            // --- the ensemble-native steps, for DA workflows -------------------------
            /// Propagate every member; returns the new mean.
            Vector Forecast(const Model &model, const Vector *u, double t) const;
            /// Perturbed-observation analysis; NaN channels skipped; returns the mean.
            Vector Analyze(const Model &model, const Vector &z, double t) const;

            const std::vector<Vector> &Members() const { return fMembers; }
            Vector Mean() const;
            /// Mean per-state ensemble standard deviation (compare against the RMSE).
            double Spread() const;
            /// Ensemble mean and sample covariance.
            void Estimate(Vector &mean, Matrix &cov) const;

        private:
            Vector Draw(const Matrix &factor) const;
    };
}

#endif // KALMAN_ENSEMBLEENGINE_H
