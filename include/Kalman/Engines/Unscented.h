/**
 * *********************************************
 *
 * \file Unscented.h
 * \brief Header of the Kalman::Engines::Unscented class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Unscented
 * \brief The sigma-point engine (UKF) — nonlinear moments without a Jacobian.
 *
 * \details Where Engines::Extended linearises `f` and `h` at the mean, this engine
 * pushes a deterministic set of `2n + 1` sigma points through the true maps and
 * recovers mean and covariance from their weighted spread (the unscented
 * transform of Julier & Uhlmann). Nothing is differentiated, so a model with a
 * poor or expensive Jacobian — or none — is handled at second-order accuracy.
 *
 * The noise is **additive**: `P = sum Wc (Y - x)(Y - x)^T + L Q L^T` on
 * prediction and `S = sum Wc (Z - z)(Z - z)^T + R` on update, which is the form
 * Kalman::Model's `f(x, u, t)` / `h(x, t)` contract supports. The Python engine
 * additionally augments the state for non-additive noise; a C++ model that needs
 * that folds the noise into its own `f`.
 *
 * `alpha`, `beta`, `kappa` are the usual spread parameters: `alpha` scales the
 * points around the mean (small, e.g. 1e-3, for a tight cloud), `beta` folds in
 * prior knowledge of the distribution (2 is optimal for a Gaussian), `kappa` is a
 * secondary scale (0 by default).
 *
 * The transition reported to the smoother is the identity, as in the Python
 * engine: a sigma-point pass has no single linearisation to hand back, and the
 * RTS step still runs on it.
 *
 * Mirrors `modules/framework/engines/unscented.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_UNSCENTEDENGINE_H
#define KALMAN_UNSCENTEDENGINE_H

#include "Kalman/Engine.h"

#include <vector>

namespace Kalman::Engines
{
    class Unscented : public Engine
    {
        private:
            double fAlpha = 1e-3;
            double fBeta = 2.;
            double fKappa = 0.;

        public:
            explicit Unscented(double alpha = 1e-3, double beta = 2., double kappa = 0.);
            ~Unscented() override = default;

            EngineKind Kind() const override { return EngineKind::Unscented; }
            const char *Title() const override { return "Unscented Kalman Engine"; }
            const char *Description() const override
            { return "Sigma-point (unscented transform) step; no Jacobian; noise added, or carried through the maps."; }
            const char *Version() const override { return "3.0"; }

            double Alpha() const { return fAlpha; }
            double Beta() const { return fBeta; }
            double Kappa() const { return fKappa; }

            /// Mean through the true map; the reported transition is the identity.
            void Propagate(const Model &model, const Vector &x, const Vector *u,
                           double t, Vector &xk, Matrix &F) const override;

            /// `h(x)` and the observation Jacobian, for the innovation diagnostics.
            void Measure(const Model &model, const Vector &x, double t,
                         Vector &zpred, Matrix &H) const override;

            /// Sigma points through `f`; covariance from their spread plus `L Q L^T`.
            void Predict(const Model &model, const Vector &x, const Matrix &P,
                         const Vector *u, double t,
                         Vector &xk, Matrix &Pk, Matrix &F) const override;

            /// Sigma points through `h`; gain from the cross-covariance. NaN channels skipped.
            std::size_t Update(const Model &model, const Vector &x, const Matrix &P,
                               const Vector &z, double t,
                               Vector &xOut, Matrix &POut) const override;

            // --- the transform itself, exposed for tests and for reuse -------------
            /// `gamma` and the mean / covariance weights for a state of dimension \p n.
            void Weights(std::size_t n, double &gamma, std::vector<double> &wm,
                         std::vector<double> &wc) const;

            /**
             * \brief The `2n + 1` sigma points of `N(mean, P)`.
             * \throws std::runtime_error when \p P has no Cholesky factor even after jitter.
             */
            static std::vector<Vector> SigmaPoints(const Vector &mean, const Matrix &P, double gamma);
    };
}

#endif // KALMAN_UNSCENTEDENGINE_H
