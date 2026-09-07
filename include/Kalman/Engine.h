/**
 * *********************************************
 *
 * \file Engine.h
 * \brief Header of the Kalman::Engine class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engine
 * \brief The **algorithm** — HOW to infer under uncertainty, one step at a time.
 *
 * \details An engine is stateless. It reads a model, takes a belief `(x, P)` in
 * and hands a new belief back; it never stores a run, and it knows nothing about
 * the order in which steps are driven (that is the filter's job).
 *
 * The class splits into two layers, exactly as in the Python framework:
 *
 * - **Propagate() / Measure()** — the only part that differs between algorithms:
 *   evaluate the dynamics and observation maps and hand back their Jacobians.
 *   Each concrete engine overrides these two.
 * - **Predict() / Update() / Smooth()** — the covariance machinery, shared by
 *   every Jacobian-based engine: inject `Q`, form the gain, apply the Joseph-form
 *   correction, run one reverse RTS step. Written once, here.
 *
 * \code{.cpp}
 * Kalman::Engines::Linear engine;
 * Kalman::Vector xk; Kalman::Matrix Pk, F;
 * engine.Predict(model, x, P, nullptr, t, xk, Pk, F);
 * engine.Update(model, xk, Pk, z, t, x, P);
 * \endcode
 *
 * *********************************************
 */

#ifndef KALMAN_ENGINE_H
#define KALMAN_ENGINE_H

#include "Kalman/Matrix.h"
#include "Kalman/Model.h"
#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Constants.h"

namespace Kalman
{
    class Engine : public Abstract::Metadata
    {
        public:
            Engine() = default;
            virtual ~Engine() = default;

            virtual EngineKind Kind() const = 0;

            /**
             * \brief Start a run: a stateless engine ignores it, a stateful one
             * (particle cloud, ensemble) reseeds itself from the model's x0 / P0.
             * Kalman::Filter calls it whenever a forward pass begins.
             */
            virtual void Reset(const Model &) {}

            /// False for engines whose belief is a sample set: Filter::Backward() then refuses.
            virtual bool HasSmoother() const { return true; }

            // --- per-algorithm hooks -----------------------------------------
            /**
             * \brief Advance the mean one step and report the transition matrix.
             * \param model    the system supplying f / F (or the matrices).
             * \param x        the current mean.
             * \param u        the control vector, or nullptr when uncontrolled.
             * \param t        the time tag of this step.
             * \param[out] xk  the propagated mean.
             * \param[out] F   the transition (Jacobian) used, needed by the smoother.
             */
            virtual void Propagate(const Model &model, const Vector &x, const Vector *u,
                                   double t, Vector &xk, Matrix &F) const = 0;

            /**
             * \brief Predict the measurement and report the observation matrix.
             * \param model       the system supplying h / H.
             * \param x           the mean to predict from.
             * \param t           the time tag of this step.
             * \param[out] zpred  the predicted measurement h(x).
             * \param[out] H      the observation (Jacobian).
             */
            virtual void Measure(const Model &model, const Vector &x, double t,
                                 Vector &zpred, Matrix &H) const = 0;

            // --- shared covariance machinery ---------------------------------
            /**
             * \brief Prediction step: `x = f(x, u)`, `P = F P F^T + L Q L^T`.
             * \param model     the system.
             * \param x, P      the current belief.
             * \param u         the control vector, or nullptr.
             * \param t         the time tag of this step.
             * \param[out] xk   the predicted mean.
             * \param[out] Pk   the predicted covariance.
             * \param[out] F    the transition used.
             */
            virtual void Predict(const Model &model, const Vector &x, const Matrix &P,
                                 const Vector *u, double t,
                                 Vector &xk, Matrix &Pk, Matrix &F) const;

            /**
             * \brief Measurement update in Joseph form; NaN channels of \p z are skipped.
             *
             * \code{.unparsed}
             *     S = H P H^T + R          innovation covariance
             *     K = P H^T S^-1           Kalman gain
             *     x = x + K (z - h(x))     corrected mean
             * \endcode
             *
             * The residual is taken against the true (possibly nonlinear) `h(x)` rather
             * than `H x`, which is what makes the same code correct for the extended
             * engine.
             *
             * \param model      the system.
             * \param x, P       the predicted belief.
             * \param z          the measurement; NaN marks a missing channel.
             * \param t          the time tag of this step.
             * \param[out] xOut  the corrected mean.
             * \param[out] POut  the corrected covariance.
             * \return the number of measurement channels actually used (0 when the whole
             *         measurement was missing, in which case the belief passes through
             *         unchanged — a predict-only step).
             */
            virtual std::size_t Update(const Model &model, const Vector &x, const Matrix &P,
                                       const Vector &z, double t,
                                       Vector &xOut, Matrix &POut) const;

            /**
             * \brief One reverse Rauch–Tung–Striebel step.
             *
             * \code{.unparsed}
             *     C   = P_k F^T inv(P_{k+1|k})              smoother gain
             *     x_k = x_k + C (x^s_{k+1} - x_{k+1|k})
             * \endcode
             *
             * \param xPost, PPost              the forward posterior at step k.
             * \param xPriorNext, PPriorNext    the prediction into step k+1.
             * \param xSmoothNext, PSmoothNext  the already-smoothed estimate at k+1.
             * \param F                         the transition stored at step k+1.
             * \param[out] xSmooth, PSmooth     the smoothed estimate at step k.
             */
            virtual void Smooth(const Vector &xPost, const Matrix &PPost,
                                const Vector &xPriorNext, const Matrix &PPriorNext,
                                const Vector &xSmoothNext, const Matrix &PSmoothNext,
                                const Matrix &F,
                                Vector &xSmooth, Matrix &PSmooth) const;

            /**
             * \brief Innovation `nu = z - h(x)` and its covariance `S`.
             *
             * Post-hoc diagnostics (NIS, gain history) are built from these; NaN
             * channels of \p z propagate into \p nu untouched.
             *
             * \param model     the system.
             * \param x, P      the predicted belief.
             * \param z         the measurement.
             * \param t         the time tag of this step.
             * \param[out] nu   the innovation.
             * \param[out] S    its covariance.
             */
            void Innovation(const Model &model, const Vector &x, const Matrix &P,
                            const Vector &z, double t, Vector &nu, Matrix &S) const;
    };

    /**
     * \typedef Kalman::Estimator
     * \brief Alias of Kalman::Engine.
     *
     * `Estimator` is the truer name for this role — it estimates the state's mean
     * and uncertainty from data. `Engine` is the established name and stays
     * primary; this is a pure alias, identical in every way, matching
     * `KalmanEstimator = KalmanEngine` on the Python side so the two vocabularies
     * do not diverge.
     */
    using Estimator = Engine;
}

#endif // KALMAN_ENGINE_H
