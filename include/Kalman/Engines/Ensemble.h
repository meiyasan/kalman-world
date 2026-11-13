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
 * changing another's draws. A caller that draws elsewhere (Python, whose
 * numpy generator makes a seeded run repeatable) hands the standard normals in
 * with SetNormals(), and the engine reads them in place of its own stream.
 *
 * A model whose state is a field implements Engines::EnsembleModel beside
 * Kalman::Model: the forecast and the observation of the whole ensemble at once,
 * its localization tapers, and its process noise as a diagonal, so that nothing
 * of size `n x n` is formed during a cycle.
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
    /**
     * \brief The Gaspari-Cohn taper: fifth order, close to a Gaussian of scale
     * \p L, and exactly zero beyond `2 L`.
     *
     * The localization function of `engines/ensemble.py` (`gaspari_cohn`).
     */
    double GaspariCohn(double distance, double L);

    /**
     * \brief What a model offers the ensemble engine beyond `f` and `h`.
     *
     * \details A model implements this next to Kalman::Model, the way a trackable
     * model implements Tracking::Detector, and the engine finds it with a
     * `dynamic_cast`. The Python hooks of the same names are `forecast(X)`,
     * `observe(X)`, `localization()` and the noise the engine draws from `Q`.
     * Each default does what the engine does without the hook.
     */
    class EnsembleModel
    {
        public:
            virtual ~EnsembleModel() = default;

            /// Advance every member one step, in place (`model.forecast(X)`).
            virtual void Forecast(const Model &model, std::vector<Vector> &members,
                                  const Vector &u, double t) const;
            /// The measurement of every member (`model.observe(X)`): out[s] = h(members[s]).
            virtual void Observe(const Model &model, const std::vector<Vector> &members, double t,
                                 std::vector<Vector> &out) const;
            /// The tapers on `P H^T` (n x m) and `H P H^T` (m x m); false for none.
            virtual bool Localization(Matrix &rhoXY, Matrix &rhoYY) const;
            /// The variances of a diagonal `Q`, without forming it; false to use `Q(t)`.
            virtual bool ProcessNoiseVariances(double t, Vector &variances) const;
    };

    class Ensemble : public Engine
    {
        private:
            std::size_t fNMembers = 50;
            double fInflation = 1.;
            mutable std::mt19937_64 fRandom;
            mutable std::vector<Vector> fMembers;   // (N) state vectors
            std::vector<Vector> fStart;             // the initial ensemble handed in; empty = draw it
            Matrix fRhoXY;                          // (n, m) taper on P H^T, empty = none
            Matrix fRhoYY;                          // (m, m) taper on H P H^T, empty = none
            Matrix fModelRhoXY;                     // the model's own tapers, taken at Reset()
            Matrix fModelRhoYY;
            bool fHasNormals = false;               // draws handed in (SetNormals)
            std::vector<double> fNormals;
            mutable std::size_t fNormalsUsed = 0;

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
             * Pass empty matrices to switch localization off. Tapers set here win
             * over a model's own (EnsembleModel::Localization()).
             */
            void SetLocalization(const Matrix &rhoXY, const Matrix &rhoYY);

            /**
             * \brief The initial ensemble of every run from now on, instead of a draw
             * from x0 / P0 (a climatology, the members a caller already drew).
             * \throws std::invalid_argument unless there are NMembers() of one size.
             * Pass an empty vector to draw again.
             */
            void SetMembers(std::vector<Vector> members);

            /**
             * \brief Standard normal draws to use in place of the engine's own stream.
             *
             * \details Read in numpy's layout: a draw of `rows` values for each of the
             * N members takes the next `rows x N` numbers as a row-major `(rows, N)`
             * block, member `s` reading column `s`, as `rng.standard_normal((rows, N))`
             * does. The forecast draws `n` rows when `Q` is not zero, the analysis `p`
             * rows (the channels present). The initial ensemble is not drawn from them
             * (hand it in with SetMembers()). Running out throws.
             */
            void SetNormals(std::vector<double> normals);
            /// Draws handed in and not yet used.
            std::size_t NormalsLeft() const { return fHasNormals ? fNormals.size() - fNormalsUsed : 0; }

            /// (Re)seed the members: the ensemble handed in, else a draw from the model's x0 / P0.
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

            /**
             * \brief The analysis arithmetic, `X + K (z + eps - Y)`.
             *
             * \details `X` (n x N) the members as columns, `Y` (p x N) their
             * measurements, `R` (p x p), `z` (p), `eps` (p x N) the perturbations,
             * and the tapers already cut to the p channels (null for none). The one
             * implementation behind Analyze() and the `enkf_analysis` kernel that
             * the Python engine calls.
             */
            static Matrix Analysis(const Matrix &X, const Matrix &Y, const Matrix &R, const Vector &z,
                                   const Matrix &eps, const Matrix *rhoXY, const Matrix *rhoYY);

        private:
            /// N draws of `rows` standard normals: the engine's stream, or those handed in.
            std::vector<Vector> Normals(std::size_t rows) const;
            Vector Draw(const Matrix &factor) const;
            void Observations(const Model &model, double t, std::vector<Vector> &Y) const;
    };
}

#endif // KALMAN_ENSEMBLEENGINE_H
