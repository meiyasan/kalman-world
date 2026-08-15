/**
 * *********************************************
 *
 * \file IMM.h
 * \brief Header of the Kalman::Filters::IMM class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::IMM
 * \brief Interacting Multiple Model filter — several models, one estimate.
 *
 * \details When no single model covers a system's whole behaviour, run all of
 * them and carry a probability on each: a target that cruises *and* manoeuvres
 * needs both descriptions, and neither alone. The IMM mixes the modes at every
 * step so a model that becomes relevant inherits the state the others have built
 * up, rather than starting cold. The modes themselves are yours — the framework
 * supplies the machinery, not the regimes.
 *
 * One step:
 *
 * \code{.unparsed}
 *   1. Markov prediction   c_j = sum_i Pi(i,j) mu_i
 *   2. State mixing        x0_j = sum_i mu_ij x_i,  P0_j likewise + spread
 *   3. Per model           predict + update  ->  nu_j, S_j, NIS_j, L_j
 *   4. Router              mu = router(nis, c, likelihoods)     <- pluggable
 *   5. Combination         x = sum_j mu_j x_j,  P likewise + spread
 * \endcode
 *
 * Step 4 is the only decision, and it is delegated to a Kalman::Routers::Router:
 * classical Bayes, a NIS rule, a blend, or a policy trained in Python and run
 * through ONNX. Everything else is fixed by the algorithm.
 *
 * \code{.cpp}
 * Kalman::Routers::Bayesian router;
 * Kalman::Filters::IMM imm({&cruise, &manoeuvre}, Pi, &router);
 *
 * auto record = imm.Forward(measurements);
 * const Kalman::Matrix &modes = imm.ModeHistory();   // (T, M) probabilities
 * \endcode
 *
 * \note Neither the models nor the router are owned; all must outlive the filter.
 * Every model must share the same state and measurement dimensions — that is what
 * makes mixing meaningful — and each contributes its own declared engine, so a
 * linear and a nonlinear mode can sit side by side.
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_IMM_H
#define KALMAN_FILTERS_IMM_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Engine.h"
#include "Kalman/IO/Record.h"
#include "Kalman/Model.h"
#include "Kalman/Routers/Base.h"

#include <memory>
#include <vector>

namespace Kalman::Filters
{
    class IMM : public Abstract::Metadata
    {
        private:
            std::vector<Model *> fModels;                    // not owned
            std::vector<std::unique_ptr<Engine>> fEngines;   // one per model, built here
            const Routers::Router *fRouter = nullptr;        // not owned
            std::unique_ptr<Routers::Router> fOwnRouter;     // the Bayesian default

            Matrix fPi;                     // (M,M) Markov transition
            Vector fMu;                     // (M) current mode probabilities

            std::vector<Vector> fX;         // (M) per-mode mean
            std::vector<Matrix> fP;         // (M) per-mode covariance

            Matrix fNis;                    // (T,M) NIS history
            Matrix fModeHistory;            // (T,M) mode probabilities per step
            std::vector<Vector> fCombined;  // (T) combined mean
            std::vector<Matrix> fCombinedP; // (T) combined covariance
            std::vector<double> fTime;

        public:
            /**
             * \param models a mode per entry; all must share n and m, none is owned.
             * \param Pi     (M,M) Markov transition, rows summing to 1.
             * \param router the decision layer, or nullptr for Kalman::Routers::Bayesian.
             */
            IMM(const std::vector<Model *> &models, const Matrix &Pi,
                const Routers::Router *router = nullptr);

            ~IMM() override = default;

            IMM(const IMM &) = delete;
            IMM &operator=(const IMM &) = delete;

            const char *Title() const override { return "Interacting Multiple Model Filter"; }
            const char *Description() const override
            { return "Runs M models in parallel, mixing their states and routing between them."; }

            std::size_t NModes() const { return fModels.size(); }
            std::size_t NState() const { return fModels.empty() ? 0 : fModels.front()->NState(); }

            /// Reset the modes to a uniform prior and each model's own x0 / P0.
            void Reset();

            /// Set the initial mode probabilities (normalised on the way in).
            void SetModeProbabilities(const Vector &mu);

            const Routers::Router *GetRouter() const { return fRouter; }
            /// Swap the decision layer; not owned, must outlive the filter.
            void SetRouter(const Routers::Router *router);

            /**
             * \brief Run the forward pass.
             * \param zk one measurement per step; NaN channels are skipped as usual.
             * \return the combined estimate, as a record.
             */
            Record Forward(const std::vector<Vector> &zk);

            /// Current mode probabilities.
            const Vector &ModeProbabilities() const { return fMu; }
            /// `(T, M)` mode probability per step — the answer to "which regime, when".
            const Matrix &ModeHistory() const { return fModeHistory; }
            /// `(T, M)` NIS per mode per step, the feature the routers consume.
            const Matrix &NisHistory() const { return fNis; }

            /// Index of the most probable mode at step \p k.
            std::size_t DominantMode(std::size_t k) const;

        protected:
            /// Gaussian innovation likelihood `N(nu; 0, S)`.
            static double Likelihood(const Vector &nu, const Matrix &S);
    };
}

#endif // KALMAN_FILTERS_IMM_H
