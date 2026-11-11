/**
 * *********************************************
 *
 * \file Neural.h
 * \brief Header of the Kalman::Routers::Neural class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::Neural
 * \brief Delegates the mode decision to a learned policy.
 *
 * \details Mirrors `modules/framework/routers/neural.py`, including the division
 * of labour that file states in its first two lines:
 *
 * - **This class owns feature extraction and normalisation** — the NIS window,
 *   the front-padding, the standardisation learned at training time.
 * - **The model owns the architecture and the forward pass** — see
 *   Kalman::Routers::NN::Model.
 *
 * \code{.cpp}
 * Kalman::Routers::NN::Onnx policy("router.onnx");   // or NN::MLP, or your own
 * Kalman::Routers::Neural   router(&policy, 5);
 * Kalman::Filters::IKF      ikf(models, Pi, &router);
 * \endcode
 *
 * \note There is no ONNX in this header, and none in its implementation. The
 * runtime is confined to Kalman::Routers::NN::Onnx, so this router works in a
 * build with no optional dependencies at all — with NN::MLP, or with any model
 * you implement. That is what "optional" has to mean to be worth anything: not a
 * class that throws when the dependency is missing, but a layer that still
 * functions without it.
 *
 * \section neural_features Features
 *
 * The last `window` NIS rows, front-padded with 1.0 (the chi2(1) mean) and
 * flattened row-major to `M × window`, then standardised. The layout is fixed by
 * `routers/neural.py` and must match bit for bit — a mismatch does not fail, it
 * quietly returns a meaningless probability vector.
 *
 * Standardisation is applied only when SetNormalisation() has been given
 * matching-length vectors. `NeuralRouter.export_onnx()` bakes it into the graph
 * by default, in which case there is nothing to set here and nothing to get out
 * of step.
 *
 * \note Until a model is attached, Route() falls back to the classical Bayesian
 * update rather than failing. Check IsReady() if you need to know which one you
 * are getting.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_NEURAL_H
#define KALMAN_ROUTERS_NEURAL_H

#include "Kalman/Routers/Base.h"
#include "Kalman/Routers/NN/Model.h"

namespace Kalman::Routers
{
    class Neural : public Router
    {
        private:
            const NN::Model *fModel = nullptr;   // not owned
            std::size_t fWindow = 5;
            Vector fMean;                        // feature standardisation, from training
            Vector fScale;

        public:
            Neural() = default;

            /**
             * \param model  the policy; not owned, and must outlive the router.
             * \param window NIS window length; must match the one used at training.
             */
            explicit Neural(const NN::Model *model, std::size_t window = 5);

            const char *Title() const override { return "Neural Router"; }
            const char *Description() const override
            { return "IKF routing delegated to a learned policy; owns the NIS features."; }

            /// True when a model is attached and evaluable.
            bool IsReady() const { return fModel && fModel->IsReady(); }

            const NN::Model *GetModel() const { return fModel; }
            /// Swap the policy; not owned.
            void SetModel(const NN::Model *model) { fModel = model; }

            std::size_t Window() const { return fWindow; }
            void SetWindow(std::size_t window);

            /**
             * \brief Feature standardisation recorded at training time.
             *
             * Only needed when the graph does *not* carry it — `export_onnx()` bakes
             * it in by default. Applying the wrong one is silent: the policy still
             * returns a probability vector, just a meaningless one.
             */
            void SetNormalisation(const Vector &mean, const Vector &scale);
            const Vector &Mean() const { return fMean; }
            const Vector &Scale() const { return fScale; }

            /// The flattened, standardised vector this router would feed the policy.
            Vector Features(const Matrix &nis) const;

            Vector Route(const Matrix &nis, const Vector &c,
                         const Vector &likelihoods) const override;
    };
}

#endif // KALMAN_ROUTERS_NEURAL_H
