/**
 * *********************************************
 *
 * \file Model.h
 * \brief Header of the Kalman::Routers::NN::Model contract
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::NN::Model
 * \brief A learned policy: features in, mode probabilities out.
 *
 * \details The C++ counterpart of `modules/framework/routers/nn/`, and it keeps
 * that package's division of labour exactly:
 *
 * - **The router** (Kalman::Routers::Neural) owns feature extraction and
 *   normalisation — the NIS window, the front-padding, the standardisation.
 * - **The model** owns the architecture and the forward pass.
 *
 * That split is what makes the architecture swappable without touching the
 * router, and it is why this contract is one method wide. Python states it as
 * "any callable ``model(features) → probs`` works"; here it is:
 *
 * \code{.cpp}
 * Vector probabilities = model(features);
 * \endcode
 *
 * \section nn_training Training lives in Python
 *
 * Python's models carry `build()` and `fit()` alongside `__call__`, because that
 * is where training belongs — the data loaders, the optimiser, the notebooks.
 * This contract has **no** training hooks, deliberately: a model here is always
 * already trained. Two ways to get one:
 *
 * - Kalman::Routers::NN::Onnx — execute a policy exported from any of the Python
 *   architectures (MLP, CNN, LSTM, Transformer). Needs ONNX Runtime.
 * - Kalman::Routers::NN::MLP — evaluate a two-layer tanh MLP directly from its
 *   weights. Needs nothing at all, which is what keeps the routing layer usable
 *   in a build with no optional dependencies whatsoever.
 *
 * Implement this interface for anything else; the router will not notice.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_NN_MODEL_H
#define KALMAN_ROUTERS_NN_MODEL_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Matrix.h"

namespace Kalman::Routers::NN
{
    class Model : public Abstract::Metadata
    {
        public:
            ~Model() override = default;

            /**
             * \brief Forward pass.
             * \param features standardised feature vector of length NInputs().
             * \return one probability per mode, length NOutputs().
             *
             * Implementations return probabilities, not logits: the softmax belongs
             * to the architecture, and putting it here rather than in the router is
             * what lets a model whose head is *already* a softmax — an exported ONNX
             * graph, say — satisfy the same contract without a second one applied on
             * top.
             */
            virtual Vector operator()(const Vector &features) const = 0;

            /// Features the model expects, or 0 when it does not care / cannot say.
            virtual std::size_t NInputs() const { return 0; }

            /// Modes the model produces, or 0 when unknown.
            virtual std::size_t NOutputs() const { return 0; }

            /// True when the model can actually be evaluated.
            virtual bool IsReady() const { return true; }
    };
}

#endif // KALMAN_ROUTERS_NN_MODEL_H
