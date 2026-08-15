/**
 * *********************************************
 *
 * \file MLP.h
 * \brief Header of the Kalman::Routers::NN::MLP class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::NN::MLP
 * \brief A tanh multilayer perceptron with a softmax head, evaluated directly.
 *
 * \details The inference half of `modules/framework/routers/nn/mlp.py`, which is
 * `Linear → tanh → Linear → tanh → Linear → softmax`. Training stays in Python,
 * where Torch and the optimiser are; this evaluates the finished weights.
 *
 * Its reason to exist is **optionality**. Kalman::Routers::NN::Onnx is the
 * general path — it runs a graph exported from any of the Python architectures —
 * but it needs ONNX Runtime. This one needs nothing, so a build with no optional
 * dependencies at all still has a working learned router rather than only a
 * declaration of one.
 *
 * \code{.cpp}
 * Kalman::Routers::NN::MLP mlp;
 * mlp.AddLayer(W1, b1);      // (hidden, d_in)
 * mlp.AddLayer(W2, b2);      // (hidden, hidden)
 * mlp.AddLayer(W3, b3);      // (n_modes, hidden)
 *
 * Kalman::Routers::Neural router(&mlp, 5);
 * \endcode
 *
 * Layers are applied in the order added: `tanh` after every layer except the
 * last, `softmax` after the last. That matches the Python architecture exactly,
 * so weights lifted from a trained `MLP` produce the same probabilities.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_NN_MLP_H
#define KALMAN_ROUTERS_NN_MLP_H

#include "Kalman/Routers/NN/Model.h"

#include <string>
#include <vector>

namespace Kalman::Routers::NN
{
    class MLP : public Model
    {
        private:
            std::vector<Matrix> fWeights;   // per layer: (out, in)
            std::vector<Vector> fBiases;    // per layer: (out)

        public:
            MLP() = default;

            const char *Title() const override { return "MLP Policy"; }
            const char *Description() const override
            { return "Tanh multilayer perceptron with a softmax head, evaluated from its weights."; }

            /**
             * \brief Append a layer.
             * \param weight (out, in) matrix; \p in must match the previous layer's out.
             * \param bias   (out) vector.
             * \throws std::runtime_error on a shape mismatch — caught here rather than
             *         as a silent garbage prediction later.
             */
            void AddLayer(const Matrix &weight, const Vector &bias);

            void Clear();

            std::size_t NLayers() const { return fWeights.size(); }
            std::size_t NInputs() const override
            { return fWeights.empty() ? 0 : fWeights.front().Cols(); }
            std::size_t NOutputs() const override
            { return fWeights.empty() ? 0 : fWeights.back().Rows(); }
            bool IsReady() const override { return !fWeights.empty(); }

            Vector operator()(const Vector &features) const override;

            /**
             * \brief Load weights from the plain text format written by
             *        `NeuralRouter.export_weights()`.
             *
             * A deliberately dumb format — one layer per block, shape line then rows —
             * so it can be read without a parser dependency and diffed by eye. Use
             * ONNX when the architecture is anything but this MLP.
             */
            void Load(const std::string &path);
    };
}

#endif // KALMAN_ROUTERS_NN_MLP_H
