/**
 * *********************************************
 *
 * \file Onnx.h
 * \brief Header of the Kalman::Routers::NN::Onnx class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::NN::Onnx
 * \brief Executes a policy exported from the Python framework.
 *
 * \details The general inference backend: any of the Python architectures — MLP,
 * CNN, LSTM, Transformer — trains in `modules/framework/routers/nn/` and exports
 * to a single ONNX graph, which this runs. It is the seam between the two
 * implementations, and a *trained policy* is what crosses it, not a language
 * binding: once at load time, rather than once per step.
 *
 * \code{.py}
 * router = NeuralRouter(model=MLP(hidden=32), window=5)
 * ikf.fit(training_zk, labels)
 * router.export_onnx("router.onnx")
 * \endcode
 *
 * \code{.cpp}
 * Kalman::Routers::NN::Onnx policy("router.onnx");
 * Kalman::Routers::Neural   router(&policy, 5);
 * Kalman::Filters::IMM      imm(models, Pi, &router);
 * \endcode
 *
 * \note **This class is the optional part, and it is the only optional part.**
 * ONNX Runtime is reached from here and nowhere else — behind a pImpl, so the
 * header is free of it and consumers never need it on their include path.
 * Kalman::Routers::Neural knows nothing about ONNX; without a runtime it still
 * works with Kalman::Routers::NN::MLP or any model you implement. Constructing
 * *this* class in a build without ONNX Runtime throws, because asking for a
 * specific graph that cannot be executed deserves an error rather than a
 * silently different answer.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_NN_ONNX_H
#define KALMAN_ROUTERS_NN_ONNX_H

#include "Kalman/Routers/NN/Model.h"

#include <memory>
#include <string>

namespace Kalman::Routers::NN
{
    class Onnx : public Model
    {
        private:
            struct Session;                     // ONNX Runtime, hidden
            std::shared_ptr<Session> fSession;  //!< shared so the model stays copyable
            std::string fPath;

        public:
            Onnx() = default;

            /**
             * \brief Load a graph.
             * \param path the `.onnx` file.
             * \throws std::runtime_error when the file cannot be loaded, or when this
             *         build has no ONNX Runtime.
             */
            explicit Onnx(const std::string &path);

            const char *Title() const override { return "ONNX Policy"; }
            const char *Description() const override
            { return "Executes a routing policy trained in Python and exported to ONNX."; }

            /// True when this build has ONNX Runtime at all.
            static bool HasRuntime();

            bool IsReady() const override;
            const std::string &Path() const { return fPath; }

            std::size_t NInputs() const override;
            std::size_t NOutputs() const override;

            Vector operator()(const Vector &features) const override;
    };
}

#endif // KALMAN_ROUTERS_NN_ONNX_H
