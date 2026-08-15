/**
 * *********************************************
 *
 * \file Torch.h
 * \brief Header of the Kalman::Routers::NN::Torch class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::NN::Torch
 * \brief Executes a TorchScript policy through libTorch.
 *
 * \details The closest C++ mirror of `modules/framework/routers/nn/`, whose four
 * architectures are all `torch.nn` modules: this runs the very same module,
 * traced to TorchScript, with no conversion step in between.
 *
 * \code{.py}
 * router = NeuralRouter(model=LSTM(hidden=32), window=5)
 * ikf.fit(training_zk, labels)
 * router.export_torchscript("router.pt")
 * \endcode
 *
 * \code{.cpp}
 * Kalman::Routers::NN::Torch policy("router.pt");
 * Kalman::Routers::Neural    router(&policy, 5);
 * \endcode
 *
 * \section torch_vs_onnx Torch or ONNX?
 *
 * Both are optional, both satisfy Kalman::Routers::NN::Model, and the router
 * cannot tell them apart. They differ in what they cost you:
 *
 * | | TorchScript (this) | ONNX (Kalman::Routers::NN::Onnx) |
 * |---|---|---|
 * | Fidelity | the *same* module, traced | a converted graph |
 * | Coverage | anything `torch.jit.trace` accepts | anything the exporter supports |
 * | Runtime  | libTorch (~2 GB) | ONNX Runtime (~10 MB) |
 * | Reach    | Torch only | any ONNX runtime, any language |
 *
 * Prefer TorchScript when the architecture is exotic enough that the ONNX
 * exporter mangles it, or when libTorch is linked anyway — root-plus' Tensor
 * component already carries it. Prefer ONNX for deployment, where the runtime is
 * a tenth of the size and does not tie the binary to a Torch version.
 *
 * \note ONNX and libTorch are independent options. Neither is required, both may
 * be present, and Kalman::Routers::NN::MLP needs neither — so the routing layer
 * works in a build with no optional dependency at all.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_NN_TORCH_H
#define KALMAN_ROUTERS_NN_TORCH_H

#include "Kalman/Routers/NN/Model.h"

#include <memory>
#include <string>

namespace Kalman::Routers::NN
{
    class Torch : public Model
    {
        private:
            struct Module;                     // libTorch, hidden behind pImpl
            std::shared_ptr<Module> fModule;   //!< shared so the model stays copyable
            std::string fPath;
            std::size_t fOutputs = 0;

        public:
            Torch() = default;

            /**
             * \brief Load a TorchScript module.
             * \param path the traced `.pt` file.
             * \throws std::runtime_error when it cannot be loaded, or when this build
             *         has no libTorch.
             */
            explicit Torch(const std::string &path);

            const char *Title() const override { return "TorchScript Policy"; }
            const char *Description() const override
            { return "Executes a routing policy traced from torch.nn through libTorch."; }

            /// True when this build has libTorch at all.
            static bool HasRuntime();

            bool IsReady() const override;
            const std::string &Path() const { return fPath; }

            /// Modes the module produces; discovered on the first call, 0 before that.
            std::size_t NOutputs() const override { return fOutputs; }

            Vector operator()(const Vector &features) const override;
    };
}

#endif // KALMAN_ROUTERS_NN_TORCH_H
