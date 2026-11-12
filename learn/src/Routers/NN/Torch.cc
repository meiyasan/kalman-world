/**
 * *********************************************
 *
 * \file Torch.cc
 * \brief Implementation of the Kalman::Routers::NN::Torch class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * libTorch is reached from this file and nowhere else, behind the pImpl
 * `Module`. Kalman::Matrix is never routed through a tensor: a Torch op carries
 * a ~230 ns dispatch floor, which is more than an entire small matrix product,
 * so tensors are for the *policy* and nothing else.
 */

#include "Kalman/Routers/NN/Torch.h"
#include "Kalman/Config.h"

// Set by learn/CMakeLists.txt when the runtime was found.
#ifndef KALMAN_WITH_TORCH
#  define KALMAN_WITH_TORCH 0
#endif

#include <sstream>
#include <stdexcept>

#if KALMAN_WITH_TORCH
#  include <torch/script.h>
#endif

namespace Kalman::Routers::NN
{
#if KALMAN_WITH_TORCH

    struct Torch::Module
    {
        torch::jit::script::Module module;

        explicit Module(const std::string &path)
        {
            module = torch::jit::load(path);
            module.eval();
        }
    };

    bool Torch::HasRuntime() { return true; }

    Torch::Torch(const std::string &path) : fPath(path)
    {
        try
        {
            fModule = std::make_shared<Module>(path);
        }
        catch (const c10::Error &e)
        {
            std::ostringstream os;
            os << "Kalman::Routers::NN::Torch: could not load '" << path << "': " << e.what();
            throw std::runtime_error(os.str());
        }
    }

    bool Torch::IsReady() const { return fModule != nullptr; }

    Vector Torch::operator()(const Vector &features) const
    {
        if (!IsReady())
            throw std::runtime_error("Kalman::Routers::NN::Torch: no module is loaded.");

        torch::NoGradGuard noGrad;

        // (1, d_in): the traced module keeps its batch axis, as the Python side had.
        std::vector<float> in(features.Size());
        for (std::size_t i = 0; i < features.Size(); ++i) in[i] = static_cast<float>(features(i));

        at::Tensor input = torch::from_blob(
            in.data(), {1, static_cast<std::int64_t>(in.size())}, torch::kFloat32).clone();

        at::Tensor out;
        try
        {
            out = fModule->module.forward({input}).toTensor().contiguous().to(torch::kFloat32);
        }
        catch (const c10::Error &e)
        {
            std::ostringstream os;
            os << "Kalman::Routers::NN::Torch: forward failed on a " << features.Size()
               << "-feature input: " << e.what();
            throw std::runtime_error(os.str());
        }

        const std::size_t n = static_cast<std::size_t>(out.numel());
        const float *data = out.data_ptr<float>();

        Vector mu(n);
        for (std::size_t j = 0; j < n; ++j) mu(j) = static_cast<double>(data[j]);

        const_cast<Torch *>(this)->fOutputs = n;   // discovered, not declared
        return mu;
    }

#else  // ---------------------------------------------------------------------

    /// Placeholder so the type stays complete without libTorch.
    struct Torch::Module
    {
    };

    bool Torch::HasRuntime() { return false; }

    Torch::Torch(const std::string &path) : fPath(path)
    {
        // Same rule as the ONNX backend: asking for a *specific* module that cannot
        // be executed is an error. The graceful degradation lives one level up, in
        // Kalman::Routers::Neural.
        throw std::runtime_error(
            "Kalman::Routers::NN::Torch: this build has no libTorch, so '" + path +
            "' cannot be executed. Rebuild with libTorch on CMAKE_PREFIX_PATH, use "
            "Kalman::Routers::NN::Onnx, or use Kalman::Routers::NN::MLP, which needs nothing.");
    }

    bool Torch::IsReady() const { return false; }

    Vector Torch::operator()(const Vector &) const
    {
        throw std::runtime_error("Kalman::Routers::NN::Torch: built without libTorch.");
    }

#endif
}
