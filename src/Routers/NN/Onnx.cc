/**
 * *********************************************
 *
 * \file Onnx.cc
 * \brief Implementation of the Kalman::Routers::NN::Onnx class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * ONNX Runtime is reached from this file and nowhere else in libKalmanFramework,
 * and only through the pImpl `Session`. That is what makes the dependency
 * genuinely optional: without it this one class is inert and everything else --
 * including Kalman::Routers::Neural -- carries on.
 */

#include "Kalman/Routers/NN/Onnx.h"
#include "Kalman/Config.h"

#include <sstream>
#include <stdexcept>

#if KALMAN_WITH_ONNXRUNTIME
#  include <onnxruntime_cxx_api.h>
#endif

namespace Kalman::Routers::NN
{
#if KALMAN_WITH_ONNXRUNTIME

    /// The live session, plus the input/output names and shapes it was built with.
    struct Onnx::Session
    {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "libKalmanFramework"};
        Ort::SessionOptions options;
        std::unique_ptr<Ort::Session> session;
        Ort::AllocatorWithDefaultOptions allocator;

        std::string inputName;
        std::string outputName;
        std::size_t inputSize = 0;
        std::size_t outputSize = 0;

        explicit Session(const std::string &path)
        {
            // One thread each: a routing policy is a handful of small matrix products
            // run once per filter step, so a thread pool costs more in synchronisation
            // than it recovers.
            options.SetIntraOpNumThreads(1);
            options.SetInterOpNumThreads(1);
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

            session = std::make_unique<Ort::Session>(env, path.c_str(), options);

            if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
                throw std::runtime_error(
                    "Kalman::Routers::NN::Onnx: expected a single-input, single-output graph.");

            inputName = session->GetInputNameAllocated(0, allocator).get();
            outputName = session->GetOutputNameAllocated(0, allocator).get();

            const auto in = session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            const auto out = session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();

            // The batch axis is exported dynamic (-1); the feature axis must not be.
            inputSize = in.empty() ? 0 : static_cast<std::size_t>(in.back() > 0 ? in.back() : 0);
            outputSize = out.empty() ? 0 : static_cast<std::size_t>(out.back() > 0 ? out.back() : 0);
        }

        std::vector<float> Run(std::vector<float> &features)
        {
            const std::array<std::int64_t, 2> shape{1, static_cast<std::int64_t>(features.size())};
            auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input = Ort::Value::CreateTensor<float>(
                memory, features.data(), features.size(), shape.data(), shape.size());

            const char *inNames[] = {inputName.c_str()};
            const char *outNames[] = {outputName.c_str()};

            auto outputs = session->Run(Ort::RunOptions{nullptr}, inNames, &input, 1, outNames, 1);
            const float *data = outputs.front().GetTensorData<float>();
            const std::size_t n = outputs.front().GetTensorTypeAndShapeInfo().GetElementCount();
            return std::vector<float>(data, data + n);
        }
    };

    bool Onnx::HasRuntime() { return true; }

    Onnx::Onnx(const std::string &path) : fPath(path)
    {
        try
        {
            fSession = std::make_shared<Session>(path);
        }
        catch (const Ort::Exception &e)
        {
            std::ostringstream os;
            os << "Kalman::Routers::NN::Onnx: could not load '" << path << "': " << e.what();
            throw std::runtime_error(os.str());
        }
    }

    bool Onnx::IsReady() const { return fSession && fSession->session; }
    std::size_t Onnx::NInputs() const { return fSession ? fSession->inputSize : 0; }
    std::size_t Onnx::NOutputs() const { return fSession ? fSession->outputSize : 0; }

    Vector Onnx::operator()(const Vector &features) const
    {
        if (!IsReady())
            throw std::runtime_error("Kalman::Routers::NN::Onnx: no graph is loaded.");

        std::vector<float> in(features.Size());
        for (std::size_t i = 0; i < features.Size(); ++i) in[i] = static_cast<float>(features(i));

        const std::vector<float> out = fSession->Run(in);

        Vector mu(out.size());
        for (std::size_t j = 0; j < out.size(); ++j) mu(j) = static_cast<double>(out[j]);
        return mu;
    }

#else  // ---------------------------------------------------------------------

    /// Placeholder so the type stays complete without ONNX Runtime.
    struct Onnx::Session
    {
    };

    bool Onnx::HasRuntime() { return false; }

    Onnx::Onnx(const std::string &path) : fPath(path)
    {
        // Constructing this asks for a *specific* graph, so failing to provide it
        // has to be loud. The graceful degradation lives one level up, in
        // Kalman::Routers::Neural, which falls back to Bayes when it has no model.
        throw std::runtime_error(
            "Kalman::Routers::NN::Onnx: this build has no ONNX Runtime, so '" + path +
            "' cannot be executed. Rebuild with ONNX Runtime on CMAKE_PREFIX_PATH, "
            "or use Kalman::Routers::NN::MLP, which needs nothing.");
    }

    bool Onnx::IsReady() const { return false; }
    std::size_t Onnx::NInputs() const { return 0; }
    std::size_t Onnx::NOutputs() const { return 0; }

    Vector Onnx::operator()(const Vector &) const
    {
        throw std::runtime_error("Kalman::Routers::NN::Onnx: built without ONNX Runtime.");
    }

#endif
}
