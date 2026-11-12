/**
 * \file 15_learn.cc
 * \brief libKalmanLearn: the ONNX and TorchScript backends, with or without their runtime.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Learn/Learn.h"
#include "Check.h"

#include <cstdio>
#include <stdexcept>

using namespace Kalman;
using namespace Kalman::Routers;

int main()
{
    // Which runtimes this build carries. The classes exist and link either way:
    // an absent runtime is an answer at run time, not a link error.
    std::printf("  [backends: ONNX=%s  Torch=%s]\n",
                NN::Onnx::HasRuntime() ? "yes" : "no",
                NN::Torch::HasRuntime() ? "yes" : "no");

    // --- the optional backends behave the same way whether present or not -----
    {
        // Asking for a specific artifact that cannot be executed is always loud,
        // whether the file is missing or the runtime is.
        bool threw = false;
        try { NN::Onnx missing("definitely-not-here.onnx"); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "NN::Onnx: a missing graph or absent runtime throws");

        threw = false;
        try { NN::Torch missing("definitely-not-here.pt"); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "NN::Torch: a missing module or absent runtime throws");

        NN::Onnx idleOnnx;
        NN::Torch idleTorch;
        CHECK(!idleOnnx.IsReady(), "NN::Onnx: default-constructed is not ready");
        CHECK(!idleTorch.IsReady(), "NN::Torch: default-constructed is not ready");

        // ... and an unready model leaves the router on its Bayesian fallback.
        Neural router(&idleOnnx, 5);
        CHECK(!router.IsReady(), "Neural: an unready model means an unready router");
        const Matrix nis(3, 2, 1.);
        Vector c{0.5, 0.5}, L{3.0, 1.0};
        CHECK_CLOSE(router.Route(nis, c, L)(0), 0.75, 1e-12,
                    "Neural: unready model still routes, via Bayes");
    }

    return Check::Summary("15_learn");
}
