/**
 * *********************************************
 *
 * \file Learn.h
 * \brief Umbrella header for libKalmanLearn
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \page libkalmanlearn libKalmanLearn — learned components
 *
 * libKalman defines what a learned policy is and needs nothing to evaluate a
 * small one:
 *
 * | In libKalman | What it is |
 * |---|---|
 * | Kalman::Routers::NN::Model | the contract: a feature vector in, one probability per mode out |
 * | Kalman::Routers::NN::MLP | a multilayer perceptron evaluated with no dependency |
 * | Kalman::Routers::Neural | the router that owns the features and holds a model |
 *
 * libKalmanLearn adds the backends that run on an AI runtime:
 *
 * | In libKalmanLearn | Runtime | Loads |
 * |---|---|---|
 * | Kalman::Routers::NN::Onnx | ONNX Runtime | a policy exported with `export_onnx()` |
 * | Kalman::Routers::NN::Torch | libTorch | a policy exported with `export_torchscript()` |
 *
 * Training stays in Python, where the optimiser and the data are. What crosses
 * to C++ is the trained policy, as one self-contained file.
 *
 * \code{.cpp}
 * #include <Kalman/Kalman.h>
 * #include <Kalman/Learn/Learn.h>
 *
 * Kalman::Routers::NN::Onnx policy("router.onnx");
 * Kalman::Routers::Neural   router(&policy, window, mean, scale);
 * Kalman::Filters::IKF      ikf({&cruise, &manoeuvre}, Pi, &router);
 * \endcode
 *
 * Link `libKalmanLearn` as well as `libKalman`. Both runtimes are optional:
 * `NN::Onnx::HasRuntime()` and `NN::Torch::HasRuntime()` say which were built
 * in, and a backend without its runtime refuses to load a file instead of
 * failing to link.
 *
 * *********************************************
 */

#ifndef KALMAN_LEARN_LEARN_H
#define KALMAN_LEARN_LEARN_H

#include "Kalman/Routers/NN/Model.h"
#include "Kalman/Routers/NN/MLP.h"
#include "Kalman/Routers/NN/Onnx.h"
#include "Kalman/Routers/NN/Torch.h"
#include "Kalman/Routers/Neural.h"

#endif // KALMAN_LEARN_LEARN_H
