/**
 * *********************************************
 *
 * \file Kalman.h
 * \brief Umbrella header for libKalman
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \page libkalman libKalman — the Kalman framework in C++
 *
 * The framework core in C++: **a standalone library with no required
 * dependency**. Not ROOT, not Eigen, not Torch — just a C++20 compiler.
 *
 * \code{.cpp}
 * #include <Kalman/Kalman.h>
 *
 * class MyModel : public Kalman::Model { ... };   // your system
 *
 * MyModel model;
 * Kalman::Filters::KF filter(&model);
 * auto record = filter.Forward(measurements);
 * printf("mean NIS = %.3f\n", filter.MeanNIS());
 * \endcode
 *
 * \section libkalman_roles The three roles
 *
 * The dependency arrows point one way only — Filter → Engine → Model:
 *
 * | Role | Question | Class |
 * |------|----------|-------|
 * | Model  | WHAT is the system? | Kalman::Model |
 * | Engine | HOW to infer?       | Kalman::Engine (Kalman::Engines::Linear, Extended, Unscented, Particle, Ensemble) |
 * | Filter | WHEN does it run?   | Kalman::Filter (Kalman::Filters::KF, EKF, UKF, PF, EnKF); Kalman::Filters::IMM, Kalman::Filters::CKF |
 *
 * \section libkalman_routing The decision layer
 *
 * Kalman::Filters::IMM runs several models at once and carries a probability on
 * each; a Kalman::Routers::Router owns that probability update and is the only
 * pluggable part of the step. Kalman::Routers::Bayesian is the classical rule,
 * Kalman::Routers::Threshold a NIS-based one, Kalman::Routers::Hybrid a blend,
 * and Kalman::Routers::Neural executes a policy trained in the Python framework
 * and exported to ONNX — the trained policy, not a language binding, being what
 * crosses between the two implementations.
 *
 * Supporting types: Kalman::Record (one pass, with its covariance band),
 * Kalman::Sampler (truth and measurements drawn from a model), Kalman::Matrix /
 * Kalman::Vector (the small dense types), and Kalman::Utils (shared algebra).
 *
 * \section libkalman_noapps No applications here
 *
 * libKalman ships **no physics**. A pendulum, a drift chamber, a suspension or a
 * power grid is an *application*: it subclasses Kalman::Model and lives with the
 * domain knowledge, not in the framework. What is here is the contract those
 * applications are written against.
 *
 * \section libkalman_optional Optional backends
 *
 * Everything works with none of these; they only make some things faster or
 * richer. See `Kalman/Config.h`.
 *
 * - **BLAS** (`KALMAN_WITH_BLAS`) — matrix products at or above
 *   `KALMAN_BLAS_MIN_DIM` go through `cblas_dgemm`. Below it the hand-rolled
 *   kernel is measurably faster, so the choice is per operation.
 * - **libTorch** (`KALMAN_WITH_TORCH`) — for tensor-shaped work: ensembles,
 *   field states, learned components. Deliberately *not* used for
 *   Kalman::Matrix, where its per-op dispatch would dominate.
 * - **ONNX Runtime** (`KALMAN_WITH_ONNXRUNTIME`) — for executing decision and
 *   routing policies trained in the Python framework and exported to ONNX.
 *
 * *********************************************
 */

#ifndef KALMAN_KALMAN_H
#define KALMAN_KALMAN_H

#include "Kalman/Config.h"

// Core contracts — mirrors modules/framework/*.py
#include "Kalman/Constants.h"
#include "Kalman/Matrix.h"
#include "Kalman/Utils.h"
#include "Kalman/Grid.h"
#include "Kalman/FFT.h"
#include "Kalman/Integrator.h"
#include "Kalman/Solver.h"
#include "Kalman/Model.h"
#include "Kalman/Engine.h"
#include "Kalman/Filter.h"
#include "Kalman/Sampler.h"
#include "Kalman/Diagnostics.h"
#include "Kalman/Factory.h"

// The C surface — what a non-C++ consumer (the Python package, a ROOT macro
// through cling, another language) sees of the two headers above.
#include "Kalman/CAPI.h"

// Concretes — mirrors the matching subpackages
#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Integrators/Euler.h"
#include "Kalman/Integrators/RK4.h"
#include "Kalman/Integrators/DormandPrince.h"
#include "Kalman/Solvers/Spectral.h"
#include "Kalman/Solvers/FiniteDifference.h"
#include "Kalman/Engines/Linear.h"
#include "Kalman/Engines/Extended.h"
#include "Kalman/Engines/Unscented.h"
#include "Kalman/Engines/Particle.h"
#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Tracking.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Models/Linear.h"
#include "Kalman/Models/Extended.h"
#include "Kalman/Filters/KF.h"
#include "Kalman/Filters/EKF.h"
#include "Kalman/Filters/UKF.h"
#include "Kalman/Filters/PF.h"
#include "Kalman/Filters/EnKF.h"
#include "Kalman/Filters/CKF.h"
#include "Kalman/Filters/IMM.h"
#include "Kalman/IO/Record.h"
#include "Kalman/IO/Collection.h"
#include "Kalman/Routers/Base.h"
#include "Kalman/Routers/Bayesian.h"
#include "Kalman/Routers/Threshold.h"
#include "Kalman/Routers/Hybrid.h"
#include "Kalman/Routers/NN/Model.h"
#include "Kalman/Routers/NN/MLP.h"
#include "Kalman/Routers/NN/Onnx.h"
#include "Kalman/Routers/NN/Torch.h"
#include "Kalman/Routers/Neural.h"

#endif // KALMAN_KALMAN_H
