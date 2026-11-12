/**
 * *********************************************
 *
 * \file Model.h
 * \brief Header of the Kalman::Apps::Model class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Model
 * \brief Base of the application models in libKalmanApps.
 *
 * \details libKalman ships the contracts and no physics; libKalmanApps
 * is where the physics of `modules/apps/` lives in C++. Each model here is the
 * twin of one Python class: the Python class keeps its name, its constructor
 * and everything a notebook does with it, and hands the filtering pass to its
 * twin when the library is present.
 *
 * For the two to be interchangeable they must agree, so this base differences
 * `f` and `h` the way the Python framework does (`helpers.numerical_jacobian`:
 * forward differences, step `1e-6 · max(1, |x_j|)`) instead of the library's
 * central scheme. A model with an analytic Jacobian overrides Jacobian() itself.
 *
 * A twin is configured through the named parameters of Kalman::Model, and its
 * noise covariances, initial state and sample rate are pushed from the Python
 * instance, so nothing is defined twice.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_MODEL_H
#define KALMAN_APPS_MODEL_H

#include <Kalman/Models/Extended.h>

namespace Kalman::Apps
{
    class Model : public Kalman::Models::Extended
    {
        public:
            /// Relative step of the forward difference (helpers.numerical_jacobian's eps).
            static constexpr double kJacobianStep = 1e-6;

            Model(std::string name, std::size_t n, std::size_t m, std::size_t l = 0)
                : Kalman::Models::Extended(std::move(name), n, m, l)
            {
                // Forward differences, as the Python framework takes them, so a twin
                // and its Python class linearise alike.
                SetDifferentiator(DifferentiatorKind::Forward, kJacobianStep);
            }

            const char *Title() const override { return "Application Model"; }
            const char *Description() const override
            { return "Nonlinear application model; Jacobians differenced as the Python framework does."; }
    };
}

#endif // KALMAN_APPS_MODEL_H
