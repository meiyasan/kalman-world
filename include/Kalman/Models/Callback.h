/**
 * *********************************************
 *
 * \file Callback.h
 * \brief Header of the Kalman::Models::Callback class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Callback
 * \brief A model whose maps are functions supplied by the caller.
 *
 * \details This is how a model written in another language runs on the C++
 * engines: its `f` and `h`, and optionally their Jacobians, are plain C
 * function pointers, each called with an opaque `user` pointer. The Python
 * package builds one from a Python model's own methods, so a model defined in
 * a notebook is filtered by the same Kalman::Filter as a model written in C++.
 *
 * A function returns 0 on success. Any other value stops the pass with a
 * `std::runtime_error`, so an exception raised on the other side of the
 * boundary is reported, not swallowed.
 *
 * When no Jacobian function is given the model differences `f` or `h` itself
 * (central differences, Model::Epsilon()).
 *
 * The cost is one call across the language boundary per evaluation. A model
 * written in C++ pays none; see docs/15 for the measurements.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_CALLBACK_H
#define KALMAN_MODELS_CALLBACK_H

#include "Kalman/Models/Extended.h"

namespace Kalman::Models
{
    class Callback : public Extended
    {
        public:
            /// `x_next = f(x, u, t)`; \p u is null when the model has no control.
            using StepFn = int (*)(void *user, const double *x, const double *u, double t, double *x_next);
            /// `z = h(x, t)`.
            using MeasureFn = int (*)(void *user, const double *x, double t, double *z);
            /// `J = df/dx` at (x, u, t), row-major (n x n).
            using StepJacobianFn = int (*)(void *user, const double *x, const double *u, double t, double *J);
            /// `J = dh/dx` at (x, t), row-major (m x n).
            using MeasureJacobianFn = int (*)(void *user, const double *x, double t, double *J);

        private:
            StepFn fStep;
            MeasureFn fMeasure;
            StepJacobianFn fStepJacobian;
            MeasureJacobianFn fMeasureJacobian;
            void *fUser;

        public:
            Callback(std::string name, std::size_t n, std::size_t m, std::size_t l,
                     StepFn step, MeasureFn measure,
                     StepJacobianFn stepJacobian = nullptr, MeasureJacobianFn measureJacobian = nullptr,
                     void *user = nullptr);

            const char *Title() const override { return "Callback Model"; }
            const char *Description() const override
            { return "Model whose f/h are functions supplied by the caller (another language, typically)."; }

            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;
            Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const override;
            Matrix JacobianH(const Vector &x, double t = 0.) const override;
    };
}

#endif // KALMAN_MODELS_CALLBACK_H
