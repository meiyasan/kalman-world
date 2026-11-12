/**
 * *********************************************
 *
 * \file Differentiator.h
 * \brief Header of the Kalman::Differentiator class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Differentiator
 * \brief Where a Jacobian comes from.
 *
 * \details The extended filter needs `df/dx` and `dh/dx`. A model declares how
 * they are obtained, as a Kalman::DifferentiatorKind:
 *
 * - Analytic: the model gives the formula (F(), H(), or its own Jacobian()).
 * - Forward: `(f(x + s e_j) - f(x)) / s`, one extra evaluation per component.
 *   First order in the step. What the Python framework has always used.
 * - Central: `(f(x + s e_j) - f(x - s e_j)) / 2s`, two evaluations per
 *   component. Second order.
 * - Automatic: exact derivatives by automatic differentiation. Reserved: it
 *   needs a tensor library, so it belongs to libKalmanLearn.
 *
 * The step is relative: `s = step * max(1, |x_j|)`.
 *
 * The role next to it: the Integrator advances a model in time, the Solver
 * handles space for a field, the Differentiator gives the derivatives.
 *
 * *********************************************
 */

#ifndef KALMAN_DIFFERENTIATOR_H
#define KALMAN_DIFFERENTIATOR_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Constants.h"
#include "Kalman/Matrix.h"

#include <functional>
#include <memory>

namespace Kalman
{
    /// A map `y = fn(x)` to differentiate.
    using Map = std::function<Vector(const Vector &)>;

    class Differentiator : public Abstract::Metadata
    {
        protected:
            double fStep;

        public:
            explicit Differentiator(double step) : fStep(step) {}
            virtual ~Differentiator() = default;

            virtual DifferentiatorKind Kind() const = 0;

            /// The relative step.
            double Step() const { return fStep; }

            /**
             * \brief `d fn / dx` at \p x, as a (rows of fn) x (size of x) matrix.
             * \param fn  the map to differentiate.
             * \param x   the point.
             * \param y0  `fn(x)` when the caller already has it (saves the forward
             *            scheme one evaluation); null to have it computed.
             */
            virtual Matrix Jacobian(const Map &fn, const Vector &x, const Vector *y0 = nullptr) const = 0;
    };

    namespace Differentiators
    {
        /// The default relative step of each numerical scheme.
        double DefaultStep(DifferentiatorKind kind);

        /// Build a numerical differentiator. Analytic and Automatic have no instance here and throw.
        std::unique_ptr<Differentiator> Make(DifferentiatorKind kind, double step = 0.);

        /// `d fn / dx` at \p x with the scheme \p kind, without building an object (step <= 0: the default).
        Matrix Jacobian(DifferentiatorKind kind, const Map &fn, const Vector &x, double step = 0.,
                        const Vector *y0 = nullptr);
    }
}

#endif // KALMAN_DIFFERENTIATOR_H
