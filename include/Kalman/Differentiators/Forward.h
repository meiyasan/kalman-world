/**
 * *********************************************
 *
 * \file Forward.h
 * \brief Header of the Kalman::Differentiators::Forward class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_DIFFERENTIATORS_FORWARD_H
#define KALMAN_DIFFERENTIATORS_FORWARD_H

#include "Kalman/Differentiator.h"

namespace Kalman::Differentiators
{
    /// One-sided differences: (f(x + s e_j) - f(x)) / s. First order, one evaluation per component.
    class Forward : public Differentiator
    {
        public:
            explicit Forward(double step = 0.)
                : Differentiator(step > 0. ? step : DefaultStep(DifferentiatorKind::Forward)) {}

            DifferentiatorKind Kind() const override { return DifferentiatorKind::Forward; }
            const char *Title() const override { return "Forward Differentiator"; }
            const char *Description() const override
            { return "One-sided differences: (f(x + s e_j) - f(x)) / s. First order, one evaluation per component."; }

            Matrix Jacobian(const Map &fn, const Vector &x, const Vector *y0 = nullptr) const override;
    };
}

#endif // KALMAN_DIFFERENTIATORS_FORWARD_H
