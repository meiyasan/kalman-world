/**
 * *********************************************
 *
 * \file Central.h
 * \brief Header of the Kalman::Differentiators::Central class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_DIFFERENTIATORS_CENTRAL_H
#define KALMAN_DIFFERENTIATORS_CENTRAL_H

#include "Kalman/Differentiator.h"

namespace Kalman::Differentiators
{
    /// Two-sided differences: (f(x + s e_j) - f(x - s e_j)) / 2s. Second order, two evaluations per component.
    class Central : public Differentiator
    {
        public:
            explicit Central(double step = 0.)
                : Differentiator(step > 0. ? step : DefaultStep(DifferentiatorKind::Central)) {}

            DifferentiatorKind Kind() const override { return DifferentiatorKind::Central; }
            const char *Title() const override { return "Central Differentiator"; }
            const char *Description() const override
            { return "Two-sided differences: (f(x + s e_j) - f(x - s e_j)) / 2s. Second order, two evaluations per component."; }

            Matrix Jacobian(const Map &fn, const Vector &x, const Vector *y0 = nullptr) const override;
    };
}

#endif // KALMAN_DIFFERENTIATORS_CENTRAL_H
