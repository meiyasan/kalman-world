/**
 * *********************************************
 *
 * \file Euler.h
 * \brief Explicit forward Euler — first order, one derivative per step.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_INTEGRATORS_EULER_H
#define KALMAN_INTEGRATORS_EULER_H

#include "Kalman/Integrator.h"

namespace Kalman::Integrators
{
    /// `x(t + dt) = x + dt f(x, u, t)`. Mirrors `integrators/euler.py`.
    class Euler : public Integrator
    {
        public:
            IntegratorKind Kind() const override { return IntegratorKind::Euler; }
            const char *Title() const override { return "Euler Integrator"; }
            const char *Description() const override
            { return "Explicit forward Euler: x + dt f(x, u, t) (first order)."; }

            Tableau GetTableau() const override { return Tableau{{0.}, {{}}, {1.}}; }

            Vector Step(const Derivative &f, const Vector &x, const Vector &u,
                        double t, double dt) const override
            {
                Vector out = x;
                out += dt * f(x, u, t);
                return out;
            }
    };
}

#endif // KALMAN_INTEGRATORS_EULER_H
