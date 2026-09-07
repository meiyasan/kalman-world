/**
 * *********************************************
 *
 * \file RK4.h
 * \brief The classic four-stage Runge–Kutta scheme — fourth order.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_INTEGRATORS_RK4_H
#define KALMAN_INTEGRATORS_RK4_H

#include "Kalman/Integrator.h"

namespace Kalman::Integrators
{
    /// The one home for the RK4 arithmetic; the control is held across the step. Mirrors `integrators/rk4.py`.
    class RK4 : public Integrator
    {
        public:
            IntegratorKind Kind() const override { return IntegratorKind::RK4; }
            const char *Title() const override { return "RK4 Integrator"; }
            const char *Description() const override
            { return "Classic 4-stage Runge-Kutta (fourth order); ZOH on the control."; }

            Tableau GetTableau() const override
            {
                return Tableau{{0., 0.5, 0.5, 1.},
                               {{}, {0.5}, {0., 0.5}, {0., 0., 1.}},
                               {1. / 6., 1. / 3., 1. / 3., 1. / 6.}};
            }

            Vector Step(const Derivative &f, const Vector &x, const Vector &u,
                        double t, double dt) const override
            {
                return RungeKuttaStep(GetTableau(),
                                      [&](const Vector &xx, double tt) { return f(xx, u, tt); },
                                      x, t, dt);
            }
    };
}

#endif // KALMAN_INTEGRATORS_RK4_H
