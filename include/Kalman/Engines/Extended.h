/**
 * *********************************************
 *
 * \file Extended.h
 * \brief Header of the Kalman::Engines::Extended class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Extended
 * \brief Jacobian-linearised Kalman step for nonlinear models.
 *
 * \details Propagates the mean through the true map `f(x, u, t)` and the
 * covariance through its Jacobian `F = df/dx`, taken from Kalman::Model::Jacobian()
 * — analytic when the model provides one, central differences otherwise. Likewise
 * `h` and `H = dh/dx` on the measurement side.
 *
 * On a linear model this reduces to the linear computation (the Jacobian of `F x`
 * is `F`), so it is always a safe default when in doubt.
 *
 * *********************************************
 */

#ifndef KALMAN_EXTENDEDENGINE_H
#define KALMAN_EXTENDEDENGINE_H

#include "Kalman/Engine.h"

namespace Kalman::Engines
{
    class Extended : public Engine
    {
        public:
            Extended() = default;
            ~Extended() override = default;

            EngineKind Kind() const override { return EngineKind::Extended; }
            const char *Title() const override { return "Extended Kalman Engine"; }
            const char *Description() const override
            { return "Jacobian-linearised Kalman step for nonlinear models."; }

            void Propagate(const Model &model, const Vector &x, const Vector *u,
                           double t, Vector &xk, Matrix &F) const override;

            void Measure(const Model &model, const Vector &x, double t,
                         Vector &zpred, Matrix &H) const override;
    };
}

#endif // KALMAN_EXTENDEDENGINE_H
