/**
 * *********************************************
 *
 * \file Linear.h
 * \brief Header of the Kalman::Engines::Linear class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Linear
 * \brief Standard linear Kalman step, using the model's matrices directly.
 *
 * \details `x_prior = F x + B u` and `z_pred = H x`, read straight off
 * Kalman::Model::F() / B() / H(). On a nonlinear model this would silently ignore
 * the nonlinearity, so use Kalman::Engines::Extended there.
 *
 * *********************************************
 */

#ifndef KALMAN_LINEARENGINE_H
#define KALMAN_LINEARENGINE_H

#include "Kalman/Engine.h"

namespace Kalman::Engines
{
    class Linear : public Engine
    {
        public:
            Linear() = default;
            ~Linear() override = default;

            EngineKind Kind() const override { return EngineKind::Linear; }
            const char *Title() const override { return "Linear Kalman Engine"; }
            const char *Description() const override
            { return "Standard linear Kalman step (constant F/H matrices)."; }

            void Propagate(const Model &model, const Vector &x, const Vector *u,
                           double t, Vector &xk, Matrix &F) const override;

            void Measure(const Model &model, const Vector &x, double t,
                         Vector &zpred, Matrix &H) const override;
    };
}

#endif // KALMAN_LINEARENGINE_H
