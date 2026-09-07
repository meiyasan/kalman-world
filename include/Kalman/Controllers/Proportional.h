/**
 * *********************************************
 *
 * \file Proportional.h
 * \brief `u = -Kp (x_hat - x_ref)`.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLERS_PROPORTIONAL_H
#define KALMAN_CONTROLLERS_PROPORTIONAL_H

#include "Kalman/Controller.h"

namespace Kalman::Controllers
{
    /// Proportional feedback on the state error. Mirrors `controllers/proportional.py`.
    class Proportional : public Controller
    {
        protected:
            Matrix fKp;   // (l, n)

        public:
            Proportional(Model *model, const Matrix &Kp, const Vector &xRef = Vector(),
                         const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "Proportional Controller"; }
            const char *Description() const override { return "u = -Kp (x_hat - x_ref)."; }

            const Matrix &Kp() const { return fKp; }

            Vector Policy(const Vector &xHat, const Matrix &, double, const Vector *) override;
    };
}

#endif // KALMAN_CONTROLLERS_PROPORTIONAL_H
