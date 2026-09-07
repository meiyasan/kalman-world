/**
 * *********************************************
 *
 * \file PID.h
 * \brief `u = -Kp e - Ki int e dt - Kd de/dt` on the state error.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLERS_PID_H
#define KALMAN_CONTROLLERS_PID_H

#include "Kalman/Controller.h"

namespace Kalman::Controllers
{
    /// PID on `e = x_hat - x_ref`; an empty Ki or Kd is zero (P, PD, PI ...). Mirrors `controllers/pid.py`.
    class PID : public Controller
    {
        protected:
            Matrix fKp, fKi, fKd;   // (l, n) each
            Vector fIntegral;       // (n)
            Vector fPrevError;      // (n)
            double fDt = 1.;

        public:
            PID(Model *model, const Matrix &Kp, const Matrix &Ki = Matrix(), const Matrix &Kd = Matrix(),
                const Vector &xRef = Vector(), const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "PID Controller"; }
            const char *Description() const override { return "u = -Kp e - Ki int e - Kd de/dt on the state error."; }

            Vector Policy(const Vector &xHat, const Matrix &, double, const Vector *) override;
            void Reset() override;
    };
}

#endif // KALMAN_CONTROLLERS_PID_H
