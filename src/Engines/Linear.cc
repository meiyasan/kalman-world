/**
 * *********************************************
 *
 * \file Linear.cc
 * \brief Implementation of the Kalman::Engines::Linear class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Linear.h"

namespace Kalman::Engines
{
    void Linear::Propagate(const Model &model, const Vector &x, const Vector *u,
                                 double t, Vector &xk, Matrix &F) const
    {
        F = model.F(t);
        xk = F * x;

        if (u && model.NControl() > 0)
        {
            const Matrix B = model.B(t);
            if (B.Empty()) xk += *u;
            else           xk += B * (*u);
        }
    }

    void Linear::Measure(const Model &model, const Vector &x, double t,
                               Vector &zpred, Matrix &H) const
    {
        H = model.H(t);
        zpred = H * x;
    }
}
