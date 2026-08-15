/**
 * *********************************************
 *
 * \file Extended.cc
 * \brief Implementation of the Kalman::Engines::Extended class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Extended.h"

namespace Kalman::Engines
{
    void Extended::Propagate(const Model &model, const Vector &x, const Vector *u,
                                   double t, Vector &xk, Matrix &F) const
    {
        // The control vector is passed to both the map and the Jacobian, so an
        // uncontrolled model still gets a well-formed (empty) vector rather than a
        // dangling pointer.
        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;

        // Mean through the true map, covariance through its linearisation.
        xk = model.f(x, uu, t);
        F = model.Jacobian(x, uu, t);
    }

    void Extended::Measure(const Model &model, const Vector &x, double t,
                                 Vector &zpred, Matrix &H) const
    {
        zpred = model.h(x, t);
        H = model.JacobianH(x, t);
    }
}
