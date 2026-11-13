/**
 * *********************************************
 *
 * \file Tornado.cc
 * \brief Implementation of the Kalman::Apps::Weather::Tornado class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Weather/Tornado.h"

#include <Kalman/Factory.h>

namespace Kalman::Apps::Weather
{
    Tornado::Tornado() : Kalman::Apps::Model("tornado-environment", 1, 1)
    {
        DeclareState({{"stp", ""}});          // significant-tornado-parameter proxy, dimensionless
        DeclareMeasurement({{"stp", ""}});
        DeclareParameter("mu", 0.3);
        DeclareParameter("persistence", 0.85);
        ParametersChanged();
        SetSampleRate(1.);

        // Defaults of the Python class; a Python instance pushes its own.
        Matrix Q(1, 1, 0.15 * 0.15), R(1, 1, 0.4 * 0.4);
        SetQ(Q);
        SetR(R);

        Vector x0(1);
        x0(0) = fMu;
        SetInitialState(x0);
    }

    void Tornado::ParametersChanged()
    {
        fMu = Parameter("mu");
        fPersistence = Parameter("persistence");
    }

    Vector Tornado::f(const Vector &x, const Vector &, double) const
    {
        Vector out(1);
        out(0) = fMu + fPersistence * (x(0) - fMu);
        return out;
    }

    Vector Tornado::h(const Vector &x, double) const
    {
        Vector z(1);
        z(0) = x(0);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Weather::Tornado, "tornado-environment");
