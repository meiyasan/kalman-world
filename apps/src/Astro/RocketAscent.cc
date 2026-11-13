/**
 * *********************************************
 *
 * \file RocketAscent.cc
 * \brief Implementation of the Kalman::Apps::Astro::RocketAscent class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Astro/RocketAscent.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Astro
{
    RocketAscent::RocketAscent() : Kalman::Apps::Model("rocket-ascent", 4, 2, 2)
    {
        DeclareParameter("GM", 398.6004);
        DeclareParameter("R", 6.3781);
        DeclareParameter("H", 0.0085);
        DeclareParameter("ballistic_coeff", 2.0e-3);
        DeclareParameter("drag", 0.);
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();
        DeclareState({{"x", "Mm"}, {"y", "Mm"}, {"vx", "km/s"}, {"vy", "km/s"}});
        DeclareMeasurement({{"x", "Mm"}, {"y", "Mm"}});
        ParametersChanged();
        SetSampleRate(1000.);                   // one sample per second, in samples per ks

        // Defaults of the Python class; a Python instance pushes its own.
        Matrix Q(4, 4, 0.);
        for (std::size_t i = 0; i < 4; ++i) Q(i, i) = 1e-9;
        SetQ(Q);
        Matrix R(2, 2, 0.);
        R(0, 0) = R(1, 1) = 1e-6;               // 1 km, in Mm, squared
        SetR(R);
        Vector x0(4, 0.);
        x0(0) = 6.3781;                         // on the surface
        SetInitialState(x0);
    }

    void RocketAscent::ParametersChanged()
    {
        fGM = Parameter("GM");
        fRadius = Parameter("R");
        fScaleHeight = Parameter("H");
        fBallistic = Parameter("ballistic_coeff");
        fDrag = Parameter("drag") != 0.;
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    Vector RocketAscent::Derivative(const Vector &s, const Vector &u, double) const
    {
        const double rx = s(0), ry = s(1), vx = s(2), vy = s(3);
        const double r = std::sqrt(rx * rx + ry * ry);
        const double r3 = std::pow(r, 3.);
        double ax = -fGM * rx / r3, ay = -fGM * ry / r3;      // central gravity
        if (u.Size() >= 2) { ax += u(0); ay += u(1); }        // thrust, the control
        if (fDrag)
        {
            const double rho = std::exp(-std::max(r - fRadius, 0.) / fScaleHeight);
            const double speed = std::sqrt(vx * vx + vy * vy);
            ax -= fBallistic * rho * speed * vx;              // quadratic drag
            ay -= fBallistic * rho * speed * vy;
        }
        Vector d(4);
        d(0) = vx;
        d(1) = vy;
        d(2) = ax;
        d(3) = ay;
        return d;
    }

    Vector RocketAscent::h(const Vector &x, double) const
    {
        Vector z(2);
        z(0) = x(0);
        z(1) = x(1);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::RocketAscent, "rocket-ascent");
