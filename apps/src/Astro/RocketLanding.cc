/**
 * *********************************************
 *
 * \file RocketLanding.cc
 * \brief Implementation of the Kalman::Apps::Astro::RocketLanding class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Astro/RocketLanding.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Astro
{
    RocketLanding::RocketLanding() : Kalman::Apps::Model("rocket-landing", 3, 1)
    {
        DeclareParameter("m_dry", 700.);
        DeclareParameter("ve", 2500.);
        DeclareParameter("thrust_max", 30000.);
        DeclareParameter("drag_area", 0.25);
        DeclareParameter("drag", 1.);
        DeclareParameter("g", 9.80665);
        DeclareParameter("decel", 12.);
        DeclareParameter("v_touch", 1.);
        DeclareParameter("kp", 1.5);
        DeclareParameter("rho0", 1.225);
        DeclareParameter("h_atm", 8500.);
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();
        DeclareState({{"altitude", "m"}, {"velocity", "m/s"}, {"mass", "kg"}});
        DeclareMeasurement({{"altitude", "m"}});
        ParametersChanged();
        SetSampleRate(10.);

        // Defaults of the Python class; a Python instance pushes its own.
        Matrix Q(3, 3, 0.);
        Q(0, 0) = 1e-3;
        Q(1, 1) = 1e-2;
        Q(2, 2) = 1e-4;
        SetQ(Q);
        SetR(Matrix(1, 1, 20. * 20.));
        Vector x0(3, 0.);
        x0(2) = 1000.;
        SetInitialState(x0);
    }

    void RocketLanding::ParametersChanged()
    {
        fDryMass = Parameter("m_dry");
        fExhaust = Parameter("ve");
        fThrustMax = Parameter("thrust_max");
        fDragArea = Parameter("drag_area");
        fDrag = Parameter("drag") != 0.;
        fG = Parameter("g");
        fDecel = Parameter("decel");
        fTouch = Parameter("v_touch");
        fGain = Parameter("kp");
        fRho0 = Parameter("rho0");
        fScaleHeight = Parameter("h_atm");
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    double RocketLanding::Density(double h) const
    {
        return fRho0 * std::exp(-std::max(h, 0.) / fScaleHeight);
    }

    double RocketLanding::Guidance(const Vector &x) const
    {
        const double h = x(0), v = x(1), m = x(2);
        if (m <= fDryMass || h <= 0.) return 0.;             // dry tanks, or landed
        const double mass = std::max(m, fDryMass);
        const double a_max = fThrustMax / mass;
        const double v_ref = -std::sqrt(2. * fDecel * std::max(h, 0.)) - fTouch;
        // The guidance always accounts for drag, whether the dynamics include it or not.
        const double a_drag = -0.5 * Density(h) * fDragArea * v * std::abs(v) / mass;
        const double a_cmd = fGain * (v_ref - v) + fG - a_drag;   // so that v' -> kp (v_ref - v)
        return std::clamp(a_cmd, 0., a_max);
    }

    Vector RocketLanding::Derivative(const Vector &x, const Vector &, double) const
    {
        const double h = x(0), v = x(1), m = x(2);
        Vector d(3, 0.);
        if (h <= 0.) return d;                                // on the pad: frozen
        const double a_T = Guidance(x);
        const double mass = std::max(m, fDryMass);
        double a = a_T - fG;
        if (fDrag) a -= 0.5 * Density(h) * fDragArea * v * std::abs(v) / mass;
        const double flow = m > fDryMass ? a_T * mass / fExhaust : 0.;
        d(0) = v;
        d(1) = a;
        d(2) = -flow;
        return d;
    }

    Vector RocketLanding::f(const Vector &x, const Vector &u, double t) const
    {
        Vector out = Kalman::Model::f(x, u, t);
        if (out(0) <= 0.)                                     // touchdown: clamp to the pad
        {
            out(0) = 0.;
            out(1) = 0.;
        }
        out(2) = std::max(out(2), fDryMass);                  // never burn past the dry mass
        return out;
    }

    Vector RocketLanding::h(const Vector &x, double) const
    {
        Vector z(1);
        z(0) = x(0);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::RocketLanding, "rocket-landing");
