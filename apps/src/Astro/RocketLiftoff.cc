/**
 * *********************************************
 *
 * \file RocketLiftoff.cc
 * \brief Implementation of the Kalman::Apps::Astro::RocketLiftoff class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Astro/RocketLiftoff.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Astro
{
    RocketLiftoff::RocketLiftoff() : Kalman::Apps::Model("rocket-liftoff", 3, 1)
    {
        DeclareParameter("m_dry", 400.);
        DeclareParameter("ve", 2500.);
        DeclareParameter("mdot", 8.);
        DeclareParameter("drag_area", 0.25);
        DeclareParameter("drag", 1.);
        DeclareParameter("g", 9.80665);
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
        x0(2) = 1000.;                          // on the pad, fuelled
        SetInitialState(x0);
    }

    void RocketLiftoff::ParametersChanged()
    {
        fDryMass = Parameter("m_dry");
        fExhaust = Parameter("ve");
        fFlow = Parameter("mdot");
        fDragArea = Parameter("drag_area");
        fDrag = Parameter("drag") != 0.;
        fG = Parameter("g");
        fRho0 = Parameter("rho0");
        fScaleHeight = Parameter("h_atm");
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    Vector RocketLiftoff::Derivative(const Vector &x, const Vector &, double) const
    {
        const double h = x(0), v = x(1), m = x(2);
        const double flow = m > fDryMass ? fFlow : 0.;    // the engine cuts off at burnout
        const double thrust = fExhaust * flow;
        const double mass = std::max(m, fDryMass);         // keeps the division safe and physical
        double a = thrust / mass - fG;
        if (fDrag)
        {
            const double rho = fRho0 * std::exp(-std::max(h, 0.) / fScaleHeight);
            a -= 0.5 * rho * fDragArea * v * std::abs(v) / mass;
        }
        Vector d(3);
        d(0) = v;
        d(1) = a;
        d(2) = -flow;
        return d;
    }

    Vector RocketLiftoff::f(const Vector &x, const Vector &u, double t) const
    {
        // The framework integrates Derivative() over one sample; the mass never
        // drops below the dry mass.
        Vector out = Kalman::Model::f(x, u, t);
        out(2) = std::max(out(2), fDryMass);
        return out;
    }

    Vector RocketLiftoff::h(const Vector &x, double) const
    {
        Vector z(1);
        z(0) = x(0);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::RocketLiftoff, "rocket-liftoff");
