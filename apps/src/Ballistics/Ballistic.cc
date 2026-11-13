/**
 * *********************************************
 *
 * \file Ballistic.cc
 * \brief Implementation of the Kalman::Apps::Ballistics::Ballistic class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Ballistics/Ballistic.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Apps::Ballistics
{
    Ballistic::Ballistic(std::size_t order)
        : Kalman::Apps::Model("ballistic", 3 * order, 3), fOrder(order)
    {
        if (order != 2 && order != 3)
            throw std::invalid_argument("Kalman::Apps::Ballistics::Ballistic: order must be 2 or 3.");

        DeclareParameter("c_drag", 0.);
        DeclareParameter("m", 1.);
        DeclareParameter("g", 9.81);
        DeclareParameter("surface_gravity", 1.);
        DeclareParameter("G", 6.674e-11);
        DeclareParameter("M", 5.972e24);
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();

        std::vector<Component> state, measurement;
        for (const char *axis : {"x", "y", "z"})
        {
            state.push_back({axis, "m"});
            state.push_back({std::string("v") + axis, "m/s"});
            if (order >= 3) state.push_back({std::string("a") + axis, "m/s^2"});
            measurement.push_back({axis, "m"});
        }
        DeclareState(state);
        DeclareMeasurement(measurement);
        ParametersChanged();
        SetSampleRate(128.);

        Matrix R(3, 3, 0.);
        for (std::size_t i = 0; i < 3; ++i) R(i, i) = 0.25;
        SetR(R);
    }

    void Ballistic::ParametersChanged()
    {
        fDragOverMass = Parameter("c_drag") / Parameter("m");
        fG = Parameter("g");
        fSurfaceGravity = Parameter("surface_gravity") != 0.;
        fGravConstant = Parameter("G");
        fCentralMass = Parameter("M");
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    void Ballistic::Acceleration(const double *p, const double *v, double *a) const
    {
        // Gravity: constant along z near the surface, G M / r^3 about the centre in orbit.
        double grav[3] = {0., 0., 0.};
        if (fSurfaceGravity)
            grav[2] = -fG;
        else
        {
            const double r = std::sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2]);
            if (r > 0.)
                for (std::size_t i = 0; i < 3; ++i)
                    grav[i] = -fGravConstant * fCentralMass * p[i] / std::pow(r, 3.);
        }

        // Quadratic drag, opposed to the velocity.
        const double speed = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        for (std::size_t i = 0; i < 3; ++i)
            a[i] = grav[i] + (speed > 0. ? -fDragOverMass * speed * v[i] : 0.);
    }

    Vector Ballistic::Derivative(const Vector &s, const Vector &, double) const
    {
        // dp/dt = v, dv/dt = a(p, v), da/dt = 0.
        const std::size_t order = fOrder;
        const double p[3] = {s(0), s(order), s(2 * order)};
        const double v[3] = {s(1), s(order + 1), s(2 * order + 1)};
        double a[3];
        Acceleration(p, v, a);
        Vector d(NState(), 0.);
        for (std::size_t i = 0; i < 3; ++i) { d(order * i) = v[i]; d(order * i + 1) = a[i]; }
        return d;
    }

    Vector Ballistic::f(const Vector &x, const Vector &u, double t) const
    {
        // The model is continuous: the framework integrates Derivative() over one
        // sample; the acceleration slot is then filled in where the step lands.
        const std::size_t order = fOrder;
        const Vector y = Kalman::Model::f(x, u, t);

        const double p[3] = {y(0), y(order), y(2 * order)};
        const double v[3] = {y(1), y(order + 1), y(2 * order + 1)};
        double a[3];
        Acceleration(p, v, a);

        Vector out(NState());
        for (std::size_t i = 0; i < 3; ++i)
        {
            out(fOrder * i) = p[i];
            out(fOrder * i + 1) = v[i];
            if (fOrder >= 3) out(fOrder * i + 2) = a[i];
        }
        return out;
    }

    Vector Ballistic::h(const Vector &x, double) const
    {
        Vector z(3);
        for (std::size_t i = 0; i < 3; ++i) z(i) = x(fOrder * i);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Ballistics::Ballistic, "ballistic");
KALMAN_REGISTER_MODEL(Kalman::Apps::Ballistics::Ballistic, "ballistic-order2", std::size_t(2));
