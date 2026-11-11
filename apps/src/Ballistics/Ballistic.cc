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
    }

    Vector Ballistic::f(const Vector &x, const Vector &, double) const
    {
        const double dt = DeltaT();
        double p[3], v[3], a[3];
        for (std::size_t i = 0; i < 3; ++i)
        {
            p[i] = x(fOrder * i);
            v[i] = x(fOrder * i + 1);
        }

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

        Vector out(NState());
        for (std::size_t i = 0; i < 3; ++i)
        {
            out(fOrder * i) = (p[i] + dt * v[i]) + 0.5 * a[i] * (dt * dt);
            out(fOrder * i + 1) = v[i] + a[i] * dt;
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
