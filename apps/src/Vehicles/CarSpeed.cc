/**
 * *********************************************
 *
 * \file CarSpeed.cc
 * \brief Implementation of the Kalman::Apps::Vehicles::CarSpeed class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Vehicles/CarSpeed.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Vehicles
{
    CarSpeed::CarSpeed() : Kalman::Apps::Model("car-speed", 2, 1)
    {
        DeclareParameter("mass", 1500.);
        DeclareParameter("power", 150000.);
        DeclareParameter("mu", 1.);
        DeclareParameter("drag_area", 0.66);
        DeclareParameter("rho", 1.225);
        DeclareParameter("c_roll", 0.015);
        DeclareParameter("grade", 0.);
        DeclareParameter("g", 9.81);
        DeclareParameter("v_eps", 0.5);
        ParametersChanged();
        SetSampleRate(20.);

        // Defaults of the Python class; a Python instance pushes its own.
        Matrix Q(2, 2, 0.);
        Q(0, 0) = 1e-3;
        Q(1, 1) = 5e-2;
        SetQ(Q);
        SetR(Matrix(1, 1, 0.25));
    }

    void CarSpeed::ParametersChanged()
    {
        fMass = Parameter("mass");
        fPower = Parameter("power");
        fMu = Parameter("mu");
        fDragArea = Parameter("drag_area");
        fRho = Parameter("rho");
        fRoll = Parameter("c_roll");
        fGrade = Parameter("grade");
        fG = Parameter("g");
        fSpeedFloor = Parameter("v_eps");
    }

    double CarSpeed::Acceleration(double v) const
    {
        const double grip = fMu * fMass * fG;
        const double power = fPower / std::max(std::abs(v), fSpeedFloor);
        const double drive = std::min(grip, power);       // grip-limited, then power-limited
        const double drag = 0.5 * fRho * fDragArea * v * std::abs(v);
        const double roll = fRoll * fMass * fG;
        const double grade = fMass * fG * std::sin(fGrade);
        return (drive - drag - roll - grade) / fMass;
    }

    Vector CarSpeed::f(const Vector &x, const Vector &, double) const
    {
        // One RK4 step of [s, v]' = [v, a(v)], written out in the order the
        // Python integrator sums it.
        const double dt = DeltaT();
        const double s = x(0), v = x(1);

        const double k1s = v, k1v = Acceleration(v);
        const double v2 = v + 0.5 * dt * k1v;
        const double k2s = v2, k2v = Acceleration(v2);
        const double v3 = v + 0.5 * dt * k2v;
        const double k3s = v3, k3v = Acceleration(v3);
        const double v4 = v + dt * k3v;
        const double k4s = v4, k4v = Acceleration(v4);

        Vector out(2);
        out(0) = s + dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s);
        out(1) = std::max(v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v), 0.);   // no rolling back
        return out;
    }

    Vector CarSpeed::h(const Vector &x, double) const
    {
        Vector z(1);
        z(0) = x(0);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Vehicles::CarSpeed, "car-speed");
