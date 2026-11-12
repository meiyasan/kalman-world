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
        UseContinuousDynamics();
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

    Vector CarSpeed::Derivative(const Vector &x, const Vector &, double) const
    {
        // [s, v]' = [v, a(v)]
        Vector d(2);
        d(0) = x(1);
        d(1) = Acceleration(x(1));
        return d;
    }

    Vector CarSpeed::f(const Vector &x, const Vector &u, double t) const
    {
        // The model is continuous: the framework integrates Derivative() over one
        // sample (RK4), and the car does not roll back.
        Vector out = Kalman::Model::f(x, u, t);
        out(1) = std::max(out(1), 0.);
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
