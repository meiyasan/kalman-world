/**
 * *********************************************
 *
 * \file CarLane.cc
 * \brief Implementation of the Kalman::Apps::Vehicles::CarLane class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Vehicles/CarLane.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Vehicles
{
    CarLane::CarLane() : Kalman::Apps::Model("car-lane", 3, 1, 1)
    {
        DeclareParameter("wheelbase", 2.7);
        DeclareParameter("steer_limit", 0.1);
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 0.);    // IntegratorKind: 0 Euler (the Python default), 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();
        DeclareState({{"lateral", "m"}, {"heading", "rad"}, {"speed", "m/s"}});
        DeclareMeasurement({{"lateral", "m"}});
        ParametersChanged();
        SetSampleRate(50.);

        // Defaults of the Python class; a Python instance pushes its own.
        Matrix Q(3, 3, 0.);
        Q(0, 0) = 1e-4;
        Q(1, 1) = 1e-6;
        Q(2, 2) = 1e-4;
        SetQ(Q);
        SetR(Matrix(1, 1, 0.03 * 0.03));
        Vector x0(3, 0.);
        x0(2) = 15.;
        SetInitialState(x0);
    }

    void CarLane::ParametersChanged()
    {
        fWheelbase = Parameter("wheelbase");
        fSteerLimit = Parameter("steer_limit");
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    double CarLane::Steering(const Vector &u) const
    {
        if (u.Size() < 1) return 0.;
        return std::clamp(u(0), -fSteerLimit, fSteerLimit);
    }

    Vector CarLane::Derivative(const Vector &x, const Vector &u, double) const
    {
        const double psi = x(1), v = x(2);
        Vector d(3, 0.);
        d(0) = v * std::sin(psi);
        d(1) = v * std::tan(Steering(u)) / fWheelbase;
        return d;
    }

    Vector CarLane::h(const Vector &x, double) const
    {
        Vector z(1);
        z(0) = x(0);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Vehicles::CarLane, "car-lane");
