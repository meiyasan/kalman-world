/**
 * *********************************************
 *
 * \file RocketDescent.cc
 * \brief Implementation of the Kalman::Apps::Astro::RocketDescent class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Astro/RocketDescent.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Astro
{
    RocketDescent::RocketDescent(bool control)
        : Kalman::Apps::Model(control ? "rocket-descent-control" : "rocket-descent", 7, 3, control ? 2 : 0)
    {
        DeclareParameter("m_dry", 700.);
        DeclareParameter("ve", 2800.);
        DeclareParameter("thrust_max", 40000.);
        DeclareParameter("length", 40.);
        DeclareParameter("lever", 20.);
        DeclareParameter("drag_area", 2.);
        DeclareParameter("drag", 1.);
        DeclareParameter("g", 9.80665);
        DeclareParameter("decel", 8.);
        DeclareParameter("v_cruise", 55.);
        DeclareParameter("v_touch", 1.5);
        DeclareParameter("theta_max", 0.35);
        DeclareParameter("gimbal_max", 0.20);
        DeclareParameter("kp_v", 0.7);
        DeclareParameter("kx", 6.0e-3);
        DeclareParameter("kvx", 4.5e-2);
        DeclareParameter("kth", 8.);
        DeclareParameter("kom", 4.);
        DeclareParameter("rho0", 1.225);
        DeclareParameter("h_atm", 8500.);
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();
        DeclareState({{"x", "m"}, {"y", "m"}, {"vx", "m/s"}, {"vy", "m/s"},
                      {"theta", "rad"}, {"omega", "rad/s"}, {"mass", "kg"}});
        DeclareMeasurement({{"x", "m"}, {"y", "m"}, {"theta", "rad"}});
        ParametersChanged();
        SetSampleRate(20.);

        // Defaults of the Python class; a Python instance pushes its own.
        const double q[7] = {1e-3, 1e-3, 1e-2, 1e-2, 1e-5, 1e-4, 1e-3};
        Matrix Q(7, 7, 0.);
        for (std::size_t i = 0; i < 7; ++i) Q(i, i) = q[i];
        SetQ(Q);
        Matrix R(3, 3, 0.);
        R(0, 0) = R(1, 1) = 15. * 15.;
        R(2, 2) = 0.01 * 0.01;
        SetR(R);
        Vector x0(7, 0.);
        x0(6) = 1000.;
        SetInitialState(x0);
    }

    void RocketDescent::ParametersChanged()
    {
        fDryMass = Parameter("m_dry");
        fExhaust = Parameter("ve");
        fThrustMax = Parameter("thrust_max");
        fLength = Parameter("length");
        fLever = Parameter("lever");
        fDragArea = Parameter("drag_area");
        fDrag = Parameter("drag") != 0.;
        fG = Parameter("g");
        fDecel = Parameter("decel");
        fCruise = Parameter("v_cruise");
        fTouch = Parameter("v_touch");
        fThetaMax = Parameter("theta_max");
        fGimbalMax = Parameter("gimbal_max");
        fKv = Parameter("kp_v");
        fKx = Parameter("kx");
        fKvx = Parameter("kvx");
        fKth = Parameter("kth");
        fKom = Parameter("kom");
        fRho0 = Parameter("rho0");
        fScaleHeight = Parameter("h_atm");
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    double RocketDescent::Density(double y) const
    {
        return fRho0 * std::exp(-std::max(y, 0.) / fScaleHeight);
    }

    void RocketDescent::Guidance(const Vector &s, double &thrust, double &gimbal) const
    {
        const double x = s(0), y = s(1), vx = s(2), vy = s(3), th = s(4), om = s(5), m = s(6);
        thrust = gimbal = 0.;
        if (m <= fDryMass || y <= 0.) return;
        // Vertical loop: a capped descent speed, then braking at `decel` along
        // v^2 = v_touch^2 + 2 decel y, with the rate of the reference fed forward.
        const double v_brake = std::sqrt(fTouch * fTouch + 2. * fDecel * std::max(y, 0.));
        const bool braking = v_brake < fCruise;
        const double v_ref = -(braking ? v_brake : fCruise);
        const double a_ff = braking ? fDecel * std::max(-vy, 0.) / v_brake : 0.;
        const double a_vert = fG + a_ff + fKv * (v_ref - vy);
        thrust = std::clamp(m * a_vert / std::max(std::cos(th), 0.3), 0., fThrustMax);
        // Outer horizontal loop: the tilt that nulls the cross-range drift.
        const double th_des = std::clamp(fKx * x + fKvx * vx, -fThetaMax, fThetaMax);
        // Inner attitude loop: gimbal to track it and damp the tilt rate.
        gimbal = std::clamp(fKth * (th_des - th) - fKom * om, -fGimbalMax, fGimbalMax);
    }

    Vector RocketDescent::Derivative(const Vector &s, const Vector &u, double) const
    {
        const double y = s(1), vx = s(2), vy = s(3), th = s(4), om = s(5), m = s(6);
        Vector d(7, 0.);
        if (y <= 0.) return d;                                // landed: frozen on the pad

        double T, delta;
        if (NControl() > 0 && u.Size() >= 2)
        {
            // The command held over the step: throttle and gimbal, clamped.
            T = std::clamp(u(0), 0., 1.) * fThrustMax;
            delta = std::clamp(u(1), -fGimbalMax, fGimbalMax);
        }
        else
            Guidance(s, T, delta);

        const double mass = std::max(m, fDryMass);
        double ax = -(T / mass) * std::sin(th);
        double ay = (T / mass) * std::cos(th) - fG;
        if (fDrag)
        {
            const double v = std::hypot(vx, vy);
            const double k = 0.5 * Density(y) * fDragArea * v / mass;
            ax -= k * vx;
            ay -= k * vy;
        }
        const double inertia = mass * (fLength * fLength) / 12.;
        d(0) = vx;
        d(1) = vy;
        d(2) = ax;
        d(3) = ay;
        d(4) = om;
        d(5) = T * std::sin(delta) * fLever / inertia;
        d(6) = m > fDryMass ? -T / fExhaust : 0.;
        return d;
    }

    Vector RocketDescent::f(const Vector &x, const Vector &u, double t) const
    {
        Vector out = Kalman::Model::f(x, u, t);
        if (out(1) <= 0.)                                     // touchdown: clamp to the pad
        {
            out(1) = 0.;
            out(2) = out(3) = out(5) = 0.;                    // velocities and tilt rate
        }
        out(6) = std::max(out(6), fDryMass);
        return out;
    }

    Vector RocketDescent::h(const Vector &x, double) const
    {
        Vector z(3);
        z(0) = x(0);
        z(1) = x(1);
        z(2) = x(4);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::RocketDescent, "rocket-descent");
KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::RocketDescent, "rocket-descent-control", true);
