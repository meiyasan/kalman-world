/**
 * *********************************************
 *
 * \file TransLunar.cc
 * \brief Implementation of the Kalman::Apps::Astro::TransLunar class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Astro/TransLunar.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>

namespace Kalman::Apps::Astro
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;
        // kalman.runtime.astro in scaled units (1 Mm, 1 ks): GM [m^3/s^2] * 1e6 / 1e18.
        constexpr double kGMScale = 1e-12;
    }

    void AddThirdBody(const double *r, const double *rb, double gm, double *a)
    {
        const double d[3] = {rb[0] - r[0], rb[1] - r[1], rb[2] - r[2]};
        const double nd = std::pow(std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]), 3.);
        const double nb = std::pow(std::sqrt(rb[0] * rb[0] + rb[1] * rb[1] + rb[2] * rb[2]), 3.);
        for (std::size_t i = 0; i < 3; ++i)
            a[i] += gm * (d[i] / nd - rb[i] / nb);
    }

    TransLunar::TransLunar()
        : Kalman::Apps::Model("translunar", 6, 3)
    {
        // Defaults are those of kalman.runtime.astro (the Python model pushes its own).
        DeclareParameter("gm_earth", 3.986004418e14 * kGMScale);
        DeclareParameter("gm_moon", 4.9028001184e12 * kGMScale);
        DeclareParameter("gm_sun", 1.32712440018e20 * kGMScale);
        DeclareParameter("d_moon", 384.4);
        DeclareParameter("d_sun", 1.495978707e5);
        DeclareParameter("omega_moon", 2. * kPi / (27.321661 * 86.4));
        DeclareParameter("omega_sun", 2. * kPi / (365.256363 * 86.4));
        DeclareParameter("moon_phase", 125. * kPi / 180.);
        DeclareParameter("sun_phase", 60. * kPi / 180.);
        DeclareParameter("earth", 1.);
        DeclareParameter("moon", 1.);
        DeclareParameter("sun", 1.);
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();

        DeclareState({{"x", "Mm"}, {"y", "Mm"}, {"z", "Mm"},
                      {"vx", "km/s"}, {"vy", "km/s"}, {"vz", "km/s"}});
        DeclareMeasurement({{"x", "Mm"}, {"y", "Mm"}, {"z", "Mm"}});
        ParametersChanged();
        SetTimeUnit(1000.);                    // time in ks
        SetSampleRate(1. / 0.12);              // dt_s = 120 s, in ks

        Matrix Q(6, 6, 0.);
        for (std::size_t i = 0; i < 3; ++i) { Q(i, i) = 1e-6; Q(i + 3, i + 3) = 1e-7; }
        SetQ(Q);
        Matrix R(3, 3, 0.);
        for (std::size_t i = 0; i < 3; ++i) R(i, i) = 25e-6;   // 5 km tracking, in Mm
        SetR(R);
    }

    void TransLunar::ParametersChanged()
    {
        fGMEarth = Parameter("gm_earth");
        fGMMoon = Parameter("gm_moon");
        fGMSun = Parameter("gm_sun");
        fDMoon = Parameter("d_moon");
        fDSun = Parameter("d_sun");
        fOmegaMoon = Parameter("omega_moon");
        fOmegaSun = Parameter("omega_sun");
        fMoonPhase = Parameter("moon_phase");
        fSunPhase = Parameter("sun_phase");
        fEarth = Parameter("earth") != 0.;
        fMoon = Parameter("moon") != 0.;
        fSun = Parameter("sun") != 0.;
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    void TransLunar::Acceleration(const double *r, double t, double *a) const
    {
        // The order of the Python sum: Earth, then the Moon, then the Sun.
        a[0] = a[1] = a[2] = 0.;
        if (fEarth)
        {
            const double n3 = std::pow(std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), 3.);
            for (std::size_t i = 0; i < 3; ++i) a[i] += -fGMEarth * r[i] / n3;
        }
        if (fMoon)
        {
            const double angle = fOmegaMoon * t + fMoonPhase;
            const double rb[3] = {fDMoon * std::cos(angle), fDMoon * std::sin(angle), 0.};
            AddThirdBody(r, rb, fGMMoon, a);
        }
        if (fSun)
        {
            const double angle = fOmegaSun * t + fSunPhase;
            const double rb[3] = {fDSun * std::cos(angle), fDSun * std::sin(angle), 0.};
            AddThirdBody(r, rb, fGMSun, a);
        }
    }

    Vector TransLunar::Derivative(const Vector &s, const Vector &, double t) const
    {
        const double r[3] = {s(0), s(1), s(2)};
        double a[3];
        Acceleration(r, t, a);
        Vector d(6);
        for (std::size_t i = 0; i < 3; ++i) { d(i) = s(i + 3); d(i + 3) = a[i]; }
        return d;
    }

    Vector TransLunar::f(const Vector &x, const Vector &u, double t) const
    {
        // Continuous: the framework integrates Derivative() over one sample. The
        // Python model has no closing step (after_step is the identity).
        return Kalman::Model::f(x, u, t);
    }

    Vector TransLunar::h(const Vector &x, double) const
    {
        Vector z(3);
        for (std::size_t i = 0; i < 3; ++i) z(i) = x(i);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::TransLunar, "translunar");
