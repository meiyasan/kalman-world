/**
 * *********************************************
 *
 * \file Interplanetary.cc
 * \brief Implementation of the Kalman::Apps::Astro::Interplanetary class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Astro/Interplanetary.h"
#include "Kalman/Apps/Astro/TransLunar.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>
#include <string>

namespace Kalman::Apps::Astro
{
    namespace
    {
        std::string BodyName(std::size_t b, const char *what, std::size_t k = 0, bool row = false)
        {
            std::string name = "body" + std::to_string(b) + "_" + what;
            if (row) name += std::to_string(k);
            return name;
        }
    }

    Interplanetary::Interplanetary()
        : Kalman::Apps::Model("interplanetary", 6, 3)
    {
        DeclareParameter("gm_sun", 1.32712440018e20 * 1e-12);   // kalman.runtime.astro, scaled
        DeclareParameter("bodies", 0.);
        DeclareParameter("samples", 0.);
        DeclareParameter("grid_step", 172.8);                    // 2 days [ks]
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();

        DeclareState({{"x", "Mm"}, {"y", "Mm"}, {"z", "Mm"},
                      {"vx", "km/s"}, {"vy", "km/s"}, {"vz", "km/s"}});
        DeclareMeasurement({{"x", "Mm"}, {"y", "Mm"}, {"z", "Mm"}});
        ParametersChanged();
        SetSampleRate(1. / 43.2);              // dt_s = 12 h, in ks

        Matrix Q(6, 6, 0.);
        for (std::size_t i = 0; i < 3; ++i) { Q(i, i) = 1e-4; Q(i + 3, i + 3) = 1e-8; }
        SetQ(Q);
        Matrix R(3, 3, 0.);
        for (std::size_t i = 0; i < 3; ++i) R(i, i) = 0.05 * 0.05;   // 50 km tracking, in Mm
        SetR(R);
    }

    void Interplanetary::ParametersChanged()
    {
        fGMSun = Parameter("gm_sun");
        fGridStep = Parameter("grid_step");
        const std::size_t bodies = static_cast<std::size_t>(std::max(0., Parameter("bodies")));
        const std::size_t samples = static_cast<std::size_t>(std::max(0., Parameter("samples")));
        // A larger table declares the names it is filled through. Names stay
        // declared when the table shrinks; only the first bodies x samples are read.
        if (bodies > fBodies || samples > fSamples)
        {
            for (std::size_t b = 0; b < bodies; ++b)
            {
                if (!HasParameter(BodyName(b, "gm"))) DeclareParameter(BodyName(b, "gm"), 0.);
                for (std::size_t k = 0; k < samples; ++k)
                    for (const char *axis : {"x", "y", "z"})
                        if (!HasParameter(BodyName(b, axis, k, true)))
                            DeclareParameter(BodyName(b, axis, k, true), 0.);
            }
        }
        fBodies = bodies;
        fSamples = samples;
        fStale = true;
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    void Interplanetary::Refresh() const
    {
        if (!fStale) return;
        const std::size_t B = fSamples ? fBodies : 0, N = fSamples;
        fGrid.assign(N, 0.);
        // numpy.arange(0, stop, step): row k is at k * step.
        for (std::size_t k = 0; k < N; ++k) fGrid[k] = static_cast<double>(k) * fGridStep;
        fGM.assign(B, 0.);
        fTable.assign(B * N * 3, 0.);
        for (std::size_t b = 0; b < B; ++b)
        {
            fGM[b] = Parameter(BodyName(b, "gm"));
            for (std::size_t k = 0; k < N; ++k)
            {
                fTable[(b * N + k) * 3 + 0] = Parameter(BodyName(b, "x", k, true));
                fTable[(b * N + k) * 3 + 1] = Parameter(BodyName(b, "y", k, true));
                fTable[(b * N + k) * 3 + 2] = Parameter(BodyName(b, "z", k, true));
            }
        }
        fStale = false;
    }

    void Interplanetary::Position(std::size_t b, double t, double *rb) const
    {
        // numpy.interp: the end values outside the grid, a grid point's own value
        // on it, and slope * (t - t_j) + y_j between t_j <= t < t_j+1.
        const std::size_t N = fGrid.size();
        const double *row = fTable.data() + b * N * 3;
        if (N == 1 || t <= fGrid.front())
        {
            for (std::size_t i = 0; i < 3; ++i) rb[i] = row[i];
            return;
        }
        if (t >= fGrid.back())
        {
            for (std::size_t i = 0; i < 3; ++i) rb[i] = row[(N - 1) * 3 + i];
            return;
        }
        const std::size_t j = static_cast<std::size_t>(std::upper_bound(fGrid.begin(), fGrid.end(), t) - fGrid.begin()) - 1;
        const double *y0 = row + j * 3, *y1 = row + (j + 1) * 3;
        for (std::size_t i = 0; i < 3; ++i)
        {
            if (t == fGrid[j]) { rb[i] = y0[i]; continue; }
            const double slope = (y1[i] - y0[i]) / (fGrid[j + 1] - fGrid[j]);
            rb[i] = slope * (t - fGrid[j]) + y0[i];
        }
    }

    void Interplanetary::Acceleration(const double *r, double t, double *a) const
    {
        Refresh();
        // The Sun's central pull, then each perturber in the order of the table.
        const double n3 = std::pow(std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]), 3.);
        for (std::size_t i = 0; i < 3; ++i) a[i] = -fGMSun * r[i] / n3;
        for (std::size_t b = 0; b < fGM.size(); ++b)
        {
            double rb[3];
            Position(b, t, rb);
            AddThirdBody(r, rb, fGM[b], a);
        }
    }

    Vector Interplanetary::Derivative(const Vector &s, const Vector &, double t) const
    {
        const double r[3] = {s(0), s(1), s(2)};
        double a[3];
        Acceleration(r, t, a);
        Vector d(6);
        for (std::size_t i = 0; i < 3; ++i) { d(i) = s(i + 3); d(i + 3) = a[i]; }
        return d;
    }

    Vector Interplanetary::f(const Vector &x, const Vector &u, double t) const
    {
        // Continuous, no closing step: the framework integrates Derivative().
        return Kalman::Model::f(x, u, t);
    }

    Vector Interplanetary::h(const Vector &x, double) const
    {
        Vector z(3);
        for (std::size_t i = 0; i < 3; ++i) z(i) = x(i);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Astro::Interplanetary, "interplanetary");
