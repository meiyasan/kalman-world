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
#include <stdexcept>
#include <string>

namespace Kalman::Apps::Astro
{
    Interplanetary::Interplanetary()
        : Kalman::Apps::Model("interplanetary", 6, 3)
    {
        DeclareParameter("gm_sun", 1.32712440018e20 * 1e-12);   // kalman.runtime.astro, scaled
        DeclareArray("body_gm", {});            // one per perturbing body
        DeclareArray("body_positions", {});     // bodies x samples x 3, row-major
        DeclareParameter("grid_step", 172.8);                    // 2 days [ks]
        DeclareParameter("substeps", 1.);      // integrator steps per sample
        DeclareParameter("integrator", 1.);    // IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince
        UseContinuousDynamics();

        DeclareState({{"x", "Mm"}, {"y", "Mm"}, {"z", "Mm"},
                      {"vx", "km/s"}, {"vy", "km/s"}, {"vz", "km/s"}});
        DeclareMeasurement({{"x", "Mm"}, {"y", "Mm"}, {"z", "Mm"}});
        ParametersChanged();
        SetTimeUnit(1000.);                    // time in ks
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
        fStale = true;
        SetSubsteps(static_cast<std::size_t>(std::max(1., Parameter("substeps"))));
        const IntegratorKind kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
        if (kind != DefaultIntegrator()) SetIntegrator(kind);
    }

    void Interplanetary::Refresh() const
    {
        if (!fStale) return;
        fGM = Array("body_gm");
        fTable = Array("body_positions");
        const std::size_t B = fGM.size();
        // The table holds the same number of rows for every body; CheckShapes() says when it does not.
        const std::size_t N = (B && fTable.size() % (3 * B) == 0) ? fTable.size() / (3 * B) : 0;
        if (N == 0) { fGM.clear(); fTable.clear(); }
        fGrid.assign(N, 0.);
        // numpy.arange(0, stop, step): row k is at k * step.
        for (std::size_t k = 0; k < N; ++k) fGrid[k] = static_cast<double>(k) * fGridStep;
        fStale = false;
    }

    void Interplanetary::CheckShapes() const
    {
        const std::size_t B = Array("body_gm").size(), T = Array("body_positions").size();
        if ((B == 0 && T != 0) || (B != 0 && (T == 0 || T % (3 * B) != 0)))
            throw std::runtime_error("Kalman::Apps::Astro::Interplanetary: body_positions holds " + std::to_string(T)
                                     + " values, which is not bodies x samples x 3 for the " + std::to_string(B)
                                     + " bodies of body_gm.");
        Kalman::Apps::Model::CheckShapes();
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
