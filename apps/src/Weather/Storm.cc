/**
 * *********************************************
 *
 * \file Storm.cc
 * \brief Implementation of the Kalman::Apps::Weather::Storm class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Weather/Storm.h"

#include <Kalman/Factory.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Apps::Weather
{
    namespace
    {
        constexpr double kDegToRad = 3.14159265358979323846 / 180.;
    }

    Storm::Storm(bool observeIntensity)
        : Kalman::Apps::Model("storm", 5, observeIntensity ? 3 : 2), fObserveIntensity(observeIntensity)
    {
        DeclareState({{"lon", "deg E"}, {"lat", "deg N"}, {"ve", "km/h east"}, {"vn", "km/h north"},
                      {"intensity", "kt"}});
        if (observeIntensity) DeclareMeasurement({{"lon", "deg E"}, {"lat", "deg N"}, {"intensity", "kt"}});
        else                  DeclareMeasurement({{"lon", "deg E"}, {"lat", "deg N"}});
        DeclareParameter("omega", 0.);
        DeclareParameter("beta_drift", 5.);
        DeclareParameter("hemisphere", 1.);
        DeclareParameter("dt_hours", 6.);
        DeclareArray("intensity_coef", {});
        DeclareArray("intensity_mean", {});
        DeclareArray("intensity_scale", {});
        ParametersChanged();
        SetSampleRate(1. / fDt);

        // Defaults of the Python class; a Python instance pushes its own.
        const double q[5] = {0.25, 0.25, 6., 6., 8.};
        Matrix Q(5, 5, 0.);
        for (std::size_t i = 0; i < 5; ++i) Q(i, i) = q[i] * q[i];
        SetQ(Q);

        const double r[3] = {0.3, 0.3, 10.};
        Matrix R(NMeasure(), NMeasure(), 0.);
        for (std::size_t i = 0; i < NMeasure(); ++i) R(i, i) = r[i] * r[i];
        SetR(R);

        Vector x0(5);
        x0(0) = 135.; x0(1) = 15.; x0(2) = -20.; x0(3) = 6.; x0(4) = 80.;
        SetInitialState(x0);
    }

    void Storm::ParametersChanged()
    {
        fOmega = Parameter("omega");
        fDt = Parameter("dt_hours");
        fCoef = Array("intensity_coef");
        fMean = Array("intensity_mean");
        fScale = Array("intensity_scale");

        // Beta drift: ~20 degrees off due west, toward the pole.
        const double beta = Parameter("beta_drift");
        const double pole = Parameter("hemisphere") >= 0. ? 1. : -1.;
        const double angle = 20. * kDegToRad;
        fBetaE = -beta * std::cos(angle);
        fBetaN = pole * beta * std::sin(angle);
    }

    Vector Storm::f(const Vector &x, const Vector &, double) const
    {
        const double lon = x(0), lat = x(1), ve = x(2), vn = x(3);
        const double vte = ve + fBetaE, vtn = vn + fBetaN;   // advecting velocity (+ beta)

        double de, dn, veNext = ve, vnNext = vn;
        if (std::abs(fOmega) < 1e-9)
        {
            de = vte * fDt;
            dn = vtn * fDt;
        }
        else
        {
            const double s = std::sin(fOmega * fDt), c = std::cos(fOmega * fDt);
            de = (vte * s - vtn * (1 - c)) / fOmega;           // curved displacement [km]
            dn = (vte * (1 - c) + vtn * s) / fOmega;
            veNext = ve * c - vn * s;                          // the velocity turns
            vnNext = ve * s + vn * c;
        }

        Vector out(5);
        out(0) = lon + de / (kKmPerDegLon * std::cos(lat * kDegToRad));
        out(1) = lat + dn / kKmPerDegLat;
        out(2) = veNext;
        out(3) = vnNext;
        out(4) = x(4) + IntensityTendency(lon, lat, x(4)) * fDt;   // persists without a tendency
        return out;
    }

    double Storm::IntensityTendency(double lon, double lat, double intensity) const
    {
        if (fCoef.empty()) return 0.;
        // analytic_environment: a warm pool near 135 E, 10 N, and shear growing poleward of 15 degrees.
        const double a = (lon - 135.) / 45., b = (lat - 10.) / 14.;
        const double sst = 30. * std::exp(-(a * a) - b * b);
        const double shear = 3. + 0.9 * std::max(std::abs(lat) - 15., 0.);
        const double features[3] = {sst, shear, intensity};
        double tendency = 0.;
        for (std::size_t i = 0; i < 3; ++i) tendency += (features[i] - fMean[i]) / fScale[i] * fCoef[i];
        return tendency + fCoef[3];
    }

    void Storm::CheckShapes() const
    {
        Kalman::Model::CheckShapes();
        if (!fCoef.empty() && (fCoef.size() != 4 || fMean.size() != 3 || fScale.size() != 3))
            throw std::runtime_error("Storm: a learned intensity tendency needs 4 coefficients, 3 means and 3 scales.");
    }

    Vector Storm::h(const Vector &x, double) const
    {
        Vector z(NMeasure());
        z(0) = x(0);
        z(1) = x(1);
        if (fObserveIntensity) z(2) = x(4);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Weather::Storm, "storm");
KALMAN_REGISTER_MODEL(Kalman::Apps::Weather::Storm, "storm-track-only", false);
