/**
 * *********************************************
 *
 * \file BarotropicFlow.h
 * \brief Header of the Kalman::Apps::Weather::BarotropicFlow class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Weather::BarotropicFlow
 * \brief Barotropic vorticity on a mid-latitude beta-plane — the twin of
 * `kalman.apps.weather.BarotropicFlowModel`.
 *
 * \details The field model of Weather::KolmogorovFlow with the physics of the
 * beta-plane: `beta = 2 Omega cos(lat) / a`, no forcing, a rectangular box in
 * metres, SI units. The defaults are the Python class's: 64 x 64 over a
 * 20 x 20 degree window at 50 N (2,224 km by 1,430 km), nu = 2e4 m^2/s,
 * 60 steps of 60 s per cycle. The parameters and arrays are KolmogorovFlow's.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_WEATHER_BAROTROPICFLOW_H
#define KALMAN_APPS_WEATHER_BAROTROPICFLOW_H

#include "Kalman/Apps/Weather/KolmogorovFlow.h"

namespace Kalman::Apps::Weather
{
    class BarotropicFlow : public KolmogorovFlow
    {
        public:
            static constexpr double kOmega = 7.292e-5;          // Earth's rotation rate [rad/s]
            static constexpr double kEarthRadius = 6.371e6;     // [m]

            BarotropicFlow();

            const char *Title() const override { return "Barotropic beta-plane flow"; }
            const char *Description() const override
            { return "2D barotropic vorticity with the beta effect and sparse point probes (SI units)."; }
            const char *Version() const override { return "1.0"; }
    };
}

#endif // KALMAN_APPS_WEATHER_BAROTROPICFLOW_H
