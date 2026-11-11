/**
 * *********************************************
 *
 * \file Storm.h
 * \brief Header of the Kalman::Apps::Weather::Storm class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Weather::Storm
 * \brief Reduced tropical-cyclone track and intensity model — the twin of
 * `kalman.apps.weather.StormModel`.
 *
 * \details State `[lon, lat, v_e, v_n, I]`: centre [deg], translation velocity
 * [km/h] and intensity [kt]. Over one step `dt` the velocity turns at the rate
 * `omega`, the centre is advected by the velocity plus the beta drift, and the
 * intensity persists. A typhoon, a hurricane and a cyclone are one phenomenon,
 * so the hemisphere is the only regional setting.
 *
 * The observation is the centre fix `[lon, lat]`, plus the intensity when the
 * model is built with `observeIntensity`.
 *
 * | Parameter    | Meaning                                         |
 * |--------------|-------------------------------------------------|
 * | `omega`      | turn rate [rad/h]; 0 = straight-running          |
 * | `beta_drift` | beta-drift speed [km/h]                          |
 * | `hemisphere` | +1 north, -1 south (direction of the beta drift) |
 * | `dt_hours`   | step length [h]                                  |
 *
 * Steering flows and intensity tendencies given as Python callables have no
 * C++ counterpart: with either set, the Python class keeps its own dynamics.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_WEATHER_STORM_H
#define KALMAN_APPS_WEATHER_STORM_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Weather
{
    class Storm : public Kalman::Apps::Model
    {
        private:
            bool fObserveIntensity;
            double fOmega = 0.;     // rad/h
            double fDt = 6.;        // h
            double fBetaE = 0.;     // km/h
            double fBetaN = 0.;     // km/h

        public:
            static constexpr double kKmPerDegLat = 110.574;
            static constexpr double kKmPerDegLon = 111.320;   // x cos(lat)

            explicit Storm(bool observeIntensity = true);

            const char *Title() const override { return "Storm Tracker"; }
            const char *Description() const override
            { return "Reduced tropical-cyclone track + intensity model (beta drift + coordinated turn)."; }
            const char *Version() const override { return "3.0"; }

            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_WEATHER_STORM_H
