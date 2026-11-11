/**
 * *********************************************
 *
 * \file CarSpeed.h
 * \brief Header of the Kalman::Apps::Vehicles::CarSpeed class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Vehicles::CarSpeed
 * \brief Straight-line acceleration run of one car — the twin of
 * `kalman.apps.vehicles.CarSpeedModel`.
 *
 * \details State `[s, v]`, distance and speed; the position is observed. The
 * speed follows the longitudinal force balance
 *
 * \code{.unparsed}
 *   v' = [ F_drive(v) - F_drag(v) - F_roll - F_grade ] / m
 *   F_drive = min( mu m g , P / max(|v|, v_eps) )
 * \endcode
 *
 * grip-limited off the line and power-limited at speed. One RK4 step per
 * sample, after which the speed is clamped at zero: the car does not roll back.
 *
 * | Parameter   | Meaning                                  |
 * |-------------|------------------------------------------|
 * | `mass`      | vehicle mass [kg]                        |
 * | `power`     | peak power at the wheels [W]             |
 * | `mu`        | tyre-road grip coefficient               |
 * | `drag_area` | `C_d A` [m^2]                            |
 * | `rho`       | air density [kg/m^3]                     |
 * | `c_roll`    | rolling-resistance coefficient           |
 * | `grade`     | road grade [rad]                         |
 * | `g`         | gravity [m/s^2]                          |
 * | `v_eps`     | speed floor [m/s] that keeps `P/v` finite |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_VEHICLES_CARSPEED_H
#define KALMAN_APPS_VEHICLES_CARSPEED_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Vehicles
{
    class CarSpeed : public Kalman::Apps::Model
    {
        private:
            double fMass = 1500., fPower = 150000., fMu = 1., fDragArea = 0.66, fRho = 1.225;
            double fRoll = 0.015, fGrade = 0., fG = 9.81, fSpeedFloor = 0.5;

        public:
            CarSpeed();

            const char *Title() const override { return "Car Speed Test (straight line)"; }
            const char *Description() const override
            { return "Longitudinal car dynamics (traction/power-limited drive + drag); position measured."; }

            /// Longitudinal acceleration [m/s^2] at speed \p v: the force balance over the mass.
            double Acceleration(double v) const;

            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_VEHICLES_CARSPEED_H
