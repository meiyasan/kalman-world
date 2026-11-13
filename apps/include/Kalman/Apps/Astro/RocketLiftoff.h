/**
 * *********************************************
 *
 * \file RocketLiftoff.h
 * \brief Header of the Kalman::Apps::Astro::RocketLiftoff class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Astro::RocketLiftoff
 * \brief Vertical liftoff with a finite propellant load - the twin of
 * `kalman.apps.astro.rocket.RocketLiftoffModel`.
 *
 * \details State `[h, v, m]`, altitude, climb rate and mass (SI units); the
 * altitude is observed. The mass burns down, so at constant thrust the
 * acceleration grows until burnout (the rocket equation in flight):
 *
 * \code{.unparsed}
 *   h' = v
 *   v' = T / m_eff - g - 0.5 rho(h) (C_d A) v |v| / m_eff
 *   m' = -mdot                    while m > m_dry, then 0 (T = ve mdot, then 0)
 *   m_eff = max(m, m_dry),  rho(h) = rho0 exp(-max(h, 0) / h_atm)
 * \endcode
 *
 * After each sample the mass is kept at or above the dry mass.
 *
 * | Parameter    | Meaning                                         |
 * |--------------|-------------------------------------------------|
 * | `m_dry`      | dry (burnout) mass [kg]                         |
 * | `ve`         | effective exhaust velocity [m/s]                |
 * | `mdot`       | propellant mass flow [kg/s]                     |
 * | `drag_area`  | `C_d A` [m^2]                                   |
 * | `drag`       | 1: atmospheric drag on; 0: off                  |
 * | `g`          | gravity [m/s^2]                                 |
 * | `rho0`       | sea-level air density [kg/m^3]                  |
 * | `h_atm`      | atmospheric scale height [m]                    |
 * | `substeps`   | integrator steps per sample                     |
 * | `integrator` | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_ASTRO_ROCKETLIFTOFF_H
#define KALMAN_APPS_ASTRO_ROCKETLIFTOFF_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Astro
{
    class RocketLiftoff : public Kalman::Apps::Model
    {
        private:
            double fDryMass = 400., fExhaust = 2500., fFlow = 8., fDragArea = 0.25;
            bool fDrag = true;
            double fG = 9.80665, fRho0 = 1.225, fScaleHeight = 8500.;

        public:
            RocketLiftoff();

            const char *Title() const override { return "Rocket Liftoff (vertical, fuel burns)"; }
            const char *Description() const override
            { return "1-D vertical liftoff whose mass burns down to burnout; altitude measured."; }

            /// The continuous dynamics: [h, v, m]' = [v, a(h, v, m), -mdot].
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            /// One sample: Derivative() integrated, with the mass kept at or above the dry mass.
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_ASTRO_ROCKETLIFTOFF_H
