/**
 * *********************************************
 *
 * \file RocketDescent.h
 * \brief Header of the Kalman::Apps::Astro::RocketDescent class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Astro::RocketDescent
 * \brief Planar powered descent of a gimballed rigid booster - the twin of
 * `kalman.apps.astro.rocket.RocketDescentModel`.
 *
 * \details State `[x, y, vx, vy, theta, omega, m]` (SI units, `y` the altitude of
 * the base, `theta` the tilt from vertical); `[x, y, theta]` is observed.
 *
 * \code{.unparsed}
 *   vx' = -(T / m_eff) sin(theta)     - 0.5 rho(y) C_dA |v| vx / m_eff
 *   vy' =  (T / m_eff) cos(theta) - g - 0.5 rho(y) C_dA |v| vy / m_eff
 *   omega' = T sin(delta) lever / I(m_eff),   I(m) = m length^2 / 12
 *   m' = -T / ve                          (0 once dry)
 * \endcode
 *
 * The engine thrust `T` and gimbal `delta` come from the onboard guidance
 * (Guidance(): a vertical throttle loop on the descent profile, an outer loop
 * commanding a tilt against the cross-range drift, an inner attitude loop on
 * the gimbal), or, built with `control = true` (registered as
 * "rocket-descent-control", `l = 2`), from the control input
 * `u = [throttle, gimbal]`, clamped to `[0, 1]` and `+-gimbal_max`. On the pad
 * the state is frozen; after each sample a landed booster is put on the pad at
 * rest and the mass kept at or above the dry mass.
 *
 * | Parameter    | Meaning                                         |
 * |--------------|-------------------------------------------------|
 * | `m_dry`      | dry mass [kg]                                   |
 * | `ve`         | exhaust velocity [m/s]                          |
 * | `thrust_max` | maximum engine thrust [N]                       |
 * | `length`     | body length [m] (the inertia)                   |
 * | `lever`      | base-to-centre-of-mass lever arm [m]            |
 * | `drag_area`  | `C_d A` [m^2]                                   |
 * | `drag`       | 1: atmospheric drag on; 0: off                  |
 * | `g`          | gravity [m/s^2]                                 |
 * | `decel`      | deceleration of the near-ground flare [m/s^2]   |
 * | `v_cruise`   | capped descent speed away from the ground [m/s] |
 * | `v_touch`    | target touchdown speed [m/s]                    |
 * | `theta_max`  | largest commanded tilt [rad]                    |
 * | `gimbal_max` | largest engine gimbal [rad]                     |
 * | `kp_v`       | vertical descent-rate gain [1/s]                |
 * | `kx`, `kvx`  | cross-range to desired-tilt gains               |
 * | `kth`, `kom` | attitude-loop gains                             |
 * | `rho0`       | sea-level air density [kg/m^3]                  |
 * | `h_atm`      | atmospheric scale height [m]                    |
 * | `substeps`   | integrator steps per sample                     |
 * | `integrator` | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_ASTRO_ROCKETDESCENT_H
#define KALMAN_APPS_ASTRO_ROCKETDESCENT_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Astro
{
    class RocketDescent : public Kalman::Apps::Model
    {
        private:
            double fDryMass = 700., fExhaust = 2800., fThrustMax = 40000., fLength = 40., fLever = 20.;
            double fDragArea = 2.;
            bool fDrag = true;
            double fG = 9.80665, fDecel = 8., fCruise = 55., fTouch = 1.5;
            double fThetaMax = 0.35, fGimbalMax = 0.20;
            double fKv = 0.7, fKx = 6.0e-3, fKvx = 4.5e-2, fKth = 8., fKom = 4.;
            double fRho0 = 1.225, fScaleHeight = 8500.;

            double Density(double y) const;

        public:
            /// \param control false: the guidance flies the booster (l = 0);
            /// true: `u = [throttle, gimbal]` does (l = 2).
            explicit RocketDescent(bool control = false);

            const char *Title() const override { return "Rocket Descent (2-D, gimballed, lands upright)"; }
            const char *Description() const override
            { return "Planar powered descent of a gimballed rigid booster; position and attitude measured."; }

            /// The onboard guidance at state \p x: engine thrust [N] and gimbal angle [rad].
            void Guidance(const Vector &x, double &thrust, double &gimbal) const;

            /// The continuous dynamics of the 7 states.
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            /// One sample: Derivative() integrated, a landed booster put on the pad at rest.
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_ASTRO_ROCKETDESCENT_H
