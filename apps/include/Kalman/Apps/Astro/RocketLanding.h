/**
 * *********************************************
 *
 * \file RocketLanding.h
 * \brief Header of the Kalman::Apps::Astro::RocketLanding class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Astro::RocketLanding
 * \brief Vertical propulsive landing (a hoverslam) - the twin of
 * `kalman.apps.astro.rocket.RocketLandingModel`.
 *
 * \details State `[h, v, m]`, altitude, vertical speed and mass (SI units); the
 * altitude is observed. The stage falls engine-off, then a closed-loop burn
 * tracks the descent profile `v_ref(h) = -sqrt(2 decel h) - v_touch` to the pad:
 *
 * \code{.unparsed}
 *   a_T = clip( kp (v_ref - v) + g - a_drag , 0 , T_max / m_eff )   (0 when dry or landed)
 *   h' = v
 *   v' = a_T - g - 0.5 rho(h) (C_d A) v |v| / m_eff
 *   m' = -a_T m_eff / ve                                            (0 once dry)
 * \endcode
 *
 * On the pad (`h <= 0`) the state is frozen; after each sample a landed stage
 * is clamped to the pad at rest and the mass kept at or above the dry mass.
 *
 * | Parameter    | Meaning                                          |
 * |--------------|--------------------------------------------------|
 * | `m_dry`      | dry mass [kg]                                    |
 * | `ve`         | effective exhaust velocity [m/s]                 |
 * | `thrust_max` | maximum engine thrust [N]                        |
 * | `drag_area`  | `C_d A` [m^2]                                    |
 * | `drag`       | 1: atmospheric drag on; 0: off                   |
 * | `g`          | gravity [m/s^2]                                  |
 * | `decel`      | reference deceleration of the profile [m/s^2]    |
 * | `v_touch`    | target touchdown speed [m/s]                     |
 * | `kp`         | descent-rate tracking gain [1/s]                 |
 * | `rho0`       | sea-level air density [kg/m^3]                   |
 * | `h_atm`      | atmospheric scale height [m]                     |
 * | `substeps`   | integrator steps per sample                      |
 * | `integrator` | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince  |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_ASTRO_ROCKETLANDING_H
#define KALMAN_APPS_ASTRO_ROCKETLANDING_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Astro
{
    class RocketLanding : public Kalman::Apps::Model
    {
        private:
            double fDryMass = 700., fExhaust = 2500., fThrustMax = 30000., fDragArea = 0.25;
            bool fDrag = true;
            double fG = 9.80665, fDecel = 12., fTouch = 1., fGain = 1.5;
            double fRho0 = 1.225, fScaleHeight = 8500.;

            double Density(double h) const;

        public:
            RocketLanding();

            const char *Title() const override { return "Rocket Landing (hoverslam / powered descent)"; }
            const char *Description() const override
            { return "1-D propulsive landing: free fall, then a closed-loop burn to the pad; altitude measured."; }

            /// The onboard guidance: commanded thrust acceleration [m/s^2], upward, at state \p x.
            double Guidance(const Vector &x) const;

            /// The continuous dynamics: [h, v, m]' = [v, a_T - g - drag, -a_T m_eff / ve].
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            /// One sample: Derivative() integrated, a landed stage clamped to the pad.
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_ASTRO_ROCKETLANDING_H
