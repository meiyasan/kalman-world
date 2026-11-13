/**
 * *********************************************
 *
 * \file RocketAscent.h
 * \brief Header of the Kalman::Apps::Astro::RocketAscent class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Astro::RocketAscent
 * \brief Planar guided ascent, thrust as the control input - the twin of
 * `kalman.apps.astro.rocket.RocketAscentModel`.
 *
 * \details State `[x, y, vx, vy]` in Earth-centred axes, scaled units (Mm and
 * km/s, time in ks); control `u = [ax, ay]`, the thrust acceleration; the
 * position is observed. Constant mass:
 *
 * \code{.unparsed}
 *   r' = v
 *   v' = -GM r / |r|^3 + u - Cb rho(alt) |v| v     (the drag term only with drag on)
 *   rho(alt) = exp(-max(|r| - R, 0) / H)
 * \endcode
 *
 * | Parameter         | Meaning                                         |
 * |-------------------|-------------------------------------------------|
 * | `GM`              | Earth's gravitational parameter [Mm^3/ks^2]     |
 * | `R`               | Earth's radius [Mm]                             |
 * | `H`               | atmospheric scale height [Mm]                   |
 * | `ballistic_coeff` | lumped `0.5 C_d A / m`, scaled                  |
 * | `drag`            | 1: atmospheric drag on; 0: off                  |
 * | `substeps`        | integrator steps per sample                     |
 * | `integrator`      | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_ASTRO_ROCKETASCENT_H
#define KALMAN_APPS_ASTRO_ROCKETASCENT_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Astro
{
    class RocketAscent : public Kalman::Apps::Model
    {
        private:
            double fGM = 398.6004, fRadius = 6.3781, fScaleHeight = 0.0085, fBallistic = 2.0e-3;
            bool fDrag = false;

        public:
            RocketAscent();

            const char *Title() const override { return "Rocket Ascent (guided, thrust as control)"; }
            const char *Description() const override
            { return "Planar powered ascent: central gravity plus thrust as the control input; position measured."; }

            /// The continuous dynamics: [r, v]' = [v, gravity + thrust + drag].
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_ASTRO_ROCKETASCENT_H
