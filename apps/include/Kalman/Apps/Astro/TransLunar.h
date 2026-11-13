/**
 * *********************************************
 *
 * \file TransLunar.h
 * \brief Header of the Kalman::Apps::Astro::TransLunar class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Astro::TransLunar
 * \brief Restricted four-body trans-lunar coast (Earth + Moon + Sun) — the twin of
 * `kalman.apps.astro.space.TransLunarModel`.
 *
 * \details Geocentric state `[x, y, z, vx, vy, vz]`, position in Mm and velocity
 * in km/s (time in ks, so 1 Mm/ks = 1 km/s); the position is observed. The
 * derivative is the Earth's central pull plus the third-body perturbations of
 * the Moon and the Sun,
 *
 * \code{.unparsed}
 *   a = -GM_E r / |r|^3 + sum_b GM_b [ (r_b - r) / |r_b - r|^3 - r_b / |r_b|^3 ]
 *   r_b(t) = D_b [cos(w_b t + phase_b), sin(w_b t + phase_b), 0]
 * \endcode
 *
 * with the Moon and the Sun on circular orbits, as the Python model has them
 * (`moon_position`, `sun_position` of `kalman.runtime.astro`). The framework
 * integrates it over one sample in `substeps` steps.
 *
 * | Parameter                | Meaning                                         |
 * |--------------------------|-------------------------------------------------|
 * | `gm_earth`, `gm_moon`, `gm_sun` | gravitational parameters [Mm^3/ks^2]     |
 * | `d_moon`, `d_sun`        | orbit radii of the Moon and the Sun [Mm]        |
 * | `omega_moon`, `omega_sun`| their angular rates [rad/ks]                    |
 * | `moon_phase`, `sun_phase`| their phases at t = 0 [rad]                     |
 * | `earth`, `moon`, `sun`   | 1 when the body pulls, 0 when it is left out    |
 * | `substeps`               | integrator steps per sample                     |
 * | `integrator`             | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince|
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_ASTRO_TRANSLUNAR_H
#define KALMAN_APPS_ASTRO_TRANSLUNAR_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Astro
{
    class TransLunar : public Kalman::Apps::Model
    {
        private:
            double fGMEarth = 0., fGMMoon = 0., fGMSun = 0.;
            double fDMoon = 0., fDSun = 0.;
            double fOmegaMoon = 0., fOmegaSun = 0.;
            double fMoonPhase = 0., fSunPhase = 0.;
            bool fEarth = true, fMoon = true, fSun = true;

        public:
            TransLunar();

            const char *Title() const override { return "Trans-Lunar Extended Model"; }
            const char *Description() const override
            { return "Geocentric restricted four-body coast: Earth, Moon and Sun on circular orbits."; }

            /// Geocentric acceleration [km/s per ks] at position p [Mm] and time t [ks], written to a.
            void Acceleration(const double *p, double t, double *a) const;

            /// The continuous dynamics: dr/dt = v, dv/dt = a(r, t).
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };

    /// Third-body perturbation in a frame centred on the primary, added to a:
    /// `GM [ (r_b - r) / |r_b - r|^3 - r_b / |r_b|^3 ]` (as `_third_body` in Python).
    void AddThirdBody(const double *r, const double *rb, double gm, double *a);
}

#endif // KALMAN_APPS_ASTRO_TRANSLUNAR_H
