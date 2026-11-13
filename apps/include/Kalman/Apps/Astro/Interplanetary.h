/**
 * *********************************************
 *
 * \file Interplanetary.h
 * \brief Header of the Kalman::Apps::Astro::Interplanetary class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Astro::Interplanetary
 * \brief Heliocentric N-body cruise — the twin of
 * `kalman.apps.astro.space.InterplanetaryModel`.
 *
 * \details Sun-centred state `[x, y, z, vx, vy, vz]`, position in Mm and
 * velocity in km/s (time in ks); the position is observed. The derivative is the
 * Sun's central pull plus the third-body perturbations of the planets,
 *
 * \code{.unparsed}
 *   a = -GM_S r / |r|^3 + sum_b GM_b [ (r_b - r) / |r_b - r|^3 - r_b / |r_b|^3 ]
 * \endcode
 *
 * where each planet's position r_b(t) is interpolated linearly in a table
 * sampled every `grid_step` from t = 0 and held at its ends, as `numpy.interp`
 * does: the table the Python model takes from the ephemeris when it is built
 * (`_PerturberField`). With no perturber the model is a two-body Sun cruise.
 *
 * | Parameter              | Meaning                                              |
 * |------------------------|------------------------------------------------------|
 * | `gm_sun`               | the Sun's gravitational parameter [Mm^3/ks^2]        |
 * | `bodies`               | number of perturbing bodies B                        |
 * | `samples`              | number of table rows N per body                      |
 * | `grid_step`            | time between two rows [ks]                           |
 * | `body<i>_gm`           | gravitational parameter of body i [Mm^3/ks^2]        |
 * | `body<i>_x<k>`, `_y<k>`, `_z<k>` | heliocentric position of body i at row k [Mm] |
 * | `substeps`             | integrator steps per sample                          |
 * | `integrator`           | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince     |
 *
 * The per-body parameters exist once `bodies` and `samples` are set: setting
 * them declares the names they call for (a table is a set of named scalars).
 * The impulsive mid-course burn of the Python model (`control=True`) has no
 * counterpart here: with a control the Python class keeps its own dynamics.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_ASTRO_INTERPLANETARY_H
#define KALMAN_APPS_ASTRO_INTERPLANETARY_H

#include "Kalman/Apps/Model.h"

#include <vector>

namespace Kalman::Apps::Astro
{
    class Interplanetary : public Kalman::Apps::Model
    {
        private:
            double fGMSun = 0.;
            std::size_t fBodies = 0, fSamples = 0;
            double fGridStep = 0.;

            // The perturber table, read from the parameters when first needed after
            // a change (a run sets thousands of them, one call each).
            mutable bool fStale = true;
            mutable std::vector<double> fGrid;            // N times [ks]
            mutable std::vector<double> fGM;              // B
            mutable std::vector<double> fTable;           // B x N x 3, row-major

            void Refresh() const;
            /// Position of body b at time t, interpolated as numpy.interp does.
            void Position(std::size_t b, double t, double *rb) const;

        public:
            Interplanetary();

            const char *Title() const override { return "Interplanetary Extended Model"; }
            const char *Description() const override
            { return "Heliocentric N-body cruise: the Sun plus tabulated planetary perturbations."; }

            /// Heliocentric acceleration at position p [Mm] and time t [ks], written to a.
            void Acceleration(const double *p, double t, double *a) const;

            /// The continuous dynamics: dr/dt = v, dv/dt = a(r, t).
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_ASTRO_INTERPLANETARY_H
