/**
 * *********************************************
 *
 * \file DriftChamber.h
 * \brief Header of the Kalman::Apps::Tracking::DriftChamber class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Tracking::DriftChamber
 * \brief A charged particle in an axial field, sampled on concentric layers.
 *
 * \details The twin of the Python `DriftChamberModel`
 * (modules/apps/tracking/driftchamber.py). In the transverse plane the
 * trajectory is a circular arc. The state is `[x, y, px, py]`, position in cm
 * and momentum in GeV/c; a hit is the position `(x, y)` where the arc crosses
 * a layer.
 *
 * The step is not a step in time: `f(x, u, t)` advances the arc (RK4 in arc
 * length, sub-steps of `step` cm) until it first crosses the radius `t`, the
 * next layer, which the track-finding engine passes in place of a time. It
 * returns NaN when the track curls back before reaching it.
 *
 * | Parameter | Default | |
 * |---|---|---|
 * | `Bz` | 1.5 | axial field [T] |
 * | `q` | 1 | charge [e] |
 * | `sigma_hit` | 0.013 | hit resolution [cm] |
 * | `p_seed` | 1 | momentum guess of a seed [GeV/c] |
 * | `step` | 0.5 | arc-length sub-step [cm] |
 *
 * The array `radii` holds the layer radii [cm], innermost first.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_TRACKING_DRIFTCHAMBER_H
#define KALMAN_APPS_TRACKING_DRIFTCHAMBER_H

#include "Kalman/Apps/Model.h"
#include "Kalman/Tracking.h"

#include <vector>

namespace Kalman::Apps::Tracking
{
    class DriftChamber : public Kalman::Apps::Model, public Kalman::Tracking::Detector
    {
        private:
            double fBz = 1.5, fCharge = 1., fSigmaHit = 0.013, fPSeed = 1., fStep = 0.5;

            void Rhs(const double *s, double *d) const;
            void RK4(const double *s, double dl, double *out) const;

        public:
            /// Curvature constant: 1/R [1/cm] = kKappa q [e] B [T] / p [GeV/c] (0.3 in m, T, GeV).
            static constexpr double kKappa = 0.299792458e-2;

            DriftChamber();

            const char *Title() const override { return "Drift Chamber (transverse tracker)"; }
            const char *Description() const override
            { return "Charged particle in an axial field, sampled on concentric layers; (x, y) observed."; }

            // --- Kalman::Tracking::Detector ------------------------------------
            std::vector<double> Radii() const override { return Array("radii"); }
            /// A seed from one hit: at the hit, momentum `p_seed` pointing away from the origin.
            void SeedState(const Kalman::Tracking::Hit &hit, Vector &x0, Matrix &P0) const override;

            /// The state at the first outward crossing of \p rTarget; false if the track curls back first.
            bool PropagateToRadius(const Vector &state, double rTarget, Vector &out) const;

            /// Propagate to the radius \p t (NaN when it cannot be reached).
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            /// The transverse position.
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_TRACKING_DRIFTCHAMBER_H
