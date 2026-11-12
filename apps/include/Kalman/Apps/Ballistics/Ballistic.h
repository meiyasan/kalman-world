/**
 * *********************************************
 *
 * \file Ballistic.h
 * \brief Header of the Kalman::Apps::Ballistics::Ballistic class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Ballistics::Ballistic
 * \brief Nonlinear 3-D ballistic model — the twin of
 * `kalman.apps.ballistics.BallisticModel`.
 *
 * \details Gravity plus quadratic aerodynamic drag. The state is axis-major,
 * `[px, vx, ax, py, vy, ay, pz, vz, az]` at order 3 and `[px, vx, py, vy, pz, vz]`
 * at order 2; the three positions are observed.
 *
 * | Parameter         | Meaning                                              |
 * |-------------------|------------------------------------------------------|
 * | `c_drag`          | drag coefficient [kg/m]                              |
 * | `m`               | object mass [kg]                                     |
 * | `g`               | surface gravity [m/s^2], used when `surface_gravity` |
 * | `surface_gravity` | 1: constant `-g` along z; 0: `-G M p / r^3`          |
 * | `G`, `M`          | gravitational constant and central mass (orbital)    |
 *
 * Thrust controls, a speed-dependent process noise and user-supplied `f`/`h`
 * have no C++ counterpart here: with any of them, the Python class keeps its
 * own dynamics.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_BALLISTICS_BALLISTIC_H
#define KALMAN_APPS_BALLISTICS_BALLISTIC_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Ballistics
{
    class Ballistic : public Kalman::Apps::Model
    {
        private:
            std::size_t fOrder;
            double fDragOverMass = 0.;
            double fG = 9.81;
            double fGM = 0.;
            bool fSurfaceGravity = true;
            double fGravConstant = 6.674e-11, fCentralMass = 5.972e24;

            /// Gravity and drag at position p and velocity v (3 each), written to a.
            void Acceleration(const double *p, const double *v, double *a) const;

        public:
            /// \param order 3 = position, velocity, acceleration per axis; 2 = position, velocity.
            explicit Ballistic(std::size_t order = 3);

            const char *Title() const override { return "Ballistic Extended Model"; }
            const char *Description() const override
            { return "Nonlinear 3D ballistic model with gravity and drag."; }

            std::size_t Order() const { return fOrder; }

            /// The continuous dynamics: dp/dt = v, dv/dt = a(p, v), da/dt = 0.
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            /// One sample: Derivative() integrated, then the acceleration slot refreshed.
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_BALLISTICS_BALLISTIC_H
