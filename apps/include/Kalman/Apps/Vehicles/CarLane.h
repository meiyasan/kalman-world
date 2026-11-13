/**
 * *********************************************
 *
 * \file CarLane.h
 * \brief Header of the Kalman::Apps::Vehicles::CarLane class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Vehicles::CarLane
 * \brief Lateral lane keeping with steering control - the twin of
 * `kalman.apps.vehicles.CarLaneModel`.
 *
 * \details State `[y, psi, v]`, lateral offset, heading and speed; control
 * `u = [delta]`, the steering angle; the lateral offset is observed. Bicycle
 * kinematics
 *
 * \code{.unparsed}
 *   y'   = v sin(psi)
 *   psi' = v tan(delta) / L      delta clipped to [-steer_limit, steer_limit]
 *   v'   = 0
 * \endcode
 *
 * integrated with explicit Euler steps by default, as the Python class does.
 *
 * | Parameter     | Meaning                                         |
 * |---------------|-------------------------------------------------|
 * | `wheelbase`   | distance between the axles `L` [m]              |
 * | `steer_limit` | largest steering angle [rad]                    |
 * | `substeps`    | integrator steps per sample                     |
 * | `integrator`  | IntegratorKind: 0 Euler, 1 RK4, 2 Dormand-Prince |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_VEHICLES_CARLANE_H
#define KALMAN_APPS_VEHICLES_CARLANE_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Vehicles
{
    class CarLane : public Kalman::Apps::Model
    {
        private:
            double fWheelbase = 2.7, fSteerLimit = 0.1;

        public:
            CarLane();

            const char *Title() const override { return "Car Lane Keeping"; }
            const char *Description() const override
            { return "Bicycle-kinematic lateral lane-keeping with steering control."; }

            /// The steering angle actually applied: the command clipped to the actuator limit.
            double Steering(const Vector &u) const;

            /// The continuous dynamics: [y, psi, v]' = [v sin psi, v tan delta / L, 0].
            Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_VEHICLES_CARLANE_H
