/**
 * *********************************************
 *
 * \file Operations.h
 * \brief The model operations as functions: a model made from another.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Each returns a model that any Kalman::Filter runs, and they nest:
 *
 * \code{.cpp}
 * auto shell = std::make_shared<MyShell>();
 * auto radar = Kalman::AddSensor(shell, Kalman::Sensor("doppler", doppler, R));   // one more sensor
 * auto model = Kalman::Joint(radar, {{"c_drag", 1e-3, 1e-6}});                    // and a parameter estimated
 * Kalman::Filters::EKF ekf(model.get());
 * \endcode
 *
 * These share the models they are given. The methods of the same names on
 * Kalman::Model (`model.AddSensor(sensor)`, `model.Joint({...})`) borrow the
 * model instead, for one that lives on the stack; it must then outlive the
 * result. Python has both forms too: `model.add_sensor(sensor)` and
 * `add_sensor(model, sensor)`.
 *
 * *********************************************
 */

#ifndef KALMAN_OPERATIONS_H
#define KALMAN_OPERATIONS_H

#include "Kalman/Models/Fusion.h"
#include "Kalman/Models/Joint.h"
#include "Kalman/Sensor.h"

#include <memory>
#include <vector>

namespace Kalman
{
    /// \p model with one more sensor on the same state (Models::Fusion).
    std::shared_ptr<Models::Fusion> AddSensor(std::shared_ptr<Model> model, Sensor sensor);

    /// \p model with some of its declared parameters estimated along with its state (Models::Joint).
    std::shared_ptr<Models::Joint> Joint(std::shared_ptr<Model> model, std::vector<EstimatedParameter> parameters);
}

#endif // KALMAN_OPERATIONS_H
