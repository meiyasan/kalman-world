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
#include "Kalman/Models/Lag.h"
#include "Kalman/Models/Composite.h"
#include "Kalman/Models/Router.h"
#include "Kalman/Sensor.h"
#include "Kalman/Instrument.h"

#include <memory>
#include <vector>

namespace Kalman
{
    /// \p model with one more sensor on the same state (Models::Fusion).
    std::shared_ptr<Models::Fusion> AddSensor(std::shared_ptr<Model> model, Sensor sensor);

    /// \p parts side by side, each with its own state (Models::Composite). The verb is the
    /// one that makes an instrument of sensors.
    std::shared_ptr<Models::Composite> Compose(std::vector<std::shared_ptr<Model>> parts,
                                               Readout readout = Readout::Stacked, std::vector<std::string> names = {});

    /// Several models of one system with a probability on each (Models::Router), for a
    /// system that changes regime: \p Pi is the Markov transition between them. Run by the
    /// interacting engine (the IMM of the literature).
    std::shared_ptr<Models::Router> Route(std::vector<std::shared_ptr<Model>> modes, const Matrix &Pi,
                                          const Vector &mu0 = Vector(), std::vector<std::string> names = {});

    /// \p model carrying its last \p depth states: for sensors that report late, and
    /// fixed-lag smoothing.
    std::shared_ptr<Models::Lag> Lag(std::shared_ptr<Model> model, std::size_t depth);

    /// \p model read by every sensor of \p instrument (a chain of Models::Fusion).
    std::shared_ptr<Model> AddInstrument(std::shared_ptr<Model> model, const Instrument &instrument);

    /// \p model with some of its declared parameters estimated along with its state (Models::Joint).
    std::shared_ptr<Models::Joint> Joint(std::shared_ptr<Model> model, std::vector<EstimatedParameter> parameters);
}

#endif // KALMAN_OPERATIONS_H
