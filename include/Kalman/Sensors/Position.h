/**
 * *********************************************
 *
 * \file Position.h
 * \brief A sensor that reads some components of the state.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details The sensors of Kalman::Sensors are generic: each knows a geometry
 * and a noise, and nothing of any product. It is told which components of the
 * state it reads by their indices, so it fits any model that has them:
 *
 * \code{.cpp}
 * // a shell with state (px, vx, ax, py, vy, ay, pz, vz, az)
 * auto tracked = Kalman::AddSensor(shell, Kalman::Sensors::Position("gps", 9, {0, 3, 6}, 10.));
 * \endcode
 *
 * A given piece of hardware is an adapter: a Kalman::Instrument made of these
 * with what its data sheet and its mounting say. None is part of the library.
 *
 * Registered with Kalman::SensorFactory as `position`; its numbers are k, the
 * k axes, sigma, then the k x k rotation (row-major) or nothing.
 *
 * *********************************************
 */

#ifndef KALMAN_SENSORS_POSITION_H
#define KALMAN_SENSORS_POSITION_H

#include "Kalman/Sensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Kalman::Sensors
{
    /**
     * \brief Reads some components of the state: `z = rotation * x[axes]`.
     *
     * \param name     what the sensor is called.
     * \param n        the number of state components of the model it will read.
     * \param axes     the indices of the components it reads.
     * \param sigma    the standard deviation of its noise, the same on every channel
     *                 and given in the sensor's own axes.
     * \param rotation k x k, for a sensor not aligned with the state's axes: its rows
     *                 are the sensor's axes. Empty: aligned.
     *
     * Its readings are a matrix, so a model of matrices that takes it stays one.
     */
    class Position : public Kalman::Sensor
    {
        public:
            Position(std::string name, std::size_t n, const std::vector<std::size_t> &axes, double sigma,
                     const Matrix &rotation = Matrix());
    };
}

#endif // KALMAN_SENSORS_POSITION_H
