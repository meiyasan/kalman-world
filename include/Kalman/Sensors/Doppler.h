/**
 * *********************************************
 *
 * \file Doppler.h
 * \brief A sensor that reads the velocity along the line of sight.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Generic, as every sensor of Kalman::Sensors (see Position.h).
 *
 * Registered with Kalman::SensorFactory as `doppler`; its numbers are d, the d
 * position indices, the d velocity indices, sigma, then where the sensor is
 * (d numbers) or nothing.
 *
 * *********************************************
 */

#ifndef KALMAN_SENSORS_DOPPLER_H
#define KALMAN_SENSORS_DOPPLER_H

#include "Kalman/Sensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Kalman::Sensors
{
    /**
     * \brief The velocity along the line of sight, one channel:
     *
     *     v_r = v . (p - at) / |p - at|
     *
     * \param name              what the sensor is called.
     * \param n                 the number of state components of the model it will read.
     * \param position,velocity the indices of the position and of the velocity in the
     *                          state, as many of each (1, 2 or 3 dimensions).
     * \param sigma             the standard deviation of the reading.
     * \param at                where the sensor is; the origin when empty.
     *
     * Where it is are its settings `x`, `y`, `z` (as many as dimensions): the model
     * that takes it has the parameters `<name>.x`, ..., which Kalman::Joint can
     * estimate. At the sensor itself the line of sight is not defined: the reading is 0.
     */
    class Doppler : public Kalman::Sensor
    {
        public:
            Doppler(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                    const std::vector<std::size_t> &velocity, double sigma, const Vector &at = Vector());
    };
}

#endif // KALMAN_SENSORS_DOPPLER_H
