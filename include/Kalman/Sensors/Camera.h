/**
 * *********************************************
 *
 * \file Camera.h
 * \brief A sensor that reads the two pixel coordinates of a point in space.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Generic, as every sensor of Kalman::Sensors (see Position.h).
 *
 * Registered with Kalman::SensorFactory as `camera`; its numbers are the 3
 * position indices, focal, sigma, then where the camera is (3) and its rotation
 * (9, row-major), or nothing.
 *
 * *********************************************
 */

#ifndef KALMAN_SENSORS_CAMERA_H
#define KALMAN_SENSORS_CAMERA_H

#include "Kalman/Sensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Kalman::Sensors
{
    /**
     * \brief A pinhole projection, two channels:
     *
     *     q = rotation * (p - at)        u = focal * q_x / q_z        v = focal * q_y / q_z
     *
     * \param name     what the sensor is called.
     * \param n        the number of state components of the model it will read.
     * \param position the indices of the point's three coordinates in the state.
     * \param focal    the focal length, in pixels.
     * \param sigma    the standard deviation of each pixel coordinate.
     * \param at       where the camera is; the origin when empty.
     * \param rotation 3 x 3, world to camera: its rows are the camera's axes, the third
     *                 the direction it looks in. Empty: the identity.
     *
     * The focal length and where the camera is are its settings `focal`, `x`, `y`,
     * `z`: the model that takes it has the parameters `<name>.focal`, ..., which
     * Kalman::Joint can estimate. A point at or behind the camera has no image; its
     * depth is then taken as Camera::NEAR, so the reading stays finite.
     */
    class Camera : public Kalman::Sensor
    {
        public:
            static constexpr double NEAR = 1e-6;   ///< the smallest depth a point is given

            Camera(std::string name, std::size_t n, const std::vector<std::size_t> &position, double focal,
                   double sigma, const Vector &at = Vector(), const Matrix &rotation = Matrix());
    };
}

#endif // KALMAN_SENSORS_CAMERA_H
