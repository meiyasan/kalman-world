/**
 * *********************************************
 *
 * \file Sensors.h
 * \brief Generic sensors: a position reading, a Doppler radar, a camera.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Kalman::Sensor is the contract; these fill it in for three readings
 * the applications kept writing by hand. Each says which components of the
 * state it reads by their indices, so it fits any model that has them:
 *
 * \code{.cpp}
 * using namespace Kalman::Apps::Fusion;
 * // a shell with state (px, vx, ax, py, vy, ay, pz, vz, az)
 * auto tracked = Kalman::AddSensor(shell, PositionSensor("gps", 9, {0, 3, 6}, 10.));
 * auto radar   = Kalman::AddInstrument(tracked, DopplerRadar("doppler", 9, {0, 3, 6}, {1, 4, 7}, 1.));
 * auto seen    = Kalman::AddInstrument(radar, Camera("camera", 9, {0, 3, 6}, 800., 1.5));
 * \endcode
 *
 * They are generic: a geometry and a noise, nothing of any product. A given
 * piece of hardware is an adapter, a class deriving from one of these that
 * fixes what its data sheet and its mounting say. None is shipped here.
 *
 * The radar and the camera are instruments (Kalman::Instrument) of one sensor,
 * so each can carry a calibration: `DopplerRadar(...).Calibrate(1., 0.)` gives
 * the model the parameters `doppler.gain` and `doppler.offset`, which
 * Kalman::Joint can estimate.
 *
 * Each is also registered with Kalman::SensorFactory, for a program that holds
 * no C++ (`modules/apps/fusion/sensors.py` asks for them this way). The settings, in order:
 *
 * | kind       | settings |
 * |------------|----------|
 * | `position` | k, the k axes, sigma, then the k x k rotation (row-major) or nothing |
 * | `doppler`  | d, the d position indices, the d velocity indices, sigma, then where the radar is (d numbers) or nothing |
 * | `camera`   | the 3 position indices, focal, sigma, then where the camera is (3) and its rotation (9, row-major), or nothing |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_FUSION_SENSORS_H
#define KALMAN_APPS_FUSION_SENSORS_H

#include "Kalman/Instrument.h"
#include "Kalman/Sensor.h"

#include <cstddef>
#include <string>
#include <vector>

namespace Kalman::Apps::Fusion
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
    class PositionSensor : public Kalman::Sensor
    {
        public:
            PositionSensor(std::string name, std::size_t n, const std::vector<std::size_t> &axes, double sigma,
                           const Matrix &rotation = Matrix());
    };

    /**
     * \brief A Doppler radar: the velocity along the line of sight, one channel.
     *
     *     v_r = v . (p - at) / |p - at|
     *
     * \param position,velocity the indices of the position and of the velocity in the
     *                          state, as many of each (1, 2 or 3 dimensions).
     * \param sigma             the standard deviation of the reading.
     * \param at                where the radar is; the origin when empty.
     *
     * At the radar itself the line of sight is not defined: the reading is 0.
     */
    class DopplerRadar : public Kalman::Instrument
    {
        public:
            DopplerRadar(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                         const std::vector<std::size_t> &velocity, double sigma, const Vector &at = Vector());

            /// The sensor on its own, with its Jacobian.
            static Sensor Reading(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                                  const std::vector<std::size_t> &velocity, double sigma, const Vector &at = Vector());
    };

    /**
     * \brief A pinhole camera: two pixel coordinates of a point in space.
     *
     *     q = rotation * (p - at)        u = focal * q_x / q_z        v = focal * q_y / q_z
     *
     * \param position the indices of the point's three coordinates in the state.
     * \param focal    the focal length, in pixels.
     * \param sigma    the standard deviation of each pixel coordinate.
     * \param at       where the camera is; the origin when empty.
     * \param rotation 3 x 3, world to camera: its rows are the camera's axes, the third
     *                 the direction it looks in. Empty: the identity.
     *
     * A point at or behind the camera has no image; its depth is then taken as
     * Camera::NEAR, so the reading stays finite.
     */
    class Camera : public Kalman::Instrument
    {
        public:
            static constexpr double NEAR = 1e-6;   ///< the smallest depth a point is given

            Camera(std::string name, std::size_t n, const std::vector<std::size_t> &position, double focal,
                   double sigma, const Vector &at = Vector(), const Matrix &rotation = Matrix());

            /// The sensor on its own, with its Jacobian.
            static Sensor Reading(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                                  double focal, double sigma, const Vector &at = Vector(),
                                  const Matrix &rotation = Matrix());
    };
}

#endif // KALMAN_APPS_FUSION_SENSORS_H
