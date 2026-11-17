/**
 * *********************************************
 *
 * \file Sensor.h
 * \brief Header of the Kalman::Sensor structure
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \struct Kalman::Sensor
 * \brief What acquires a measurement: what it reads of the state, and how well.
 *
 * \details A sensor reads `k` channels of the state,
 *
 * \code{.unparsed}
 *     z_sensor = g(x, t) + v,      v ~ N(0, R)
 * \endcode
 *
 * given either as a function `g` (with its Jacobian, or differenced when none
 * is given) or as a matrix `G`, for `g(x) = G x`. Its channels are named: each
 * is an *observable*, a quantity that can be measured, with its unit. A sensor
 * may report less often than the model steps: its `rate`. On the steps between
 * two of its reports its channels are simply missing (NaN), which every engine
 * skips, so sensors of different rates need no special filter.
 *
 * A model lists its sensors (Model::Sensors()): the one it was built with, made
 * of its own `h` and `R`, then those added with Model::AddSensor(). Several
 * sensors on one state is what the filter fuses (Models::Fusion).
 *
 * The words, used the same way in C++ and in Python: an **observable** is a
 * quantity that can be measured; a **sensor** measures one or more of them; a
 * **measurement** is the numbers that come out.
 *
 * *********************************************
 */

#ifndef KALMAN_SENSOR_H
#define KALMAN_SENSOR_H

#include "Kalman/Matrix.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace Kalman
{
    /// One component of a state or of a measurement: what it is and its unit.
    struct Component
    {
        std::string name;
        std::string unit;   ///< "" when dimensionless or not stated
    };

    struct Sensor
    {
        /// The readings `g(x, t)`, a vector of length k.
        using Measure = std::function<Vector(const Vector &x, double t)>;
        /// `dg/dx` at (x, t), k x n.
        using MeasureJacobian = std::function<Matrix(const Vector &x, double t)>;

        std::string name;                   ///< what the sensor is called
        Measure g;                          ///< the readings; empty when they are a matrix
        Matrix G;                           ///< (k,n) when the readings are `G x`
        Matrix R;                           ///< (k,k) noise covariance
        MeasureJacobian jacobian;           ///< `dg/dx`; differenced when empty
        std::vector<Component> channels;    ///< its observables; named after the sensor when empty
        double rate = 0.;                   ///< how often it reports, per unit of the model's time; 0: at every step
        double delay = 0.;                  ///< how late it reports, in the model's unit of time: it reads the
                                            ///< state as it was `delay` ago (needs a lagged model, Models::Lag)

        double station = 0.;                ///< for a sensor of a sequential instrument (a detector's layer): where
                                            ///< it sits along the sequence, the value handed to `f` as its time

        // What a sensor gets from the instrument that carries it (Kalman::Instrument).
        std::string instrument;             ///< the instrument's name; empty for a sensor on its own
        bool calibrated = false;            ///< whether its reading goes through a gain and an offset
        double gain = 1.;                   ///< z = gain * g(x, t) + offset
        double offset = 0.;

        /// The numbers its readings depend on, by name (a camera's focal length, where a
        /// radar is). The model that takes the sensor declares each as its parameter
        /// `<owner>.<name>`, which Kalman::Joint can then estimate. `g` and `jacobian`
        /// read them through this pointer, so the copies of a sensor share them.
        using Settings = std::vector<std::pair<std::string, double>>;
        std::shared_ptr<Settings> settings;

        Sensor() = default;
        /// A sensor that reads `g(x, t)`.
        Sensor(std::string name, Measure g, const Matrix &R, MeasureJacobian jacobian = nullptr,
               std::vector<Component> channels = {});
        /// A sensor that reads `G x`.
        Sensor(std::string name, const Matrix &G, const Matrix &R, std::vector<Component> channels = {});

        /// The number of channels, k.
        std::size_t Channels() const { return R.Rows(); }
        /// True when the readings are a matrix.
        bool IsLinear() const { return !g; }

        /// Its observables: the given ones, or the sensor's name (numbered when k > 1).
        std::vector<Component> Observables() const;

        /// Its readings of the state \p x: `g(x, t)` or `G x`, through its calibration.
        /// \throws std::runtime_error when `g` returns another number of readings.
        Vector Read(const Vector &x, double t = 0.) const;

        /// The names of its settings, in the order they were declared.
        std::vector<std::string> SettingNames() const;
        bool HasSetting(const std::string &setting) const;
        /// \throws std::out_of_range for a setting it does not have.
        double Setting(const std::string &setting) const;
        void Set(const std::string &setting, double value);

        /// The name its calibration parameters carry: the instrument's, or its own.
        const std::string &Owner() const { return instrument.empty() ? name : instrument; }

        /**
         * \brief Check the sensor against a state of \p n components.
         * \throws std::invalid_argument for no name, no channel, both or neither of
         *         `g` and `G`, an `R` that is not square, a `G` that is not k x n,
         *         a number of channel names other than k, or a negative rate.
         */
        void Check(std::size_t n) const;
    };
}

#endif // KALMAN_SENSOR_H
