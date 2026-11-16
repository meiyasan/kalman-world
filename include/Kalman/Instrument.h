/**
 * *********************************************
 *
 * \file Instrument.h
 * \brief Header of the Kalman::Instrument structure
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \struct Kalman::Instrument
 * \brief A set of sensors, and what they share.
 *
 * \details A sensor is one reading of the state. An instrument is the piece of
 * hardware that carries one or more of them: a weather station (temperature,
 * CO2 and humidity on one board), a receiver, a camera. What its sensors share
 * is the instrument's:
 *
 * - a **name**, under which its sensors are grouped (Model::Instruments());
 * - a **rate**, for the sensors that state none of their own;
 * - a **calibration**: every reading goes through one gain and one offset,
 *
 * \code{.unparsed}
 *     z = gain * g(x, t) + offset
 * \endcode
 *
 * which become two named parameters of the model that carries the instrument,
 * `<name>.gain` and `<name>.offset`. Being parameters, they can be estimated
 * along with the state (Models::Joint): that is calibrating the instrument
 * while it measures. Whether the data can tell an offset from what is measured
 * is what Kalman::Observability answers; a second instrument, already
 * calibrated, usually settles it.
 *
 * An instrument is made by composing sensors, the way a composite model is
 * made by composing models: `Kalman::Compose({s1, s2}, "station")`, or
 * `s1 | s2`. Model::AddInstrument() then gives the model read by all of them.
 *
 * *********************************************
 */

#ifndef KALMAN_INSTRUMENT_H
#define KALMAN_INSTRUMENT_H

#include "Kalman/Sensor.h"

#include <string>
#include <vector>

namespace Kalman
{
    struct Instrument
    {
        std::string name;               ///< what the instrument is called
        std::vector<Sensor> sensors;    ///< its sensors, in the order of their channels
        double rate = 0.;               ///< for its sensors that state none; 0: at every step
        bool calibrated = false;        ///< whether its readings go through a gain and an offset
        double gain = 1.;
        double offset = 0.;

        Instrument() = default;
        Instrument(std::string name, std::vector<Sensor> sensors, double rate = 0.);

        /// Give the instrument a calibration to carry: `z = gain * g + offset`, as the
        /// parameters `<name>.gain` and `<name>.offset` of the model that takes it.
        Instrument &Calibrate(double gain = 1., double offset = 0.);

        /// Its sensors as a model takes them: each stamped with the instrument's name,
        /// its rate when the sensor states none, and its calibration.
        std::vector<Sensor> Sensors() const;

        /// The total number of channels.
        std::size_t Channels() const;

        /// \throws std::invalid_argument for no name, no sensor, two sensors of one name,
        ///         a negative rate, or a sensor that does not fit a state of \p n.
        void Check(std::size_t n) const;
    };

    /// An instrument made of \p sensors (the verb is the one that makes a composite of models).
    Instrument Compose(std::vector<Sensor> sensors, std::string name = "", double rate = 0.);

    /// Two sensors make an instrument, named after both.
    Instrument operator|(const Sensor &a, const Sensor &b);
    /// One more sensor on an instrument.
    Instrument operator|(Instrument instrument, const Sensor &sensor);
}

#endif // KALMAN_INSTRUMENT_H
