/**
 * *********************************************
 *
 * \file Adapters.h
 * \brief Umbrella header of the adapters: specific hardware.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details The library ships generic sensors (`Kalman/Sensors.h`): a geometry and
 * a noise, nothing of any product. An adapter is one piece of hardware: a class
 * deriving from Kalman::Instrument, made of generic sensors, with what its data
 * sheet and its mounting say. It is the counterpart of one class of
 * `kalman.adapters`.
 *
 * One header per device, `Kalman/Adapters/<Device>.h`, in the namespace
 * Kalman::Adapters, each included here. None is shipped yet.
 *
 * \code{.cpp}
 * #include <Kalman/Kalman.h>
 *
 * namespace Kalman::Adapters
 * {
 *     /// The radar of one test bench.
 *     struct BenchRadar : Kalman::Instrument
 *     {
 *         BenchRadar(std::size_t n, const std::vector<std::size_t> &p, const std::vector<std::size_t> &v)
 *             : Kalman::Instrument("bench-radar", {Kalman::Sensors::Doppler("doppler", n, p, v, 0.15)}, 10.)
 *         { Calibrate(1., 0.); }
 *     };
 * }
 * \endcode
 *
 * *********************************************
 */

#ifndef KALMAN_ADAPTERS_H
#define KALMAN_ADAPTERS_H

#include "Kalman/Instrument.h"
#include "Kalman/Sensors.h"

#endif // KALMAN_ADAPTERS_H
