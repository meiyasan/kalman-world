/**
 * *********************************************
 *
 * \file SensorFactory.h
 * \brief Sensors a library offers by name.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details A Kalman::Sensor is built in C++ from a function or a matrix. A
 * program that holds no C++ of its own (the Python package, the servers) can
 * only hand over a matrix, or a function to be called back at every step. A
 * library that ships sensors written in C++ registers each under a name here,
 * and such a program then asks for one by that name and a list of numbers:
 *
 * \code{.cpp}
 * Kalman::SensorFactory::Instance().Register("doppler",
 *     [](const std::string &name, std::size_t n, const std::vector<double> &settings)
 *     { return MySensor(name, n, settings); });
 *
 * Kalman::Sensor radar = Kalman::SensorFactory::Instance().Create("doppler", "radar", 9, settings);
 * \endcode
 *
 * What the numbers mean is the sensor's business; it checks them and says what
 * is wrong. The library registers its own generic sensors (Kalman::Sensors:
 * `position`, `doppler`, `camera`); a plugin adds what it ships.
 *
 * *********************************************
 */

#ifndef KALMAN_SENSOR_FACTORY_H
#define KALMAN_SENSOR_FACTORY_H

#include "Kalman/Sensor.h"

#include <cstddef>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace Kalman
{
    class SensorFactory
    {
        public:
            /// Builds the sensor \p name for a state of \p n components from \p settings.
            using Creator = std::function<Sensor(const std::string &name, std::size_t n,
                                                 const std::vector<double> &settings)>;

        private:
            std::map<std::string, Creator> fKinds;
            mutable std::mutex fMutex;

            SensorFactory() = default;

        public:
            static SensorFactory &Instance();

            /// Register a creator under \p kind; false (and nothing changed) when the kind exists.
            /// \throws std::invalid_argument on an empty kind or a null creator.
            bool Register(const std::string &kind, Creator create);

            bool Has(const std::string &kind) const;

            /// The registered kinds, sorted.
            std::vector<std::string> Kinds() const;

            /// \throws std::invalid_argument for a kind that is not registered; whatever the
            ///         sensor throws for settings it does not accept.
            Sensor Create(const std::string &kind, const std::string &name, std::size_t n,
                          const std::vector<double> &settings) const;
    };
}

#endif // KALMAN_SENSOR_FACTORY_H
