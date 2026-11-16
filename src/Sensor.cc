/**
 * *********************************************
 *
 * \file Sensor.cc
 * \brief Source of the Kalman::Sensor structure
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Sensor.h"

#include <stdexcept>
#include <utility>

namespace Kalman
{
    Sensor::Sensor(std::string name_, Measure g_, const Matrix &R_, MeasureJacobian jacobian_,
                   std::vector<Component> channels_)
        : name(std::move(name_)), g(std::move(g_)), R(R_), jacobian(std::move(jacobian_)), channels(std::move(channels_))
    {
        if (!g) throw std::invalid_argument("Kalman::Sensor: the sensor '" + name + "' needs a function g or a matrix G.");
    }

    Sensor::Sensor(std::string name_, const Matrix &G_, const Matrix &R_, std::vector<Component> channels_)
        : name(std::move(name_)), G(G_), R(R_), channels(std::move(channels_)) {}

    std::vector<Component> Sensor::Observables() const
    {
        if (!channels.empty()) return channels;
        const std::size_t k = Channels();
        std::vector<Component> out(k);
        for (std::size_t i = 0; i < k; ++i) out[i].name = k == 1 ? name : name + std::to_string(i);
        return out;
    }

    void Sensor::Check(std::size_t n) const
    {
        if (name.empty()) throw std::invalid_argument("Kalman::Sensor: a sensor needs a name.");
        const std::string who = "Kalman::Sensor: the sensor '" + name + "' ";
        const std::size_t k = Channels();
        if (k == 0) throw std::invalid_argument(who + "has no channel: give its noise covariance R (k x k).");
        if (R.Cols() != k) throw std::invalid_argument(who + "needs a square R.");
        if (static_cast<bool>(g) == !G.Empty())
            throw std::invalid_argument(who + "needs a function g or a matrix G, one of them.");
        if (!g && (G.Rows() != k || G.Cols() != n))
            throw std::invalid_argument(who + "has a G of " + std::to_string(G.Rows()) + " x " + std::to_string(G.Cols())
                                        + ", expected " + std::to_string(k) + " x " + std::to_string(n) + ".");
        if (!channels.empty() && channels.size() != k)
            throw std::invalid_argument(who + "has " + std::to_string(k) + " channels and "
                                        + std::to_string(channels.size()) + " names.");
        if (!(rate >= 0.)) throw std::invalid_argument(who + "has a negative rate.");
        for (const auto &c : channels)
            if (c.name.empty()) throw std::invalid_argument(who + "has a channel with no name.");
    }
}
