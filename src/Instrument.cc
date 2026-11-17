/**
 * *********************************************
 *
 * \file Instrument.cc
 * \brief Source of the Kalman::Instrument structure
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Instrument.h"

#include <set>
#include <stdexcept>
#include <utility>

namespace Kalman
{
    Instrument::Instrument(std::string name_, std::vector<Sensor> sensors_, double rate_)
        : name(std::move(name_)), sensors(std::move(sensors_)), rate(rate_) {}

    Instrument &Instrument::Calibrate(double gain_, double offset_)
    {
        calibrated = true;
        gain = gain_;
        offset = offset_;
        return *this;
    }

    std::vector<Sensor> Instrument::Sensors() const
    {
        std::vector<Sensor> out = sensors;
        for (Sensor &s : out)
        {
            s.instrument = name;
            if (s.rate == 0.) s.rate = rate;
            if (calibrated)
            {
                s.calibrated = true;
                s.gain = gain;
                s.offset = offset;
            }
        }
        return out;
    }

    std::size_t Instrument::Channels() const
    {
        if (sequential) return sensors.empty() ? 0 : sensors.front().Channels();
        std::size_t k = 0;
        for (const Sensor &s : sensors) k += s.Channels();
        return k;
    }

    void Instrument::Check(std::size_t n) const
    {
        if (name.empty()) throw std::invalid_argument("Kalman::Instrument: an instrument needs a name.");
        if (sensors.empty()) throw std::invalid_argument("Kalman::Instrument: '" + name + "' has no sensor.");
        if (!(rate >= 0.)) throw std::invalid_argument("Kalman::Instrument: '" + name + "' has a negative rate.");
        std::set<std::string> seen;
        for (const Sensor &s : sensors)
        {
            s.Check(n);
            if (!seen.insert(s.name).second)
                throw std::invalid_argument("Kalman::Instrument: '" + name + "' has two sensors called '" + s.name + "'.");
        }
    }

    Instrument Compose(std::vector<Sensor> sensors, std::string name, double rate)
    {
        if (name.empty())
            for (const Sensor &s : sensors) name += (name.empty() ? "" : "+") + s.name;
        return Instrument(std::move(name), std::move(sensors), rate);
    }

    Instrument operator|(const Sensor &a, const Sensor &b) { return Compose({a, b}); }

    Instrument operator|(Instrument instrument, const Sensor &sensor)
    {
        instrument.sensors.push_back(sensor);
        return instrument;
    }
}
