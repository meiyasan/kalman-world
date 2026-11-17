/**
 * *********************************************
 *
 * \file SensorFactory.cc
 * \brief Implementation of the Kalman::SensorFactory class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/SensorFactory.h"

#include <stdexcept>

namespace Kalman
{
    SensorFactory &SensorFactory::Instance()
    {
        static SensorFactory instance;   // built on first use, before any registration
        return instance;
    }

    bool SensorFactory::Register(const std::string &kind, Creator create)
    {
        if (kind.empty()) throw std::invalid_argument("SensorFactory::Register: the kind is empty.");
        if (!create) throw std::invalid_argument("SensorFactory::Register: no creator for '" + kind + "'.");
        std::lock_guard<std::mutex> lock(fMutex);
        return fKinds.emplace(kind, std::move(create)).second;
    }

    bool SensorFactory::Has(const std::string &kind) const
    {
        std::lock_guard<std::mutex> lock(fMutex);
        return fKinds.count(kind) > 0;
    }

    std::vector<std::string> SensorFactory::Kinds() const
    {
        std::lock_guard<std::mutex> lock(fMutex);
        std::vector<std::string> kinds;
        for (const auto &entry : fKinds) kinds.push_back(entry.first);
        return kinds;
    }

    Sensor SensorFactory::Create(const std::string &kind, const std::string &name, std::size_t n,
                                 const std::vector<double> &settings) const
    {
        Creator create;
        {
            std::lock_guard<std::mutex> lock(fMutex);
            const auto it = fKinds.find(kind);
            if (it == fKinds.end())
            {
                std::string known;
                for (const auto &entry : fKinds) known += (known.empty() ? "" : ", ") + entry.first;
                throw std::invalid_argument("SensorFactory::Create: no sensor of kind '" + kind + "' (registered: "
                                            + (known.empty() ? "none" : known) + ").");
            }
            create = it->second;
        }
        return create(name, n, settings);
    }
}
