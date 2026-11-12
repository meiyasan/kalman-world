/**
 * *********************************************
 *
 * \file Factory.cc
 * \brief Implementation of the Kalman::Factory class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Factory.h"

#include <algorithm>
#include <dlfcn.h>
#include <sstream>
#include <stdexcept>

namespace Kalman
{
    Factory &Factory::Instance()
    {
        // A function-local static: constructed on first use, which is before any
        // KALMAN_REGISTER_MODEL initialiser can run — those call Instance() too.
        static Factory instance;
        return instance;
    }

    bool Factory::Register(const std::string &name, Creator create)
    {
        if (name.empty()) throw std::invalid_argument("Factory::Register: the model name is empty.");
        if (!create) throw std::invalid_argument("Factory::Register: no creator for '" + name + "'.");

        std::lock_guard<std::mutex> lock(fMutex);
        const auto it = fModels.find(name);
        if (it != fModels.end())
        {
            const std::string what = "Factory::Register: '" + name + "' is already registered"
                                     + (it->second.plugin.empty() ? "" : " (by " + it->second.plugin + ")")
                                     + "; model names are unique.";
            // Inside a dlopen the registrar is a static initialiser, and an exception
            // escaping one of those terminates the process. Park the message instead
            // and let LoadPlugin() throw it once the loader has returned.
            if (!fLoading.empty()) { fLoadError = what; return false; }
            throw std::runtime_error(what);
        }
        fModels[name] = Registration{std::move(create), fLoading};
        return true;
    }

    bool Factory::Unregister(const std::string &name)
    {
        std::lock_guard<std::mutex> lock(fMutex);
        return fModels.erase(name) > 0;
    }

    bool Factory::Has(const std::string &name) const
    {
        std::lock_guard<std::mutex> lock(fMutex);
        return fModels.count(name) > 0;
    }

    std::vector<std::string> Factory::Names() const
    {
        std::lock_guard<std::mutex> lock(fMutex);
        std::vector<std::string> names;
        names.reserve(fModels.size());
        for (const auto &kv : fModels) names.push_back(kv.first);   // std::map: already sorted
        return names;
    }

    std::size_t Factory::Size() const
    {
        std::lock_guard<std::mutex> lock(fMutex);
        return fModels.size();
    }

    std::unique_ptr<Model> Factory::Create(const std::string &name) const
    {
        Creator create;
        {
            std::lock_guard<std::mutex> lock(fMutex);
            const auto it = fModels.find(name);
            if (it == fModels.end())
            {
                std::ostringstream os;
                os << "Factory::Create: unknown model '" << name << "'; registered:";
                if (fModels.empty()) os << " (none)";
                for (const auto &kv : fModels) os << " '" << kv.first << "'";
                os << ".";
                throw std::out_of_range(os.str());
            }
            create = it->second.create;
        }
        // Run the creator outside the lock: a constructor may itself consult the factory.
        std::unique_ptr<Model> model = create();
        if (!model) throw std::runtime_error("Factory::Create: the creator for '" + name + "' returned null.");
        return model;
    }

    Factory::Entry Factory::Describe(const std::string &name) const
    {
        const std::unique_ptr<Model> model = Create(name);

        Entry e;
        e.name = name;
        e.title = model->Title();
        e.description = model->Description();
        e.version = model->Version();
        e.n = model->NState();
        e.m = model->NMeasure();
        e.l = model->NControl();
        e.fs = model->SampleRate();
        e.engine = model->DefaultEngine();
        e.analytic = model->HasAnalyticJacobian();
        {
            std::lock_guard<std::mutex> lock(fMutex);
            e.plugin = fModels.at(name).plugin;
        }
        return e;
    }

    std::vector<std::string> Factory::LoadPlugin(const std::string &path)
    {
        if (path.empty()) throw std::invalid_argument("Factory::LoadPlugin: the path is empty.");

        {
            std::lock_guard<std::mutex> lock(fMutex);
            if (std::find(fPlugins.begin(), fPlugins.end(), path) != fPlugins.end())
            {
                std::vector<std::string> names;
                for (const auto &kv : fModels)
                    if (kv.second.plugin == path) names.push_back(kv.first);
                return names;
            }
            if (!fLoading.empty())
                throw std::runtime_error("Factory::LoadPlugin: '" + path + "' requested while '"
                                         + fLoading + "' is still loading.");
            fLoading = path;
            fLoadError.clear();
        }

        // RTLD_NOW so an unresolved symbol fails here, with a message, rather than
        // at the first call into the model. RTLD_LOCAL keeps two plugins that
        // both define an internal helper from clashing with each other.
        void *handle = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);

        std::lock_guard<std::mutex> lock(fMutex);
        fLoading.clear();

        if (!handle)
        {
            const char *why = ::dlerror();
            throw std::runtime_error("Factory::LoadPlugin: cannot load '" + path + "': "
                                     + (why ? why : "unknown loader error") + ".");
        }

        std::vector<std::string> names;
        for (const auto &kv : fModels)
            if (kv.second.plugin == path) names.push_back(kv.first);

        if (!fLoadError.empty())
        {
            // The library is in memory (dlopen has no undo for its initialisers), but
            // it is not recorded as loaded and whatever it did register is withdrawn,
            // so the failure is complete rather than half-applied.
            for (const auto &name : names) fModels.erase(name);
            const std::string what = fLoadError;
            fLoadError.clear();
            throw std::runtime_error("Factory::LoadPlugin: '" + path + "' rejected — " + what);
        }

        if (names.empty())
            throw std::runtime_error("Factory::LoadPlugin: '" + path + "' loaded but registered no model; "
                                     "did it forget KALMAN_REGISTER_MODEL, or was it linked against "
                                     "a different libKalman?");

        fPlugins.push_back(path);   // the handle is kept open for the life of the process
        return names;
    }

    std::vector<std::string> Factory::Plugins() const
    {
        std::lock_guard<std::mutex> lock(fMutex);
        return fPlugins;
    }
}
