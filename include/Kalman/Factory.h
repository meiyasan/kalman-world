/**
 * *********************************************
 *
 * \file Factory.h
 * \brief Header of the Kalman::Factory class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Factory
 * \brief The registry of compiled models — how a C++ model becomes *nameable*.
 *
 * \details libKalman ships no physics: a model is written downstream, in
 * an application, a ROOT+ analysis or a plugin. The factory is what lets such a
 * model be found **by name** by something that did not compile it — the Python
 * package's catalogue, the MCP server, a ROOT macro — and constructed without the
 * caller knowing its type.
 *
 * A model registers itself once, at static-initialisation time:
 *
 * \code{.cpp}
 * class ConstantVelocity : public Kalman::Models::Linear { ... };
 * KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity");
 * \endcode
 *
 * and is then available to every consumer of the library:
 *
 * \code{.cpp}
 * auto &factory = Kalman::Factory::Instance();
 * for (const auto &name : factory.Names()) std::cout << name << '\n';
 * std::unique_ptr<Kalman::Model> model = factory.Create("constant-velocity");
 * \endcode
 *
 * Models compiled into a **shared library** register the same way; LoadPlugin()
 * dlopens the library and the static registrars run. That is the one code path by
 * which a model written for ROOT+ — linking ROOT, RooFit or anything else — can
 * reach the Python side: ROOT stays inside the plugin, and the framework never
 * sees it.
 *
 * Every failure is loud: registering a name twice, creating an unknown name or
 * loading a library that does not exist throws with a message that names what was
 * expected. There is no silent fallback anywhere in this class.
 *
 * \warning Loading a plugin executes code. Nothing in this class decides *which*
 * libraries may be loaded — that policy belongs to the caller (the Python API
 * keeps an allow-list of directories), not to the registry.
 *
 * *********************************************
 */

#ifndef KALMAN_FACTORY_H
#define KALMAN_FACTORY_H

#include "Kalman/Constants.h"
#include "Kalman/Model.h"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Kalman
{
    class Factory
    {
        public:
            using Creator = std::function<std::unique_ptr<Model>()>;

            /// What the catalogue shows for a name, harvested from a fresh instance.
            struct Entry
            {
                std::string name;         ///< the registered (catalogue) name
                std::string title;        ///< Model::Title()
                std::string description;  ///< Model::Description()
                std::string version;      ///< Model::Version()
                std::size_t n = 0;        ///< state dimension
                std::size_t m = 0;        ///< measurement dimension
                std::size_t l = 0;        ///< control dimension
                double fs = 0.;           ///< sample rate [Hz]
                EngineKind engine = EngineKind::Linear;
                bool analytic = true;     ///< analytic Jacobian, or numerical
                std::string plugin;       ///< library it came from ("" = the host binary)
            };

        private:
            struct Registration
            {
                Creator create;
                std::string plugin;
            };

            mutable std::mutex fMutex;
            std::map<std::string, Registration> fModels;
            std::vector<std::string> fPlugins;      // loaded library paths, in order
            std::string fLoading;                   // path of the plugin being loaded, if any
            std::string fLoadError;                 // a registration that failed *during* a load

            Factory() = default;

        public:
            Factory(const Factory &) = delete;
            Factory &operator=(const Factory &) = delete;

            /// The process-wide registry.
            static Factory &Instance();

            /**
             * \brief Register a creator under \p name.
             * \throws std::invalid_argument on an empty name or a null creator.
             * \throws std::runtime_error if \p name is already registered.
             * \return true, so the call can seed a static initialiser (see the macro).
             */
            bool Register(const std::string &name, Creator create);

            /// Forget \p name. Returns false when it was not registered.
            bool Unregister(const std::string &name);

            bool Has(const std::string &name) const;

            /// Every registered name, sorted.
            std::vector<std::string> Names() const;

            /// Number of registered models.
            std::size_t Size() const;

            /**
             * \brief Construct a fresh instance of \p name.
             * \throws std::out_of_range naming the unknown model and listing the known ones.
             */
            std::unique_ptr<Model> Create(const std::string &name) const;

            /**
             * \brief Catalogue entry for \p name.
             *
             * Title, description and dimensions are instance properties in this
             * library, so a temporary is built and discarded to read them.
             * \throws std::out_of_range as Create().
             */
            Entry Describe(const std::string &name) const;

            /**
             * \brief dlopen a shared library whose static initialisers register models.
             * \param path the library file (absolute, or resolvable by the loader).
             * \return the names that library added, sorted.
             * \throws std::runtime_error with the loader's own message on failure, or
             *         if the library registered nothing (almost always a forgotten
             *         KALMAN_REGISTER_MODEL, and worth hearing about).
             *
             * A library already loaded returns the names it registered the first
             * time. Libraries are never unloaded: instances created from them may
             * still be alive.
             */
            std::vector<std::string> LoadPlugin(const std::string &path);

            /// Paths of every loaded plugin, in load order.
            std::vector<std::string> Plugins() const;
    };
}

#define KALMAN_FACTORY_CONCAT_(a, b) a##b
#define KALMAN_FACTORY_CONCAT(a, b) KALMAN_FACTORY_CONCAT_(a, b)

/**
 * \brief Register \p Class under \p Name at static-initialisation time.
 *
 * Any further arguments are forwarded to the constructor, so a model that takes
 * parameters registers a *configured* instance:
 *
 * \code{.cpp}
 * KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity");
 * KALMAN_REGISTER_MODEL(ConstantVelocity, "constant-velocity-2d", 2, 100.);
 * \endcode
 *
 * Use it once per name, at namespace scope, in exactly one translation unit.
 */
#define KALMAN_REGISTER_MODEL(Class, Name, ...)                                          \
    namespace                                                                           \
    {                                                                                   \
        const bool KALMAN_FACTORY_CONCAT(kalman_registered_model_, __LINE__) =          \
            ::Kalman::Factory::Instance().Register(                                     \
                Name, []() -> std::unique_ptr<::Kalman::Model>                          \
                { return std::make_unique<Class>(__VA_ARGS__); });                      \
    }

#endif // KALMAN_FACTORY_H
