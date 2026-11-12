/**
 * \file Registry.h
 * \brief The shared registry of modules/api/store.py, read (and, for plugins,
 * written) from C++. Internal to libKalmanNet.
 *
 * One SQLite file, `$KALMAN_STATE_DIR/registry.sqlite` (default
 * `~/.kalman/state`), that every Python and C++ server on the same state
 * directory sees. The schema is store.py's; the values are JSON text. Without
 * SQLite at build time every read reports an empty registry and IsRevoked()
 * throws, so a server with auth on refuses tokens rather than ignore
 * revocations.
 */
#ifndef KALMAN_NET_REGISTRY_H
#define KALMAN_NET_REGISTRY_H

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace Kalman::Net
{
    using json = nlohmann::json;

    class Registry
    {
        private:
            std::string fPath;

        public:
            /// \p path the file; "" for $KALMAN_STATE_DIR/registry.sqlite.
            explicit Registry(std::string path = "");

            const std::string &Path() const { return fPath; }
            static bool Available();       ///< built with SQLite
            bool Exists() const;           ///< the file is there

            std::optional<json> GetModel(const std::string &name) const;
            std::vector<std::string> ModelNames() const;
            json ListModels() const;

            json ListPlugins() const;
            /// Record a plugin (store.put_plugin): true when it was not recorded before.
            bool PutPlugin(const std::string &name, const std::string &path,
                           const std::vector<std::string> &models, const json &owner) const;

            std::optional<json> GetSource(const std::string &name) const;
            json ListSources() const;
            json ReadMeasurements(const std::string &source, long long since, long long limit) const;

            json Events(long long since, long long limit) const;
            long long LastEvent() const;

            /// store.is_revoked. \throws std::runtime_error when the list cannot be read.
            bool IsRevoked(const std::string &jti, const std::string &sub, const double *iat) const;
    };
}

#endif // KALMAN_NET_REGISTRY_H
