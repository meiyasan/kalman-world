/**
 * *********************************************
 *
 * \file Api.h
 * \brief Header of the Kalman::Net::Api class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Net::Api
 * \brief The JSON verbs of the Python facade `kalman.api`, for models that live in C++.
 *
 * \details One entry point, Call(): a verb name, its arguments as a JSON object,
 * the caller's token, and a JSON answer -- the same verb names, argument names
 * and answer shapes as modules/api/__init__.py, so a client written against the
 * Python REST or MCP server talks to this one unchanged.
 *
 * What it serves:
 *
 *  - every model Kalman::Factory knows: libKalmanApps (loaded at start when it
 *    is found) and the plugins loaded since, by their registered names
 *    (`{"base": "constant-velocity"}`);
 *  - authored specs of the shared registry whose base is `matrices` or
 *    `continuous`, and specs given inline with those bases;
 *  - the registry itself, read from the same SQLite file as Python
 *    (`$KALMAN_STATE_DIR/registry.sqlite`): authored models, plugins, sources,
 *    events, the revocation list.
 *
 * What it does not: models written in Python (their code is in the Python
 * process), and the verbs that need Python libraries (plots, providers, OSSE,
 * control, authoring). Those answer 501 and name the Python server.
 *
 * | verb          | scope            | arguments                                                   |
 * |---------------|------------------|-------------------------------------------------------------|
 * | catalog       | read             | --                                                          |
 * | describe      | read             | name                                                        |
 * | models        | read             | --                                                          |
 * | sources       | read             | --                                                          |
 * | describe_source | read           | name                                                        |
 * | events        | read             | since, limit                                                |
 * | plugins       | read             | --                                                          |
 * | native        | read             | --                                                          |
 * | diagnostics   | read             | run_id                                                      |
 * | whoami        | (none)           | --                                                          |
 * | sample        | send_measurement | spec, duration, seed, process_noise, measurement_noise      |
 * | run_filter    | send_measurement | spec, data or source (+query), smooth                       |
 * | forecast      | send_measurement | spec, x0, P0, steps                                         |
 * | load_plugin   | load_plugins     | name                                                        |
 *
 * Calls are serialised: one runs at a time per Api, so the servers may call it
 * from any thread.
 *
 * *********************************************
 */

#ifndef KALMAN_NET_API_H
#define KALMAN_NET_API_H

#include "Kalman/Net/Auth.h"
#include "Kalman/Net/Error.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Kalman::Net
{
    class Api
    {
        public:
            struct Options
            {
                /// Load libKalmanApps from the plugin directories at start, as the
                /// Python package does on first use.
                bool loadApps = true;
                /// The registry file; "" = $KALMAN_STATE_DIR/registry.sqlite
                /// (default ~/.kalman/state/registry.sqlite).
                std::string registry;
                /// Runs kept for diagnostics; the oldest go first (KALMAN_MAX_RUNS).
                std::size_t maxRuns = 1000;
            };

        private:
            struct Impl;
            std::unique_ptr<Impl> fImpl;

        public:
            Api();
            explicit Api(const Options &options);
            ~Api();

            Api(const Api &) = delete;
            Api &operator=(const Api &) = delete;

            /**
             * \brief Run \p verb with \p arguments (a JSON object as text, "" for none).
             * \param token the caller's bearer token ("" for none).
             * \return the answer, as JSON text.
             * \throws Kalman::Net::Error with the HTTP status of the refusal.
             */
            std::string Call(const std::string &verb, const std::string &arguments = "{}",
                             const std::string &token = "");

            /// Load a plugin as the operator does at start (kalman-serve --plugin):
            /// no token, but the same allow-list as load_plugin. \return the answer
            /// of load_plugin, as JSON text.
            std::string Preload(const std::string &plugin);

            /// The verbs served here.
            static std::vector<std::string> Verbs();
            /// The scope \p verb requires ("" for none). \throws Error 404 for an unknown verb.
            static std::string Scope(const std::string &verb);
            /// The facade verbs that stay in Python, each with the reason.
            static std::vector<std::pair<std::string, std::string>> PythonOnly();

            const Auth &GetAuth() const;

            /// The id of the newest registry event (0 for none): GET /events/cursor,
            /// which needs no token because it says only that something changed.
            long long LastEvent() const;

            /// The directories plugins may be loaded from: KALMAN_PLUGIN_PATH, or
            /// this build's install and build trees and libKalman's own directory.
            std::vector<std::string> PluginDirs() const;
            /// The file to load for a plugin name (or a path inside PluginDirs()).
            /// \throws Error 400 / 403 / 404 as resolve_plugin() in Python.
            std::string ResolvePlugin(const std::string &nameOrPath) const;
    };
}

#endif // KALMAN_NET_API_H
