/**
 * *********************************************
 *
 * \file RestServer.h
 * \brief Header of the Kalman::Net::RestServer class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Net::RestServer
 * \brief The REST API of modules/rest/app.py, for the verbs Kalman::Net::Api serves.
 *
 * \details Same routes, same bodies, same answers and the same status codes,
 * so a client of the Python server (Unity, a dashboard, `curl`) talks to this
 * one unchanged. The token is the `Authorization: Bearer` header; each route
 * checks the scope the Python route checks.
 *
 *     GET  /healthz                 GET  /whoami
 *     GET  /catalog                 GET  /describe/{name}
 *     GET  /models                  GET  /plugins          POST /plugins {name}
 *     GET  /sources                 GET  /sources/{name}
 *     GET  /events/cursor           GET  /events?since=&limit=
 *     GET  /diagnostics/{run_id}
 *     POST /filter {spec, data | source, query?, smooth?}
 *     POST /forecast {spec, x0, P0?, steps?}
 *     POST /sample {spec, duration, seed?, ...}        (this server only)
 *     POST /mcp                                        (MCP over HTTP, JSON-RPC)
 *
 * The routes of the Python server that need Python (plots, providers, OSSE,
 * control, authoring, tokens) answer 501 naming it.
 *
 * *********************************************
 */

#ifndef KALMAN_NET_RESTSERVER_H
#define KALMAN_NET_RESTSERVER_H

#include "Kalman/Net/Api.h"

#include <memory>
#include <string>

namespace Kalman::Net
{
    class RestServer
    {
        private:
            struct Impl;
            std::unique_ptr<Impl> fImpl;

        public:
            /// Serve \p api (not owned; it must outlive the server).
            explicit RestServer(Api &api);
            ~RestServer();

            RestServer(const RestServer &) = delete;
            RestServer &operator=(const RestServer &) = delete;

            /// Bind \p host : \p port (0 = any free port). \return the port bound.
            /// \throws std::runtime_error when the address cannot be bound.
            int Bind(const std::string &host, int port);

            /// Serve until Stop(). Call after Bind().
            void Listen();

            /// Stop a Listen() running on another thread.
            void Stop();

            /// One request without a socket: \p method, \p path (with its query),
            /// \p body and the Authorization header. Returns the status and fills
            /// \p response. For tests and for embedding.
            int Handle(const std::string &method, const std::string &path, const std::string &body,
                       const std::string &authorization, std::string &response);
    };
}

#endif // KALMAN_NET_RESTSERVER_H
