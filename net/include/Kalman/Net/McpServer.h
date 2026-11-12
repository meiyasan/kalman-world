/**
 * *********************************************
 *
 * \file McpServer.h
 * \brief Header of the Kalman::Net::McpServer class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Net::McpServer
 * \brief The Model Context Protocol server of modules/mcp/server.py, for the verbs
 * Kalman::Net::Api serves.
 *
 * \details JSON-RPC 2.0: `initialize`, `ping`, `tools/list` and `tools/call`,
 * over stdio (one message per line, RunStdio()) or over HTTP (`POST /mcp` of
 * Kalman::Net::RestServer). The tools have the names, arguments and results of
 * the Python server's (`list_models`, `describe_model`, `run_filter`, ...); a
 * tool result carries the answer both as `structuredContent` and as JSON text.
 *
 * The caller's token, first match wins, as in Python: the tool's `token`
 * argument, the HTTP `Authorization: Bearer` header, and on stdio
 * `KALMAN_MCP_TOKEN`.
 *
 * *********************************************
 */

#ifndef KALMAN_NET_MCPSERVER_H
#define KALMAN_NET_MCPSERVER_H

#include "Kalman/Net/Api.h"

#include <iosfwd>
#include <string>

namespace Kalman::Net
{
    class McpServer
    {
        private:
            Api &fApi;
            std::string fServiceToken;   // KALMAN_MCP_TOKEN, on stdio

        public:
            static constexpr const char *kName = "Kalman-AI";

            /// Serve \p api (not owned).
            explicit McpServer(Api &api);

            /**
             * \brief Answer one JSON-RPC message (or a batch).
             * \param message the request, as JSON text.
             * \param bearer  the transport's token ("" on stdio).
             * \return the response as JSON text, or "" for a notification.
             */
            std::string Handle(const std::string &message, const std::string &bearer = "");

            /// Read messages from \p in, one per line, and write the answers to \p out until EOF.
            void RunStdio(std::istream &in, std::ostream &out);
    };
}

#endif // KALMAN_NET_MCPSERVER_H
