/**
 * \file kalman-serve.cc
 * \brief The C++ servers of the framework: REST (`--rest --port N`) or MCP on stdio (`--mcp`).
 *
 *     kalman-serve --rest [--host 127.0.0.1] [--port 8000]
 *     kalman-serve --mcp
 *
 * Options: --plugin NAME (load a plugin at start, repeatable), --no-apps (do not
 * load libKalmanApps), --registry FILE (default $KALMAN_STATE_DIR/registry.sqlite).
 *
 * The environment is the Python servers': KALMAN_JWT_SECRET (auth on),
 * KALMAN_JWT_ISSUER / _AUDIENCE, KALMAN_STATE_DIR, KALMAN_PLUGIN_PATH,
 * KALMAN_MCP_TOKEN (stdio), KALMAN_ALLOW_ANONYMOUS, KALMAN_CORS_ORIGINS.
 */
#include "Kalman/Net/Net.h"

#include <csignal>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace
{
    Kalman::Net::RestServer *gServer = nullptr;

    void OnSignal(int)
    {
        if (gServer) gServer->Stop();
    }

    int Usage(int code)
    {
        (code ? std::cerr : std::cout)
            << "usage: kalman-serve (--rest [--host H] [--port N] | --mcp) [--plugin NAME]... [--no-apps]\n"
               "                    [--registry FILE]\n"
               "  --rest       the REST API of modules/rest/app.py (default 127.0.0.1:8000)\n"
               "  --mcp        the MCP server on stdio (JSON-RPC, one message per line)\n"
               "  --plugin     load a plugin by name from the allowed directories at start\n"
               "  --no-apps    do not load libKalmanApps\n"
               "  --registry   the shared registry file\n";
        return code;
    }

    // auth.check_server: an open API is the local user's only on loopback.
    bool CheckServer(const Kalman::Net::Auth &auth, const std::string &host)
    {
        if (auth.Enabled())
        {
            if (auth.Secret().size() < Kalman::Net::Auth::kMinSecretBytes)
            {
                std::cerr << "kalman-serve: KALMAN_JWT_SECRET is shorter than "
                          << Kalman::Net::Auth::kMinSecretBytes << " bytes; HS256 needs at least that.\n";
                return false;
            }
            return true;
        }
        if (host == "127.0.0.1" || host == "localhost" || host == "::1") return true;
        if (const char *a = std::getenv("KALMAN_ALLOW_ANONYMOUS"); a && std::string(a) == "1")
        {
            std::cerr << "WARNING: kalman-serve listens on " << host
                      << " WITHOUT authentication (KALMAN_ALLOW_ANONYMOUS=1).\n";
            return true;
        }
        std::cerr << "kalman-serve: refusing to listen on " << host << " without authentication. Set "
                     "KALMAN_JWT_SECRET (>= 32 bytes), bind to 127.0.0.1, or set KALMAN_ALLOW_ANONYMOUS=1 "
                     "on a network you trust.\n";
        return false;
    }
}

int main(int argc, char **argv)
{
    bool rest = false, mcp = false;
    std::string host = "127.0.0.1";
    int port = 8000;
    std::vector<std::string> plugins;
    Kalman::Net::Api::Options options;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << "kalman-serve: " << a << " needs a value\n"; std::exit(2); }
            return argv[++i];
        };
        if (a == "--rest") rest = true;
        else if (a == "--mcp") mcp = true;
        else if (a == "--host") host = next();
        else if (a == "--port") port = std::atoi(next().c_str());
        else if (a == "--plugin") plugins.push_back(next());
        else if (a == "--no-apps") options.loadApps = false;
        else if (a == "--registry") options.registry = next();
        else if (a == "-h" || a == "--help") return Usage(0);
        else { std::cerr << "kalman-serve: unknown option " << a << "\n"; return Usage(2); }
    }
    if (rest == mcp) return Usage(2);

    try
    {
        Kalman::Net::Api api(options);
        for (const auto &p : plugins) api.Preload(p);

        if (mcp)
        {
            Kalman::Net::McpServer server(api);
            server.RunStdio(std::cin, std::cout);
            return 0;
        }

        if (!CheckServer(api.GetAuth(), host)) return 1;
        Kalman::Net::RestServer server(api);
        const int bound = server.Bind(host, port);
        gServer = &server;
        std::signal(SIGINT, OnSignal);
        std::signal(SIGTERM, OnSignal);
        std::cerr << "kalman-serve: REST on http://" << host << ":" << bound
                  << (api.GetAuth().Enabled() ? " (auth on)" : " (auth off)") << std::endl;
        server.Listen();
        gServer = nullptr;
    }
    catch (const std::exception &e)
    {
        std::cerr << "kalman-serve: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
