/**
 * \file McpServer.cc
 * \brief JSON-RPC 2.0 / MCP over stdio and HTTP (Kalman::Net::McpServer).
 */
#include "Kalman/Net/McpServer.h"

#include "Kalman/Config.h"

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace Kalman::Net
{
    using json = nlohmann::json;

    namespace
    {
        const char *kLatest = "2025-06-18";
        const std::vector<std::string> kVersions = {"2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25"};

        json Prop(const char *type, const char *description)
        {
            return {{"type", type}, {"description", description}};
        }

        json Schema(json properties, std::vector<std::string> required)
        {
            properties["token"] = {{"type", "string"}, {"default", ""},
                                   {"description", "Bearer token, for clients that cannot set a header."}};
            return {{"type", "object"}, {"properties", properties}, {"required", required}};
        }

        // A tool: the facade verb it runs, and how its arguments map onto the verb's.
        struct Tool
        {
            std::string name;
            std::string verb;
            std::string description;
            json inputSchema;
        };

        const std::vector<Tool> &Tools()
        {
            static const std::vector<Tool> tools = {
                {"list_models", "catalog",
                 "List catalog models / engines / integrators / solvers / controllers / providers, the "
                 "authored models in the shared registry, and the named measurement sources.",
                 Schema(json::object(), {})},
                {"describe_model", "describe",
                 "Describe a model or controller: params, dims, bindings. For an authored model: its spec and owner.",
                 Schema({{"name", Prop("string", "A catalog name.")}}, {"name"})},
                {"list_registered_models", "models",
                 "The shared registry's authored models, with owner, dimensions and timestamps -- what other "
                 "people and agents have published.",
                 Schema(json::object(), {})},
                {"get_diagnostics", "diagnostics",
                 "Filter-consistency diagnostics (NIS + innovation; NEES for a sampled run) for a stored run.",
                 Schema({{"run_id", Prop("string", "The run_id of a run_filter or sample answer.")}}, {"run_id"})},
                {"list_sources", "sources",
                 "Named measurement sources: push buffers that devices, simulations or Unity write rows into, "
                 "and server-side providers set up by an admin. Filter one with run_filter(source=\"<name>\").",
                 Schema(json::object(), {})},
                {"list_events", "events",
                 "Registry events after `since` (model.created/replaced/deleted, plugin.loaded, source.*, "
                 "measurement.received). Pass the returned `last` as `since` next time to see only what is new.",
                 Schema({{"since", {{"type", "integer"}, {"default", 0}}}, {"limit", {{"type", "integer"}, {"default", 100}}}}, {})},
                {"run_filter", "run_filter",
                 "Filter measurements through a KalmanSpec's filter: give `data` (a list of rows), or `source` "
                 "(a registered push source name, with an optional `query`) to measure and filter in one call.",
                 Schema({{"spec", Prop("object", "A KalmanSpec.")}, {"data", Prop("array", "Measurement rows.")},
                         {"source", Prop("string", "A registered source.")}, {"query", Prop("object", "since/limit.")},
                         {"smooth", {{"type", "boolean"}, {"default", false}}}}, {"spec"})},
                {"forecast", "forecast",
                 "Forecast ahead from a state: the propagated state + covariance.",
                 Schema({{"spec", Prop("object", "A KalmanSpec.")}, {"x0", Prop("array", "The state.")},
                         {"P0", Prop("array", "Its covariance (default: the model's P0).")},
                         {"steps", {{"type", "integer"}, {"default", 1}}}}, {"spec", "x0"})},
                {"load_plugin", "load_plugin",
                 "Load a compiled (C++) model library by *name* from the server's allow-listed plugin directories "
                 "(KALMAN_PLUGIN_PATH); its models then appear in list_models and can be used as a spec base, in "
                 "every process sharing the registry. Paths are refused unless inside those directories.",
                 Schema({{"name", Prop("string", "A plugin name, e.g. KalmanPlugin_example.")}}, {"name"})},
                {"sample", "sample",
                 "Generate synthetic truth + measurements from a spec for `duration` seconds (C++ server only).",
                 Schema({{"spec", Prop("object", "A KalmanSpec.")}, {"duration", Prop("number", "Seconds.")},
                         {"seed", Prop("integer", "Seed of the noise.")},
                         {"process_noise", {{"type", "boolean"}, {"default", true}}},
                         {"measurement_noise", {{"type", "boolean"}, {"default", true}}}}, {"spec", "duration"})},
            };
            return tools;
        }

        // The Python server's tools that need Python.
        const std::vector<std::string> kPythonTools = {"list_providers", "describe_provider", "run_osse", "run_control",
                                                       "fetch_data", "push_measurements", "register_model",
                                                       "delete_model", "register_source", "scaffold_app"};

        json RpcError(const json &id, int code, const std::string &message)
        {
            return {{"jsonrpc", "2.0"}, {"id", id}, {"error", {{"code", code}, {"message", message}}}};
        }

        json Result(const json &id, json result) { return {{"jsonrpc", "2.0"}, {"id", id}, {"result", result}}; }

        json ToolText(const std::string &text, bool error)
        {
            return {{"content", json::array({{{"type", "text"}, {"text", text}}})}, {"isError", error}};
        }
    }

    McpServer::McpServer(Api &api) : fApi(api)
    {
        if (const char *t = std::getenv("KALMAN_MCP_TOKEN")) fServiceToken = t;
    }

    namespace
    {
        json HandleOne(Api &api, const json &msg, const std::string &bearer, bool &reply)
        {
            reply = true;
            if (!msg.is_object() || msg.value("jsonrpc", "") != "2.0" || !msg.contains("method")
                || !msg["method"].is_string())
                return RpcError(msg.is_object() && msg.contains("id") ? msg["id"] : json(nullptr), -32600,
                                "Invalid Request");
            const bool notification = !msg.contains("id");
            const json id = notification ? json(nullptr) : msg["id"];
            const std::string method = msg["method"].get<std::string>();
            const json params = msg.contains("params") && msg["params"].is_object() ? msg["params"] : json::object();
            if (notification) { reply = false; return json(); }

            if (method == "initialize")
            {
                std::string version = params.value("protocolVersion", std::string(kLatest));
                bool known = false;
                for (const auto &v : kVersions) known = known || v == version;
                if (!known) version = kLatest;
                return Result(id, {{"protocolVersion", version},
                                   {"capabilities", {{"tools", {{"listChanged", false}}},
                                                     {"resources", {{"listChanged", false}, {"subscribe", false}}}}},
                                   {"serverInfo", {{"name", McpServer::kName}, {"title", McpServer::kName},
                                                   {"version", KALMAN_VERSION}}},
                                   {"instructions", "Kalman estimation framework (C++ server): models compiled "
                                                    "into libKalman and its plugins, filters and diagnostics. "
                                                    "Models written in Python are served by the Python server."}});
            }
            if (method == "ping") return Result(id, json::object());
            if (method == "tools/list")
            {
                json tools = json::array();
                for (const auto &t : Tools())
                    tools.push_back({{"name", t.name}, {"description", t.description}, {"inputSchema", t.inputSchema}});
                return Result(id, {{"tools", tools}});
            }
            if (method == "tools/call")
            {
                const std::string name = params.value("name", "");
                json args = params.contains("arguments") && params["arguments"].is_object() ? params["arguments"]
                                                                                           : json::object();
                for (const auto &p : kPythonTools)
                    if (p == name)
                        return Result(id, ToolText("Error executing tool " + name + ": it is served by the Python "
                                                   "MCP server (python -m kalman.mcp).", true));
                const Tool *tool = nullptr;
                for (const auto &t : Tools())
                    if (t.name == name) tool = &t;
                if (!tool) return RpcError(id, -32602, "Unknown tool: " + name);

                std::string token = args.contains("token") && args["token"].is_string() ? args["token"].get<std::string>() : "";
                args.erase("token");
                if (token.empty()) token = bearer;
                try
                {
                    const std::string answer = api.Call(tool->verb, args.dump(), token);
                    json out = ToolText(answer, false);
                    out["structuredContent"] = json::parse(answer);
                    return Result(id, out);
                }
                catch (const std::exception &e)
                {
                    return Result(id, ToolText("Error executing tool " + name + ": " + e.what(), true));
                }
            }
            if (method == "resources/list")
                return Result(id, {{"resources", json::array({
                    {{"uri", "catalog://models"}, {"name", "catalog_models"}, {"mimeType", "application/json"},
                     {"description", "Every model usable as a spec base: catalog, plugins and authored."}},
                    {{"uri", "catalog://sources"}, {"name", "catalog_sources"}, {"mimeType", "application/json"},
                     {"description", "The named measurement sources."}},
                    {{"uri", "registry://events"}, {"name", "registry_events"}, {"mimeType", "application/json"},
                     {"description", "The 100 newest registry events, oldest first."}}})}});
            if (method == "resources/read")
            {
                const std::string uri = params.value("uri", "");
                try
                {
                    json body;
                    if (uri == "catalog://models")
                    {
                        const json cat = json::parse(api.Call("catalog", "{}", bearer));
                        body = {{"models", cat["models"]}, {"authored", cat.value("authored", json::array())}};
                    }
                    else if (uri == "catalog://sources") body = json::parse(api.Call("sources", "{}", bearer));
                    else if (uri == "registry://events")
                    {
                        const long long last = api.LastEvent();
                        body = json::parse(api.Call("events", json{{"since", std::max(last - 100, 0LL)}, {"limit", 100}}.dump(), bearer));
                    }
                    else return RpcError(id, -32602, "Unknown resource: " + uri);
                    return Result(id, {{"contents", json::array({{{"uri", uri}, {"mimeType", "application/json"},
                                                                  {"text", body.dump()}}})}});
                }
                catch (const std::exception &e) { return RpcError(id, -32603, e.what()); }
            }
            if (method == "prompts/list") return Result(id, {{"prompts", json::array()}});
            if (method == "resources/templates/list") return Result(id, {{"resourceTemplates", json::array()}});
            return RpcError(id, -32601, "Method not found: " + method);
        }
    }

    std::string McpServer::Handle(const std::string &message, const std::string &bearer)
    {
        json msg;
        try { msg = json::parse(message); }
        catch (const std::exception &) { return RpcError(nullptr, -32700, "Parse error").dump(); }

        bool reply = false;
        if (msg.is_array())
        {
            if (msg.empty()) return RpcError(nullptr, -32600, "Invalid Request").dump();
            json out = json::array();
            for (const auto &m : msg)
            {
                json r = HandleOne(fApi, m, bearer, reply);
                if (reply) out.push_back(r);
            }
            return out.empty() ? "" : out.dump(-1, ' ', false, json::error_handler_t::replace);
        }
        json r = HandleOne(fApi, msg, bearer, reply);
        return reply ? r.dump(-1, ' ', false, json::error_handler_t::replace) : "";
    }

    void McpServer::RunStdio(std::istream &in, std::ostream &out)
    {
        std::string line;
        while (std::getline(in, line))
        {
            if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
            const std::string answer = Handle(line, fServiceToken);
            if (!answer.empty()) out << answer << '\n' << std::flush;
        }
    }
}
