/**
 * \file RestServer.cc
 * \brief The routes of modules/rest/app.py on cpp-httplib (Kalman::Net::RestServer).
 */
#include "Kalman/Net/RestServer.h"
#include "Kalman/Net/McpServer.h"

#include <httplib.h>
#include <nlohmann/json.hpp>

#include <cstdlib>
#include <map>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace Kalman::Net
{
    using json = nlohmann::json;

    namespace
    {
        struct Reply
        {
            int status = 200;
            std::string body;
            std::string contentType = "application/json";
            bool challenge = false;          // WWW-Authenticate: Bearer
        };

        Reply Detail(int status, const std::string &message)
        {
            Reply r;
            r.status = status;
            r.body = json{{"detail", message}}.dump(-1, ' ', false, json::error_handler_t::replace);
            r.challenge = status == 401;
            return r;
        }

        std::string Decode(const std::string &s)
        {
            std::string out;
            for (std::size_t i = 0; i < s.size(); ++i)
            {
                if (s[i] == '%' && i + 2 < s.size() && std::isxdigit(s[i + 1]) && std::isxdigit(s[i + 2]))
                {
                    out += static_cast<char>(std::stoi(s.substr(i + 1, 2), nullptr, 16));
                    i += 2;
                }
                else if (s[i] == '+') out += ' ';
                else out += s[i];
            }
            return out;
        }

        std::vector<std::string> Split(const std::string &path)
        {
            std::vector<std::string> parts;
            std::string cur;
            for (char c : path)
            {
                if (c == '/') { if (!cur.empty()) parts.push_back(cur); cur.clear(); }
                else cur += c;
            }
            if (!cur.empty()) parts.push_back(cur);
            return parts;
        }

        std::string Bearer(const std::string &authorization)
        {
            if (authorization.size() > 7)
            {
                std::string head = authorization.substr(0, 7);
                for (auto &c : head) c = static_cast<char>(std::tolower(c));
                if (head == "bearer ")
                {
                    std::string t = authorization.substr(7);
                    const auto a = t.find_first_not_of(" \t");
                    const auto b = t.find_last_not_of(" \t\r\n");
                    return a == std::string::npos ? "" : t.substr(a, b - a + 1);
                }
            }
            return "";
        }

        // Routes of the Python server that need Python: method + first segment
        // (and a marker for the deeper ones).
        bool PythonRoute(const std::string &method, const std::vector<std::string> &p, std::string &why)
        {
            const std::string first = p.empty() ? "" : p[0];
            if (method == "GET" && first == "plot") { why = "plots are drawn by matplotlib"; return true; }
            if (method == "GET" && first == "providers") { why = "the providers are Python"; return true; }
            if (method == "GET" && first == "events" && p.size() == 2 && p[1] == "stream")
            { why = "the event stream (Server-Sent Events) is served by the Python server; poll GET /events"; return true; }
            if (method == "GET" && first == "sources" && p.size() == 3 && p[2] == "measurements")
            { why = "fetching a source is Python's; run_filter takes source= here"; return true; }
            if (method == "POST")
            {
                static const std::map<std::string, std::string> post = {
                    {"osse", "the twin experiment is Python"}, {"control", "the controllers are Python"},
                    {"fetch", "the providers are Python"}, {"models", "registering a model is Python's"},
                    {"sources", "registering a source or pushing rows is Python's"},
                    {"scaffold", "it emits Python source"},
                    {"token", "accounts and tokens are issued by Python"}};
                auto it = post.find(first);
                if (it != post.end()) { why = it->second; return true; }
            }
            if (method == "DELETE" && (first == "models" || first == "sources"))
            { why = "the registry's writers are Python"; return true; }
            return false;
        }
    }

    struct RestServer::Impl
    {
        Api &api;
        McpServer mcp;
        httplib::Server server;

        explicit Impl(Api &a) : api(a), mcp(a) {}

        Reply Call(const std::string &verb, const json &args, const std::string &token)
        {
            Reply r;
            try { r.body = api.Call(verb, args.dump(), token); }
            catch (const Error &e) { return Detail(e.Status(), e.what()); }
            catch (const std::exception &e) { return Detail(500, e.what()); }
            return r;
        }

        Reply Route(const std::string &method, const std::string &rawPath,
                    const std::multimap<std::string, std::string> &params, const std::string &body,
                    const std::string &authorization)
        {
            std::vector<std::string> p;
            for (const auto &s : Split(rawPath)) p.push_back(Decode(s));
            const std::string token = Bearer(authorization);
            const std::string first = p.empty() ? "" : p[0];

            auto param = [&](const char *key, long long fallback) -> long long {
                auto it = params.find(key);
                if (it == params.end()) return fallback;
                try { return std::stoll(it->second); }
                catch (const std::exception &) { throw Error(422, std::string("query parameter ") + key + " must be an integer."); }
            };
            auto parseBody = [&]() -> json {
                if (body.empty()) return json::object();
                json b;
                try { b = json::parse(body); }
                catch (const std::exception &e) { throw Error(422, std::string("the body is not JSON: ") + e.what()); }
                if (!b.is_object()) throw Error(422, "the body must be a JSON object.");
                return b;
            };

            try
            {
                std::string why;
                if (PythonRoute(method, p, why))
                    return Detail(501, method + " " + rawPath + " is served by the Python server "
                                       "(python -m kalman.rest): " + why + ".");

                if (method == "GET")
                {
                    if (p.size() == 1 && first == "healthz")
                    {
                        Reply r;
                        r.body = json{{"ok", true}, {"auth", api.GetAuth().Enabled()}, {"server", "c++"}}.dump();
                        return r;
                    }
                    if (p.size() == 1 && first == "whoami")
                    {
                        Reply r = Call("whoami", json::object(), token);
                        if (r.status == 403) return Detail(401, json::parse(r.body)["detail"].get<std::string>());
                        return r;
                    }
                    if (p.size() == 1 && first == "catalog") return Call("catalog", json::object(), token);
                    if (p.size() == 2 && first == "describe") return Call("describe", {{"name", p[1]}}, token);
                    if (p.size() == 2 && first == "diagnostics") return Call("diagnostics", {{"run_id", p[1]}}, token);
                    if (p.size() == 1 && first == "models") return Call("models", json::object(), token);
                    if (p.size() == 1 && first == "plugins") return Call("plugins", json::object(), token);
                    if (p.size() == 1 && first == "sources") return Call("sources", json::object(), token);
                    if (p.size() == 2 && first == "sources") return Call("describe_source", {{"name", p[1]}}, token);
                    if (p.size() == 2 && first == "events" && p[1] == "cursor")
                    {
                        Reply r;
                        r.body = json{{"last", api.LastEvent()}}.dump();
                        return r;
                    }
                    if (p.size() == 1 && first == "events")
                        return Call("events", {{"since", param("since", 0)}, {"limit", param("limit", 100)}}, token);
                    if (p.size() == 1 && first == "mcp")
                        return Detail(405, "MCP over HTTP takes POST /mcp with a JSON-RPC message; no SSE stream here.");
                }
                else if (method == "POST")
                {
                    if (p.size() == 1 && first == "filter") return Call("run_filter", parseBody(), token);
                    if (p.size() == 1 && first == "forecast") return Call("forecast", parseBody(), token);
                    if (p.size() == 1 && first == "sample") return Call("sample", parseBody(), token);
                    if (p.size() == 1 && first == "plugins") return Call("load_plugin", parseBody(), token);
                    if (p.size() == 1 && first == "mcp")
                    {
                        Reply r;
                        r.body = mcp.Handle(body, token);
                        if (r.body.empty()) { r.status = 202; r.body = ""; }
                        return r;
                    }
                }
                return Detail(404, "Not Found");
            }
            catch (const Error &e) { return Detail(e.Status(), e.what()); }
            catch (const std::exception &e) { return Detail(500, e.what()); }
        }
    };

    RestServer::RestServer(Api &api) : fImpl(std::make_unique<Impl>(api))
    {
        auto handler = [this](const httplib::Request &req, httplib::Response &res) {
            std::multimap<std::string, std::string> params;
            for (const auto &kv : req.params) params.emplace(kv.first, kv.second);
            const Reply r = fImpl->Route(req.method, req.path, params, req.body,
                                         req.get_header_value("Authorization"));
            res.status = r.status;
            if (r.challenge) res.set_header("WWW-Authenticate", "Bearer");
            if (!r.body.empty()) res.set_content(r.body, r.contentType);
        };
        auto &s = fImpl->server;
        s.Get(".*", handler);
        s.Post(".*", handler);
        s.Delete(".*", handler);
        s.Put(".*", handler);
        // CORS for browsers, off unless KALMAN_CORS_ORIGINS lists origins, as in Python.
        if (const char *origins = std::getenv("KALMAN_CORS_ORIGINS"); origins && *origins)
        {
            std::vector<std::string> allowed;
            std::stringstream ss(origins);
            std::string o;
            while (std::getline(ss, o, ','))
            {
                const auto a = o.find_first_not_of(' '), b = o.find_last_not_of(' ');
                if (a != std::string::npos) allowed.push_back(o.substr(a, b - a + 1));
            }
            s.set_post_routing_handler([allowed](const httplib::Request &req, httplib::Response &res) {
                const std::string origin = req.get_header_value("Origin");
                for (const auto &a : allowed)
                    if (a == "*" || a == origin)
                    {
                        res.set_header("Access-Control-Allow-Origin", a == "*" ? "*" : origin);
                        res.set_header("Access-Control-Allow-Methods", "GET, POST, DELETE");
                        res.set_header("Access-Control-Allow-Headers", "Authorization, Content-Type, Last-Event-ID");
                        break;
                    }
            });
            s.Options(".*", [](const httplib::Request &, httplib::Response &res) { res.status = 204; });
        }
    }

    RestServer::~RestServer() { Stop(); }

    int RestServer::Bind(const std::string &host, int port)
    {
        auto &s = fImpl->server;
        const int bound = port == 0 ? s.bind_to_any_port(host) : (s.bind_to_port(host, port) ? port : -1);
        if (bound < 0) throw std::runtime_error("kalman-serve: cannot bind " + host + ":" + std::to_string(port));
        return bound;
    }

    void RestServer::Listen() { fImpl->server.listen_after_bind(); }

    void RestServer::Stop()
    {
        if (fImpl && fImpl->server.is_running()) fImpl->server.stop();
    }

    int RestServer::Handle(const std::string &method, const std::string &path, const std::string &body,
                           const std::string &authorization, std::string &response)
    {
        std::string route = path;
        std::multimap<std::string, std::string> params;
        if (const auto q = path.find('?'); q != std::string::npos)
        {
            route = path.substr(0, q);
            std::stringstream ss(path.substr(q + 1));
            std::string kv;
            while (std::getline(ss, kv, '&'))
            {
                const auto eq = kv.find('=');
                if (eq == std::string::npos) params.emplace(Decode(kv), "");
                else params.emplace(Decode(kv.substr(0, eq)), Decode(kv.substr(eq + 1)));
            }
        }
        const Reply r = fImpl->Route(method, route, params, body, authorization);
        response = r.body;
        return r.status;
    }
}
