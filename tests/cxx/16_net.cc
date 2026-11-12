/**
 * \file 16_net.cc
 * \brief libKalmanNet: the JSON verbs on a registered model, the error replies,
 * the REST routes and the MCP exchange without a socket, and the JWT check
 * against a token minted by PyJWT.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Net/Net.h"
#include "Check.h"
#include "Fixtures.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <string>

using json = nlohmann::json;
using namespace Kalman;

KALMAN_REGISTER_MODEL(Fixture::ConstantVelocity, "net-cv", 1, 10.);

namespace
{
    // Minted by PyJWT 2.15.1 (modules/api/auth.py's library) with this secret:
    //   jwt.encode({"sub": "alice", "scopes": ["read", "send_measurement"],
    //               "iat": 1700000000.5, "exp": 4102444800, "jti": "0123456789abcdef"},
    //              kSecret, algorithm="HS256")
    const char *kSecret = "kalman-net-test-secret-0123456789abcdef";
    const char *kPyJwtToken =
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJhbGljZSIsInNjb3BlcyI6WyJyZWFkIiwic2VuZF9tZWFzdXJl"
        "bWVudCJdLCJpYXQiOjE3MDAwMDAwMDAuNSwiZXhwIjo0MTAyNDQ0ODAwLCJqdGkiOiIwMTIzNDU2Nzg5YWJjZGVmIn0."
        "4_96fS__yRZQg7tYz0DaMLHyDf4oL4LvsInZ-zEdiFk";

    int StatusOf(Net::Api &api, const std::string &verb, const std::string &args, const std::string &token = "")
    {
        try { api.Call(verb, args, token); }
        catch (const Net::Error &e) { return e.Status(); }
        return 200;
    }

    std::string MessageOf(Net::Api &api, const std::string &verb, const std::string &args)
    {
        try { api.Call(verb, args); }
        catch (const Net::Error &e) { return e.what(); }
        return "";
    }

    double Now()
    {
        using namespace std::chrono;
        return duration_cast<duration<double>>(system_clock::now().time_since_epoch()).count();
    }
}

int main()
{
    // A registry that does not exist: an empty one, and nothing revoked.
    ::unsetenv("KALMAN_JWT_SECRET");
    ::unsetenv("KALMAN_JWT_ISSUER");
    ::unsetenv("KALMAN_JWT_AUDIENCE");
    Net::Api::Options options;
    options.loadApps = false;
    options.registry = "/nonexistent-kalman-net-test/registry.sqlite";
    Net::Api api(options);

    const json spec = {{"base", "net-cv"}, {"fs", 10.},
                       {"initial", {{"x0", {0., 1.}}, {"P0", {{1., 0.}, {0., 1.}}}}}};

    // --- introspection --------------------------------------------------------
    {
        const json cat = json::parse(api.Call("catalog"));
        bool listed = false;
        for (const auto &m : cat["models"]) listed = listed || m == "net-cv";
        CHECK(listed, "catalog: a registered model is listed");
        CHECK(cat["native"]["server"] == "c++", "catalog: native says which server answers");

        const json d = json::parse(api.Call("describe", R"({"name": "net-cv"})"));
        CHECK(d["n"] == 2 && d["m"] == 1 && d["l"] == 0, "describe: dimensions");
        CHECK(d["backend"] == "native" && d["kind"] == "model", "describe: backend and kind");
        CHECK(d["default_engine"] == "linear", "describe: default engine");
    }

    // --- run_filter against Kalman::Filter directly ------------------------------
    std::string runId;
    {
        const std::vector<double> z = {0.1, 0.25, 0.27, 0.41, 0.52, 0.58, 0.71, 0.80};
        json data = json::array();
        for (double v : z) data.push_back({v});
        const json out = json::parse(api.Call("run_filter", json{{"spec", spec}, {"data", data}}.dump()));
        runId = out["run_id"].get<std::string>();

        Fixture::ConstantVelocity model(1, 10.);
        model.SetInitialState(Vector{0., 1.});
        model.SetInitialCovariance(Matrix::Identity(2));
        std::vector<Vector> zk;
        for (double v : z) zk.push_back(Vector{v});
        Filter filter(&model);
        filter.Forward(zk);
        double worst = 0.;
        for (std::size_t k = 0; k < z.size(); ++k)
            for (std::size_t i = 0; i < 2; ++i)
            {
                worst = std::max(worst, std::fabs(out["xk"][k][i].get<double>() - filter.PosteriorStates()[k](i)));
                for (std::size_t j = 0; j < 2; ++j)
                    worst = std::max(worst, std::fabs(out["Pk"][k][i][j].get<double>()
                                                      - filter.PosteriorCovariances()[k](i, j)));
            }
        CHECK_CLOSE(worst, 0., 1e-12, "run_filter: states and covariances = Kalman::Filter");
        CHECK_CLOSE(out["time"][3].get<double>(), 0.3, 1e-12, "run_filter: time tags t0 + k/fs");

        const json sm = json::parse(api.Call("run_filter", json{{"spec", spec}, {"data", data}, {"smooth", true}}.dump()));
        filter.Backward();
        CHECK_CLOSE(sm["xk"][0][0].get<double>(), filter.SmoothedStates()[0](0), 1e-12, "run_filter: smooth=true is RTS");

        const json diag = json::parse(api.Call("diagnostics", json{{"run_id", runId}}.dump()));
        const std::vector<double> nis = filter.NIS();
        CHECK(diag["nis"].size() == z.size(), "diagnostics: one NIS per step");
        CHECK_CLOSE(diag["nis"][4].get<double>(), nis[4], 1e-9, "diagnostics: NIS = Filter::NIS()");
        CHECK(diag["innovation_covariance"][0].size() == 1, "diagnostics: S is (m, m) per step");
    }

    // --- forecast -----------------------------------------------------------------
    {
        const json out = json::parse(api.Call("forecast", json{{"spec", spec}, {"x0", {1., 2.}}, {"steps", 3}}.dump()));
        Fixture::ConstantVelocity model(1, 10.);
        Vector x{1., 2.};
        Matrix P = Matrix::Identity(2);
        for (int k = 0; k < 3; ++k)
        {
            x = model.F() * x;
            P = model.F() * P * model.F().Transpose() + model.Q();
        }
        CHECK(out["xk"].size() == 3 && out["xk"][0].size() == 2 && out["xk"][0][0].size() == 1,
              "forecast: xk is (steps, n, 1), as the Python facade");
        CHECK_CLOSE(out["xk"][2][0][0].get<double>(), x(0), 1e-12, "forecast: mean");
        CHECK_CLOSE(out["Pk"][2][0][1].get<double>(), P(0, 1), 1e-12, "forecast: covariance");
    }

    // --- sample -------------------------------------------------------------------
    {
        const json a = json::parse(api.Call("sample", json{{"spec", spec}, {"duration", 1.}, {"seed", 7}}.dump()));
        const json b = json::parse(api.Call("sample", json{{"spec", spec}, {"duration", 1.}, {"seed", 7}}.dump()));
        CHECK(a["xk"].size() == 10 && a["zk"][0].size() == 1, "sample: int(duration * fs) steps");
        CHECK(a["zk"] == b["zk"], "sample: a seed gives the same run");
        const json c = json::parse(api.Call("sample", json{{"spec", spec}, {"duration", 1.}, {"process_noise", false},
                                                           {"measurement_noise", false}}.dump()));
        CHECK_CLOSE(c["xk"][9][0].get<double>(), 1.0, 1e-12, "sample: noise-free truth steps with f (x = t v)");
        const json d = json::parse(api.Call("diagnostics", json{{"run_id", a["run_id"]}}.dump()));
        CHECK(d.contains("nees") && d["nees"].size() == 10, "diagnostics: a sampled run has NEES");
    }

    // --- error replies ----------------------------------------------------------------
    {
        CHECK(StatusOf(api, "describe", R"({"name": "no-such-model"})") == 404, "describe: unknown name is 404");
        CHECK(StatusOf(api, "run_filter", json{{"spec", {{"base", "no-such"}}}, {"data", {{1.}}}}.dump()) == 400,
              "run_filter: unknown base is 400");
        CHECK(StatusOf(api, "run_filter", json{{"spec", spec}}.dump()) == 400, "run_filter: no data is 400");
        CHECK(StatusOf(api, "run_filter", json{{"spec", spec}, {"data", {{1., 2.}}}}.dump()) == 400,
              "run_filter: a row of the wrong width is 400");
        CHECK(StatusOf(api, "diagnostics", R"({"run_id": "run-nope"})") == 404, "diagnostics: unknown run is 404");
        CHECK(StatusOf(api, "render_plot", "{}") == 501, "render_plot: stays Python (501)");
        CHECK(StatusOf(api, "no_such_verb", "{}") == 404, "an unknown verb is 404");
        CHECK(StatusOf(api, "catalog", "[1, 2]") == 400, "arguments must be an object");
        const std::string msg = MessageOf(api, "run_filter",
                                          json{{"spec", {{"F", {{1.}}}}}, {"data", {{1.}}}}.dump());
        CHECK(msg.find("'F' belongs in dynamics['F']") != std::string::npos,
              "malformed spec: the message of _check_spec");
        CHECK(StatusOf(api, "load_plugin", R"({"name": "/etc/passwd.so"})") >= 403,
              "load_plugin: a path outside the allowed directories is refused");
    }

    // --- an inline matrices spec ------------------------------------------------------
    {
        const json m = {{"base", "matrices"}, {"fs", 2.},
                        {"dynamics", {{"F", {{1., 0.5}, {0., 1.}}}, {"H", {{1., 0.}}}}},
                        {"noise", {{"Q", {{0.01, 0.}, {0., 0.01}}}, {"R", {{0.25}}}}},
                        {"initial", {{"x0", {0., 0.}}, {"P0", {{1., 0.}, {0., 1.}}}}}};
        const json out = json::parse(api.Call("run_filter", json{{"spec", m}, {"data", {1., 2., 3.}}}.dump()));
        CHECK(out["xk"].size() == 3 && out["xk"][0].size() == 2, "matrices spec: filtered in C++");
    }

    // --- JWT --------------------------------------------------------------------------
    {
        Net::Auth auth(kSecret);
        Net::Auth none(kSecret);   // no registry: fail-closed
        bool closed = false;
        try { none.Verify(kPyJwtToken); }
        catch (const Net::Error &e) { closed = e.Status() == 401; }
        CHECK(closed, "JWT: no revocation list, no token (fail-closed)");

        // Through the Api, whose registry (absent here) is an empty list.
        ::setenv("KALMAN_JWT_SECRET", kSecret, 1);
        Net::Api secured(options);
        const json who = json::parse(secured.Call("whoami", "{}", kPyJwtToken));
        CHECK(who["user"] == "alice" && who["scopes"].size() == 2, "JWT: a PyJWT token verifies");
        CHECK(StatusOf(secured, "catalog", "{}", kPyJwtToken) == 200, "JWT: read scope allows catalog");
        CHECK(StatusOf(secured, "load_plugin", R"({"name": "x"})", kPyJwtToken) == 403,
              "JWT: missing scope is 403");
        CHECK(StatusOf(secured, "catalog", "{}") == 401, "JWT: no token is 401");
        std::string forged = kPyJwtToken;
        forged.back() = forged.back() == 'k' ? 'j' : 'k';
        CHECK(StatusOf(secured, "catalog", "{}", forged) == 401, "JWT: a forged signature is 401");
        const std::string other = Net::Auth::Sign(
            json{{"sub", "alice"}, {"scopes", {"read"}}, {"exp", 4102444800.}}.dump(), "another-secret-of-32-bytes-or-more!!");
        CHECK(StatusOf(secured, "catalog", "{}", other) == 401, "JWT: another secret is 401");
        const std::string expired = Net::Auth::Sign(
            json{{"sub", "bob"}, {"scopes", {"admin"}}, {"exp", Now() - 10.}}.dump(), kSecret);
        CHECK(StatusOf(secured, "catalog", "{}", expired) == 401, "JWT: an expired token is 401");
        const std::string admin = Net::Auth::Sign(
            json{{"sub", "root"}, {"scopes", {"admin"}}, {"exp", Now() + 60.}, {"iat", Now() - 1.}}.dump(), kSecret);
        CHECK(StatusOf(secured, "forecast", json{{"spec", spec}, {"x0", {0., 0.}}}.dump(), admin) == 200,
              "JWT: admin covers every scope");
        const std::string noSub = Net::Auth::Sign(json{{"scopes", {"admin"}}, {"exp", Now() + 60.}}.dump(), kSecret);
        CHECK(StatusOf(secured, "catalog", "{}", noSub) == 401, "JWT: sub is required");

        // REST without a socket: the same statuses and the Python routes.
        Net::RestServer rest(secured);
        std::string body;
        CHECK(rest.Handle("GET", "/healthz", "", "", body) == 200, "REST: /healthz needs no token");
        CHECK(rest.Handle("GET", "/catalog", "", "", body) == 401, "REST: /catalog without a token is 401");
        CHECK(rest.Handle("GET", "/catalog", "", std::string("Bearer ") + kPyJwtToken, body) == 200,
              "REST: /catalog with a token is 200");
        CHECK(rest.Handle("GET", "/describe/net-cv", "", std::string("Bearer ") + kPyJwtToken, body) == 200
              && json::parse(body)["n"] == 2, "REST: /describe/{name}");
        CHECK(rest.Handle("POST", "/filter", json{{"spec", spec}, {"data", {{0.1}, {0.2}}}}.dump(),
                          std::string("Bearer ") + kPyJwtToken, body) == 200, "REST: POST /filter");
        const std::string rid = json::parse(body)["run_id"];
        CHECK(rest.Handle("GET", "/diagnostics/" + rid, "", std::string("Bearer ") + kPyJwtToken, body) == 200,
              "REST: /diagnostics/{run_id}");
        CHECK(rest.Handle("POST", "/plugins", R"({"name": "x"})", std::string("Bearer ") + kPyJwtToken, body) == 403,
              "REST: POST /plugins needs load_plugins");
        CHECK(rest.Handle("GET", "/plot/" + rid, "", "", body) == 501, "REST: /plot stays Python (501)");
        CHECK(rest.Handle("GET", "/nowhere", "", "", body) == 404, "REST: unknown route is 404");
        CHECK(rest.Handle("POST", "/forecast", "{not json", std::string("Bearer ") + kPyJwtToken, body) == 422,
              "REST: a body that is not JSON is 422");

        // MCP: initialize, tools/list, tools/call, with the token as an argument.
        Net::McpServer mcp(secured);
        const json init = json::parse(mcp.Handle(
            R"({"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}})"));
        CHECK(init["result"]["serverInfo"]["name"] == "Kalman-AI", "MCP: initialize");
        CHECK(mcp.Handle(R"({"jsonrpc":"2.0","method":"notifications/initialized"})").empty(),
              "MCP: a notification gets no answer");
        const json tools = json::parse(mcp.Handle(R"({"jsonrpc":"2.0","id":2,"method":"tools/list"})"));
        bool hasFilter = false;
        for (const auto &t : tools["result"]["tools"]) hasFilter = hasFilter || t["name"] == "run_filter";
        CHECK(hasFilter, "MCP: tools/list has run_filter");
        const json call = json::parse(mcp.Handle(json{{"jsonrpc", "2.0"}, {"id", 3}, {"method", "tools/call"},
            {"params", {{"name", "describe_model"}, {"arguments", {{"name", "net-cv"}, {"token", kPyJwtToken}}}}}}.dump()));
        CHECK(call["result"]["isError"] == false && call["result"]["structuredContent"]["n"] == 2,
              "MCP: tools/call describe_model");
        const json denied = json::parse(mcp.Handle(json{{"jsonrpc", "2.0"}, {"id", 4}, {"method", "tools/call"},
            {"params", {{"name", "list_models"}, {"arguments", json::object()}}}}.dump()));
        CHECK(denied["result"]["isError"] == true, "MCP: a call without a token is an error result");
        const json bad = json::parse(mcp.Handle("{oops"));
        CHECK(bad["error"]["code"] == -32700, "MCP: a parse error is -32700");
        ::unsetenv("KALMAN_JWT_SECRET");
    }

    return Check::Summary("16_net");
}
