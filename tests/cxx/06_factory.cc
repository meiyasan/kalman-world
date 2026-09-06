/**
 * \file 06_factory.cc
 * \brief The model registry, the plugin loader, and the C surface over both.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Kalman;

// Registered at static-initialisation time, exactly as an application would.
KALMAN_REGISTER_MODEL(Fixture::ConstantVelocity, "test-cv", 1, 100.);
KALMAN_REGISTER_MODEL(Fixture::ConstantVelocity, "test-cv-2d", 2, 50.);

namespace
{
    bool Contains(const std::vector<std::string> &v, const std::string &s)
    {
        for (const auto &x : v) if (x == s) return true;
        return false;
    }
}

int main()
{
    Factory &factory = Factory::Instance();

    // --- registration and lookup ---------------------------------------------
    {
        CHECK(&factory == &Factory::Instance(), "factory: one process-wide instance");
        CHECK(factory.Has("test-cv"), "factory: static registrar ran before main()");
        CHECK(factory.Has("test-cv-2d"), "factory: a configured registration is a distinct name");
        CHECK(!factory.Has("no-such-model"), "factory: unknown name is not there");

        const std::vector<std::string> names = factory.Names();
        CHECK(Contains(names, "test-cv") && Contains(names, "test-cv-2d"), "factory: Names() lists both");
        CHECK(std::is_sorted(names.begin(), names.end()), "factory: Names() is sorted");
    }

    // --- construction --------------------------------------------------------
    {
        std::unique_ptr<Model> a = factory.Create("test-cv");
        std::unique_ptr<Model> b = factory.Create("test-cv-2d");
        CHECK(a && b, "create: both construct");
        CHECK(a->NState() == 2 && a->NMeasure() == 1, "create: 1-D model has n = 2, m = 1");
        CHECK(b->NState() == 4 && b->NMeasure() == 2, "create: 2-D model has n = 4, m = 2");
        CHECK_CLOSE(b->SampleRate(), 50., 1e-12, "create: constructor arguments were forwarded");
        CHECK(a.get() != factory.Create("test-cv").get(), "create: every call is a fresh instance");

        bool threw = false;
        std::string what;
        try { factory.Create("no-such-model"); }
        catch (const std::out_of_range &e) { threw = true; what = e.what(); }
        CHECK(threw, "create: unknown name throws out_of_range");
        CHECK(what.find("no-such-model") != std::string::npos, "create: the message names the culprit");
        CHECK(what.find("test-cv") != std::string::npos, "create: the message lists what is registered");
    }

    // --- describe ------------------------------------------------------------
    {
        const Factory::Entry e = factory.Describe("test-cv-2d");
        CHECK(e.name == "test-cv-2d", "describe: name");
        CHECK(e.title == "Linear Kalman Model", "describe: title comes from the type");
        CHECK(e.n == 4 && e.m == 2 && e.l == 0, "describe: dimensions from an instance");
        CHECK(e.engine == EngineKind::Linear, "describe: default engine");
        CHECK(e.analytic, "describe: linear model has analytic Jacobian");
        CHECK(e.plugin.empty(), "describe: host-binary model has no plugin");
    }

    // --- duplicates fail loudly ----------------------------------------------
    {
        bool threw = false;
        try { factory.Register("test-cv", [] { return std::unique_ptr<Model>(new Fixture::ConstantVelocity(1, 1.)); }); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "register: a duplicate name throws");

        threw = false;
        try { factory.Register("", [] { return std::unique_ptr<Model>(new Fixture::ConstantVelocity(1, 1.)); }); }
        catch (const std::invalid_argument &) { threw = true; }
        CHECK(threw, "register: an empty name throws");

        CHECK(factory.Register("test-temp", [] { return std::unique_ptr<Model>(new Fixture::ConstantVelocity(1, 1.)); }),
              "register: a new name returns true");
        CHECK(factory.Unregister("test-temp"), "unregister: removes it");
        CHECK(!factory.Unregister("test-temp"), "unregister: second time reports absence");
    }

    // --- plugin loading ------------------------------------------------------
    {
        bool threw = false;
        try { factory.LoadPlugin("/definitely/not/a/library.so"); }
        catch (const std::runtime_error &e)
        {
            threw = true;
            CHECK(std::string(e.what()).find("library.so") != std::string::npos, "plugin: loader failure names the path");
        }
        CHECK(threw, "plugin: a missing library throws");

#ifdef KALMAN_EXAMPLE_PLUGIN
        const std::vector<std::string> added = factory.LoadPlugin(KALMAN_EXAMPLE_PLUGIN);
        CHECK(Contains(added, "constant-velocity") && Contains(added, "pendulum"),
              "plugin: the example plugin registered its models");
        CHECK(factory.Has("constant-velocity-slow"), "plugin: configured registration present");
        CHECK(Contains(factory.Plugins(), KALMAN_EXAMPLE_PLUGIN), "plugin: recorded as loaded");
        CHECK(factory.Describe("pendulum").plugin == KALMAN_EXAMPLE_PLUGIN, "plugin: entries know their origin");

        const std::vector<std::string> again = factory.LoadPlugin(KALMAN_EXAMPLE_PLUGIN);
        CHECK(again == added, "plugin: loading twice is idempotent");

        // A model from a plugin is a full Kalman::Model: run it through the filter.
        std::unique_ptr<Model> pendulum = factory.Create("pendulum");
        Sampler sampler(pendulum.get(), 7);
        sampler.Generate(200);
        Filters::EKF filter(pendulum.get());
        const Record rec = filter.Forward(sampler.Measurements());
        CHECK(rec.NSteps() == 200, "plugin: a loaded model filters like any other");
        CHECK(std::isfinite(rec.RMSE(0, sampler.Truth())), "plugin: the estimate is finite");
#else
        std::printf("  %-62s skipped (examples not built)\n", "plugin: loading the example plugin");
#endif
    }

    // --- the C surface -------------------------------------------------------
    {
        CHECK(std::strcmp(kalman_version(), KALMAN_VERSION) == 0, "capi: version string");
        CHECK(kalman_factory_count() >= 2, "capi: count sees the registered models");
        CHECK(kalman_factory_has("test-cv") == 1 && kalman_factory_has("nope") == 0, "capi: has()");
        CHECK(kalman_factory_name(1000) == nullptr, "capi: name() out of range is NULL, not a crash");

        kalman_model_info_t info;
        CHECK(kalman_factory_describe("test-cv-2d", &info) == 0, "capi: describe() succeeds");
        CHECK(info.n == 4 && info.m == 2 && info.l == 0, "capi: describe() dimensions");
        CHECK(std::strcmp(info.name, "test-cv-2d") == 0, "capi: describe() name");
        CHECK(std::strcmp(info.title, "Linear Kalman Model") == 0, "capi: describe() title");
        CHECK(info.engine == static_cast<int>(EngineKind::Linear), "capi: describe() engine kind");

        CHECK(kalman_factory_describe("nope", &info) == -1, "capi: describe() unknown returns -1");
        CHECK(std::strlen(kalman_last_error()) > 0, "capi: ... and leaves a message");
        CHECK(kalman_factory_count() >= 0 && std::strlen(kalman_last_error()) == 0,
              "capi: a successful call clears the message");

        CHECK(kalman_model_create("nope") == nullptr, "capi: create() unknown is NULL");
        kalman_model_t *h = kalman_model_create("test-cv");
        CHECK(h != nullptr, "capi: create() known model");

        int n = 0, m = 0, l = 0;
        CHECK(kalman_model_dims(h, &n, &m, &l) == 0 && n == 2 && m == 1 && l == 0, "capi: dims()");
        CHECK_CLOSE(kalman_model_sample_rate(h), 100., 1e-12, "capi: sample_rate()");
        CHECK(kalman_model_engine(h) == static_cast<int>(EngineKind::Linear), "capi: engine()");
        CHECK(kalman_model_analytic_jacobian(h) == 1, "capi: analytic_jacobian()");
        CHECK(kalman_model_check_shapes(h) == 0, "capi: check_shapes() passes");
        CHECK(std::strcmp(kalman_model_name(h), "constant-velocity") == 0, "capi: name() is the instance name");

        // The coefficients through C agree with the C++ object.
        Fixture::ConstantVelocity ref(1, 100.);
        double F[4], H[2], Q[4], R[1];
        CHECK(kalman_model_F(h, 0., F) == 0 && kalman_model_H(h, 0., H) == 0
              && kalman_model_Q(h, 0., Q) == 0 && kalman_model_R(h, 0., R) == 0, "capi: F/H/Q/R read");
        CHECK_CLOSE(F[1], ref.F()(0, 1), 1e-15, "capi: F is row-major (F[0][1] = dt)");
        CHECK_CLOSE(Q[1], ref.Q()(0, 1), 1e-15, "capi: Q off-diagonal matches");
        CHECK_CLOSE(H[0], 1., 1e-15, "capi: H matches");

        double B[4], L[4];
        CHECK(kalman_model_B(h, 0., B) == 0, "capi: B() reports empty for an uncontrolled model");
        CHECK(kalman_model_L(h, 0., L) == 0, "capi: L() reports empty (identity) coupling");

        // Maps: f is F x for the linear model; the Jacobian is F.
        const double x[2] = {1., 2.};
        double xn[2], z[1], J[4];
        CHECK(kalman_model_f(h, x, nullptr, 0., xn) == 0, "capi: f() with NULL control");
        CHECK_CLOSE(xn[0], 1. + 2. * 0.01, 1e-15, "capi: f() = F x");
        CHECK(kalman_model_h(h, x, 0., z) == 0 && z[0] == 1., "capi: h() = H x");
        CHECK(kalman_model_jacobian(h, x, nullptr, 0., J) == 0 && J[1] == F[1], "capi: jacobian() = F");
        CHECK(kalman_model_jacobian_h(h, x, 0., J) == 0 && J[0] == 1., "capi: jacobian_h() = H");

        // Initial state round trip.
        const double x0[2] = {3., -1.};
        double back[2];
        CHECK(kalman_model_set_x0(h, x0) == 0 && kalman_model_x0(h, back) == 0
              && back[0] == 3. && back[1] == -1., "capi: x0 set/get round trip");
        double P0[4] = {2., 0., 0., 5.};
        CHECK(kalman_model_set_P0(h, P0) == 0 && kalman_model_P0(h, P0) == 0 && P0[3] == 5., "capi: P0 round trip");

        // Errors come back as -1 with a reason, never as an exception or a crash.
        CHECK(kalman_model_F(nullptr, 0., F) == -1, "capi: NULL handle is an error");
        CHECK(std::strstr(kalman_last_error(), "null") != nullptr, "capi: ... that says so");
        CHECK(kalman_model_f(h, nullptr, nullptr, 0., xn) == -1, "capi: NULL input is an error");

        kalman_model_destroy(h);
        kalman_model_destroy(nullptr);
        CHECK(true, "capi: destroy(NULL) is harmless");
    }

    return Check::Summary("06_factory");
}
