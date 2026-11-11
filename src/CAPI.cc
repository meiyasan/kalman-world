/**
 * *********************************************
 *
 * \file CAPI.cc
 * \brief Implementation of the C interface
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/CAPI.h"

#include "Kalman/Config.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Engines/Extended.h"
#include "Kalman/Engines/Linear.h"
#include "Kalman/Engines/Particle.h"
#include "Kalman/Engines/Unscented.h"
#include "Kalman/Factory.h"
#include "Kalman/Filter.h"
#include "Kalman/Model.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

// The opaque handles are the C++ objects themselves: no indirection, no bookkeeping.
struct kalman_model
{
    std::unique_ptr<Kalman::Model> model;
};

struct kalman_engine
{
    std::unique_ptr<Kalman::Engine> engine;
};

namespace
{
    // One error slot and a small ring of string slots per thread. The ring is
    // what lets kalman_factory_describe() hand back five strings at once without
    // any of them overwriting another before the caller has read them.
    thread_local std::string tError;
    thread_local std::string tStrings[8];
    thread_local std::size_t tNext = 0;

    const char *Keep(const std::string &s)
    {
        std::string &slot = tStrings[tNext++ % 8];
        slot = s;
        return slot.c_str();
    }

    void Fail(const std::string &what) { tError = what; }
    void Ok() { tError.clear(); }

    // Every entry point is one of these two shapes: it either produces an int or a
    // pointer, and any exception becomes -1 / NULL plus a message.
    template <typename F>
    int Guard(F &&body)
    {
        try { Ok(); return body(); }
        catch (const std::exception &e) { Fail(e.what()); return -1; }
        catch (...) { Fail("unknown C++ exception"); return -1; }
    }

    template <typename F>
    const char *GuardStr(F &&body)
    {
        try { Ok(); return body(); }
        catch (const std::exception &e) { Fail(e.what()); return nullptr; }
        catch (...) { Fail("unknown C++ exception"); return nullptr; }
    }

    const Kalman::Model &Deref(const kalman_model_t *h)
    {
        if (!h || !h->model) throw std::invalid_argument("null kalman_model_t handle");
        return *h->model;
    }

    const Kalman::Engine &Deref(const kalman_engine_t *h)
    {
        if (!h || !h->engine) throw std::invalid_argument("null kalman_engine_t handle");
        return *h->engine;
    }

    Kalman::Engine &Deref(kalman_engine_t *h)
    {
        if (!h || !h->engine) throw std::invalid_argument("null kalman_engine_t handle");
        return *h->engine;
    }

    template <typename Make>
    kalman_engine_t *MakeEngine(Make &&make)
    {
        try
        {
            Ok();
            return new kalman_engine{make()};
        }
        catch (const std::exception &e) { Fail(e.what()); return nullptr; }
        catch (...) { Fail("unknown C++ exception"); return nullptr; }
    }

    // Kalman::Filter builds engines through a protected factory; here the caller
    // already has one, so the plain public constructor is what we want.
    std::unique_ptr<Kalman::Engine> EngineOfKind(int kind)
    {
        switch (static_cast<Kalman::EngineKind>(kind))
        {
            case Kalman::EngineKind::Linear:        return std::make_unique<Kalman::Engines::Linear>();
            case Kalman::EngineKind::Extended:      return std::make_unique<Kalman::Engines::Extended>();
            case Kalman::EngineKind::Unscented:     return std::make_unique<Kalman::Engines::Unscented>();
            case Kalman::EngineKind::Particle:      return std::make_unique<Kalman::Engines::Particle>();
            case Kalman::EngineKind::Ensemble:      return std::make_unique<Kalman::Engines::Ensemble>();
            case Kalman::EngineKind::Combinatorial: return std::make_unique<Kalman::Engines::Combinatorial>();
        }
        throw std::invalid_argument("no engine kind " + std::to_string(kind));
    }

    Kalman::Model &Deref(kalman_model_t *h)
    {
        if (!h || !h->model) throw std::invalid_argument("null kalman_model_t handle");
        return *h->model;
    }

    void Copy(const Kalman::Matrix &M, double *out, std::size_t rows, std::size_t cols, const char *what)
    {
        if (!out) throw std::invalid_argument(std::string("null output buffer for ") + what);
        if (M.Rows() != rows || M.Cols() != cols)
            throw std::runtime_error(std::string(what) + " has shape (" + std::to_string(M.Rows()) + ", "
                                     + std::to_string(M.Cols()) + "), expected (" + std::to_string(rows)
                                     + ", " + std::to_string(cols) + ")");
        std::memcpy(out, M.Data(), rows * cols * sizeof(double));
    }

    void Copy(const Kalman::Vector &v, double *out, std::size_t n, const char *what)
    {
        if (!out) throw std::invalid_argument(std::string("null output buffer for ") + what);
        if (v.Size() != n)
            throw std::runtime_error(std::string(what) + " has size " + std::to_string(v.Size())
                                     + ", expected " + std::to_string(n));
        std::memcpy(out, v.Data(), n * sizeof(double));
    }

    Kalman::Vector Vec(const double *data, std::size_t n, const char *what)
    {
        if (!data) throw std::invalid_argument(std::string("null input for ") + what);
        Kalman::Vector v(n);
        std::memcpy(v.Data(), data, n * sizeof(double));
        return v;
    }

    Kalman::Matrix Mat(const double *data, std::size_t rows, std::size_t cols, const char *what)
    {
        if (!data) throw std::invalid_argument(std::string("null input for ") + what);
        Kalman::Matrix M(rows, cols);
        std::memcpy(M.Data(), data, rows * cols * sizeof(double));
        return M;
    }

    // `u` is optional at this boundary: NULL stands for "no control", which the
    // C++ side spells as an empty vector.
    Kalman::Vector Control(const Kalman::Model &m, const double *u)
    {
        if (!u || m.NControl() == 0) return Kalman::Vector();
        return Vec(u, m.NControl(), "u");
    }
}

// The kernels (Kernels.cc) share the error slot without seeing the helpers above.
namespace Kalman::CAPI
{
    void SetError(const std::string &what) { Fail(what); }
    void ClearError() { Ok(); }
}

extern "C" {

/* --- library ------------------------------------------------------------ */

const char *kalman_version(void) { return KALMAN_VERSION; }

const char *kalman_last_error(void) { return tError.c_str(); }

/* --- factory ------------------------------------------------------------ */

int kalman_factory_count(void)
{
    return Guard([] { return static_cast<int>(Kalman::Factory::Instance().Size()); });
}

const char *kalman_factory_name(int i)
{
    return GuardStr([i]() -> const char * {
        const std::vector<std::string> names = Kalman::Factory::Instance().Names();
        if (i < 0 || static_cast<std::size_t>(i) >= names.size()) return nullptr;
        return Keep(names[static_cast<std::size_t>(i)]);
    });
}

int kalman_factory_has(const char *name)
{
    return Guard([name] {
        if (!name) throw std::invalid_argument("null model name");
        return Kalman::Factory::Instance().Has(name) ? 1 : 0;
    });
}

int kalman_factory_describe(const char *name, kalman_model_info_t *info)
{
    return Guard([name, info] {
        if (!name) throw std::invalid_argument("null model name");
        if (!info) throw std::invalid_argument("null kalman_model_info_t");
        const Kalman::Factory::Entry e = Kalman::Factory::Instance().Describe(name);
        info->name = Keep(e.name);
        info->title = Keep(e.title);
        info->description = Keep(e.description);
        info->version = Keep(e.version);
        info->plugin = Keep(e.plugin);
        info->n = static_cast<int>(e.n);
        info->m = static_cast<int>(e.m);
        info->l = static_cast<int>(e.l);
        info->fs = e.fs;
        info->engine = static_cast<int>(e.engine);
        info->analytic = e.analytic ? 1 : 0;
        return 0;
    });
}

int kalman_factory_load(const char *path)
{
    return Guard([path] {
        if (!path) throw std::invalid_argument("null plugin path");
        auto &factory = Kalman::Factory::Instance();
        const std::vector<std::string> plugins = factory.Plugins();
        const bool seen = std::find(plugins.begin(), plugins.end(), path) != plugins.end();
        const std::vector<std::string> names = factory.LoadPlugin(path);
        return seen ? 0 : static_cast<int>(names.size());
    });
}

int kalman_factory_plugin_count(void)
{
    return Guard([] { return static_cast<int>(Kalman::Factory::Instance().Plugins().size()); });
}

const char *kalman_factory_plugin(int i)
{
    return GuardStr([i]() -> const char * {
        const std::vector<std::string> plugins = Kalman::Factory::Instance().Plugins();
        if (i < 0 || static_cast<std::size_t>(i) >= plugins.size()) return nullptr;
        return Keep(plugins[static_cast<std::size_t>(i)]);
    });
}

/* --- model: lifetime and identity ---------------------------------------- */

kalman_model_t *kalman_model_create(const char *name)
{
    try
    {
        Ok();
        if (!name) throw std::invalid_argument("null model name");
        auto *h = new kalman_model{Kalman::Factory::Instance().Create(name)};
        return h;
    }
    catch (const std::exception &e) { Fail(e.what()); return nullptr; }
    catch (...) { Fail("unknown C++ exception"); return nullptr; }
}

void kalman_model_destroy(kalman_model_t *model) { delete model; }

const char *kalman_model_name(const kalman_model_t *model)
{
    return GuardStr([model] { return Keep(Deref(model).Name()); });
}

const char *kalman_model_title(const kalman_model_t *model)
{
    return GuardStr([model] { return Keep(Deref(model).Title()); });
}

const char *kalman_model_description(const kalman_model_t *model)
{
    return GuardStr([model] { return Keep(Deref(model).Description()); });
}

const char *kalman_model_model_version(const kalman_model_t *model)
{
    return GuardStr([model] { return Keep(Deref(model).Version()); });
}

const char *kalman_model_describe(const kalman_model_t *model)
{
    return GuardStr([model] { return Keep(Deref(model).Describe()); });
}

/* --- model: dimensions and time base -------------------------------------- */

int kalman_model_dims(const kalman_model_t *model, int *n, int *m, int *l)
{
    return Guard([&] {
        const Kalman::Model &M = Deref(model);
        if (n) *n = static_cast<int>(M.NState());
        if (m) *m = static_cast<int>(M.NMeasure());
        if (l) *l = static_cast<int>(M.NControl());
        return 0;
    });
}

double kalman_model_sample_rate(const kalman_model_t *model)
{
    double fs = -1.;
    Guard([&] { fs = Deref(model).SampleRate(); return 0; });
    return fs;
}

int kalman_model_set_sample_rate(kalman_model_t *model, double fs)
{
    return Guard([&] { Deref(model).SetSampleRate(fs); return 0; });
}

double kalman_model_t0(const kalman_model_t *model)
{
    double t0 = 0.;
    Guard([&] { t0 = Deref(model).T0(); return 0; });
    return t0;
}

int kalman_model_set_t0(kalman_model_t *model, double t0)
{
    return Guard([&] { Deref(model).SetT0(t0); return 0; });
}

int kalman_model_engine(const kalman_model_t *model)
{
    return Guard([&] { return static_cast<int>(Deref(model).DefaultEngine()); });
}

int kalman_model_analytic_jacobian(const kalman_model_t *model)
{
    return Guard([&] { return Deref(model).HasAnalyticJacobian() ? 1 : 0; });
}

int kalman_model_check_shapes(const kalman_model_t *model)
{
    return Guard([&] { Deref(model).CheckShapes(); return 0; });
}

/* --- model: initial state ------------------------------------------------ */

int kalman_model_x0(const kalman_model_t *model, double *x0)
{
    return Guard([&] { const auto &M = Deref(model); Copy(M.X0(), x0, M.NState(), "x0"); return 0; });
}

int kalman_model_P0(const kalman_model_t *model, double *P0)
{
    return Guard([&] { const auto &M = Deref(model); Copy(M.P0(), P0, M.NState(), M.NState(), "P0"); return 0; });
}

int kalman_model_set_x0(kalman_model_t *model, const double *x0)
{
    return Guard([&] { auto &M = Deref(model); M.SetInitialState(Vec(x0, M.NState(), "x0")); return 0; });
}

int kalman_model_set_P0(kalman_model_t *model, const double *P0)
{
    return Guard([&] {
        auto &M = Deref(model);
        M.SetInitialCovariance(Mat(P0, M.NState(), M.NState(), "P0"));
        return 0;
    });
}

/* --- model: named parameters and noise covariances ------------------------- */

int kalman_model_parameter_count(const kalman_model_t *model)
{
    return Guard([&] { return static_cast<int>(Deref(model).ParameterNames().size()); });
}

const char *kalman_model_parameter_name(const kalman_model_t *model, int i)
{
    return GuardStr([model, i]() -> const char * {
        const std::vector<std::string> names = Deref(model).ParameterNames();
        if (i < 0 || static_cast<std::size_t>(i) >= names.size()) return nullptr;
        return Keep(names[static_cast<std::size_t>(i)]);
    });
}

int kalman_model_parameter(const kalman_model_t *model, const char *name, double *value)
{
    return Guard([&] {
        if (!name || !value) throw std::invalid_argument("null name or value");
        *value = Deref(model).Parameter(name);
        return 0;
    });
}

int kalman_model_set_parameter(kalman_model_t *model, const char *name, double value)
{
    return Guard([&] {
        if (!name) throw std::invalid_argument("null name");
        Deref(model).SetParameter(name, value);
        return 0;
    });
}

int kalman_model_set_Q(kalman_model_t *model, const double *Q)
{
    return Guard([&] {
        auto &M = Deref(model);
        M.SetQ(Mat(Q, M.NState(), M.NState(), "Q"));
        return 0;
    });
}

int kalman_model_set_R(kalman_model_t *model, const double *R)
{
    return Guard([&] {
        auto &M = Deref(model);
        M.SetR(Mat(R, M.NMeasure(), M.NMeasure(), "R"));
        return 0;
    });
}

/* --- model: coefficients at time t ---------------------------------------- */

int kalman_model_F(const kalman_model_t *model, double t, double *F)
{
    return Guard([&] { const auto &M = Deref(model); Copy(M.F(t), F, M.NState(), M.NState(), "F"); return 0; });
}

int kalman_model_H(const kalman_model_t *model, double t, double *H)
{
    return Guard([&] { const auto &M = Deref(model); Copy(M.H(t), H, M.NMeasure(), M.NState(), "H"); return 0; });
}

int kalman_model_Q(const kalman_model_t *model, double t, double *Q)
{
    return Guard([&] { const auto &M = Deref(model); Copy(M.Q(t), Q, M.NState(), M.NState(), "Q"); return 0; });
}

int kalman_model_R(const kalman_model_t *model, double t, double *R)
{
    return Guard([&] { const auto &M = Deref(model); Copy(M.R(t), R, M.NMeasure(), M.NMeasure(), "R"); return 0; });
}

int kalman_model_B(const kalman_model_t *model, double t, double *B)
{
    return Guard([&] {
        const auto &M = Deref(model);
        const Kalman::Matrix Bm = M.B(t);
        if (Bm.Rows() == 0 || Bm.Cols() == 0 || M.NControl() == 0) return 0;
        Copy(Bm, B, M.NState(), M.NControl(), "B");
        return 1;
    });
}

int kalman_model_L(const kalman_model_t *model, double t, double *L)
{
    return Guard([&] {
        const auto &M = Deref(model);
        const Kalman::Matrix Lm = M.L(t);
        if (Lm.Rows() == 0 || Lm.Cols() == 0) return 0;
        Copy(Lm, L, M.NState(), M.NState(), "L");
        return 1;
    });
}

/* --- model: maps ---------------------------------------------------------- */

int kalman_model_f(const kalman_model_t *model, const double *x, const double *u, double t, double *x_next)
{
    return Guard([&] {
        const auto &M = Deref(model);
        const Kalman::Vector out = M.f(Vec(x, M.NState(), "x"), Control(M, u), t);
        Copy(out, x_next, M.NState(), "f(x)");
        return 0;
    });
}

int kalman_model_h(const kalman_model_t *model, const double *x, double t, double *z)
{
    return Guard([&] {
        const auto &M = Deref(model);
        const Kalman::Vector out = M.h(Vec(x, M.NState(), "x"), t);
        Copy(out, z, M.NMeasure(), "h(x)");
        return 0;
    });
}

int kalman_model_jacobian(const kalman_model_t *model, const double *x, const double *u, double t, double *J)
{
    return Guard([&] {
        const auto &M = Deref(model);
        const Kalman::Matrix out = M.Jacobian(Vec(x, M.NState(), "x"), Control(M, u), t);
        Copy(out, J, M.NState(), M.NState(), "df/dx");
        return 0;
    });
}

int kalman_model_jacobian_h(const kalman_model_t *model, const double *x, double t, double *J)
{
    return Guard([&] {
        const auto &M = Deref(model);
        const Kalman::Matrix out = M.JacobianH(Vec(x, M.NState(), "x"), t);
        Copy(out, J, M.NMeasure(), M.NState(), "dh/dx");
        return 0;
    });
}

/* --- engines ---------------------------------------------------------------- */

kalman_engine_t *kalman_engine_create(int kind)
{
    return MakeEngine([kind] { return EngineOfKind(kind); });
}

kalman_engine_t *kalman_engine_create_unscented(double alpha, double beta, double kappa)
{
    return MakeEngine([=]() -> std::unique_ptr<Kalman::Engine> {
        return std::make_unique<Kalman::Engines::Unscented>(alpha, beta, kappa); });
}

kalman_engine_t *kalman_engine_create_particle(int n_particles, double threshold, unsigned long long seed)
{
    return MakeEngine([=]() -> std::unique_ptr<Kalman::Engine> {
        if (n_particles < 0) throw std::invalid_argument("n_particles must be positive");
        return std::make_unique<Kalman::Engines::Particle>(static_cast<std::size_t>(n_particles), threshold, seed); });
}

kalman_engine_t *kalman_engine_create_ensemble(int n_members, double inflation, unsigned long long seed)
{
    return MakeEngine([=]() -> std::unique_ptr<Kalman::Engine> {
        if (n_members < 0) throw std::invalid_argument("n_members must be positive");
        return std::make_unique<Kalman::Engines::Ensemble>(static_cast<std::size_t>(n_members), inflation, seed); });
}

void kalman_engine_destroy(kalman_engine_t *engine) { delete engine; }

int kalman_engine_kind(const kalman_engine_t *engine)
{
    return Guard([&] { return static_cast<int>(Deref(engine).Kind()); });
}

int kalman_engine_has_smoother(const kalman_engine_t *engine)
{
    return Guard([&] { return Deref(engine).HasSmoother() ? 1 : 0; });
}

int kalman_engine_reset(kalman_engine_t *engine, const kalman_model_t *model)
{
    return Guard([&] { Deref(engine).Reset(Deref(model)); return 0; });
}

int kalman_engine_predict(const kalman_engine_t *engine, const kalman_model_t *model,
                          const double *x, const double *P, const double *u, double t,
                          double *x_out, double *P_out, double *F_out)
{
    return Guard([&] {
        const Kalman::Model &M = Deref(model);
        const std::size_t n = M.NState();
        const Kalman::Vector xv = Vec(x, n, "x");
        const Kalman::Matrix Pm = Mat(P, n, n, "P");
        Kalman::Vector uv;
        if (u && M.NControl() > 0) uv = Vec(u, M.NControl(), "u");
        Kalman::Vector xk;
        Kalman::Matrix Pk, F;
        Deref(engine).Predict(M, xv, Pm, uv.Empty() ? nullptr : &uv, t, xk, Pk, F);
        Copy(xk, x_out, n, "x_out");
        Copy(Pk, P_out, n, n, "P_out");
        if (F_out) Copy(F, F_out, n, n, "F_out");
        return 0;
    });
}

int kalman_engine_update(const kalman_engine_t *engine, const kalman_model_t *model,
                         const double *x, const double *P, const double *z, double t,
                         double *x_out, double *P_out)
{
    return Guard([&] {
        const Kalman::Model &M = Deref(model);
        const std::size_t n = M.NState();
        Kalman::Vector xo;
        Kalman::Matrix Po;
        const std::size_t used = Deref(engine).Update(M, Vec(x, n, "x"), Mat(P, n, n, "P"),
                                                      Vec(z, M.NMeasure(), "z"), t, xo, Po);
        Copy(xo, x_out, n, "x_out");
        Copy(Po, P_out, n, n, "P_out");
        return static_cast<int>(used);
    });
}

int kalman_engine_innovation(const kalman_engine_t *engine, const kalman_model_t *model,
                             const double *x, const double *P, const double *z, double t,
                             double *nu, double *S)
{
    return Guard([&] {
        const Kalman::Model &M = Deref(model);
        const std::size_t n = M.NState(), m = M.NMeasure();
        Kalman::Vector nuv;
        Kalman::Matrix Sm;
        Deref(engine).Innovation(M, Vec(x, n, "x"), Mat(P, n, n, "P"), Vec(z, m, "z"), t, nuv, Sm);
        Copy(nuv, nu, m, "nu");
        Copy(Sm, S, m, m, "S");
        return 0;
    });
}

int kalman_engine_smooth(const kalman_engine_t *engine, int n,
                         const double *x_post, const double *P_post,
                         const double *x_prior_next, const double *P_prior_next,
                         const double *x_smooth_next, const double *P_smooth_next,
                         const double *F, double *x_smooth, double *P_smooth)
{
    return Guard([&] {
        if (n <= 0) throw std::invalid_argument("n must be positive");
        const std::size_t nn = static_cast<std::size_t>(n);
        Kalman::Vector xs;
        Kalman::Matrix Ps;
        Deref(engine).Smooth(Vec(x_post, nn, "x_post"), Mat(P_post, nn, nn, "P_post"),
                             Vec(x_prior_next, nn, "x_prior_next"), Mat(P_prior_next, nn, nn, "P_prior_next"),
                             Vec(x_smooth_next, nn, "x_smooth_next"), Mat(P_smooth_next, nn, nn, "P_smooth_next"),
                             Mat(F, nn, nn, "F"), xs, Ps);
        Copy(xs, x_smooth, nn, "x_smooth");
        Copy(Ps, P_smooth, nn, nn, "P_smooth");
        return 0;
    });
}

/* --- a whole pass ------------------------------------------------------------ */

int kalman_filter_run(const kalman_model_t *model, kalman_engine_t *engine, int N,
                      const double *zk, const double *uk, int smooth,
                      double *x_prior, double *P_prior, double *x_post, double *P_post,
                      double *x_smooth, double *P_smooth, double *innovations, double *transitions)
{
    return Guard([&] {
        if (N <= 0) throw std::invalid_argument("N must be positive");
        if (!zk) throw std::invalid_argument("null zk");
        // The filter mutates nothing in the model, but its API takes a pointer.
        Kalman::Model &M = const_cast<Kalman::Model &>(Deref(model));
        Kalman::Engine &E = Deref(engine);
        const std::size_t n = M.NState(), m = M.NMeasure(), l = M.NControl();
        const std::size_t steps = static_cast<std::size_t>(N);

        std::vector<Kalman::Vector> z(steps), u;
        for (std::size_t k = 0; k < steps; ++k) z[k] = Vec(zk + k * m, m, "zk");
        if (uk && l > 0)
        {
            u.resize(steps);
            for (std::size_t k = 0; k < steps; ++k) u[k] = Vec(uk + k * l, l, "uk");
        }

        Kalman::Filter filter(&M, &E);
        filter.Forward(z, u.empty() ? nullptr : &u);
        if (smooth) filter.Backward();

        for (std::size_t k = 0; k < steps; ++k)
        {
            if (x_prior) Copy(filter.PriorStates()[k], x_prior + k * n, n, "x_prior");
            if (P_prior) Copy(filter.PriorCovariances()[k], P_prior + k * n * n, n, n, "P_prior");
            if (x_post) Copy(filter.PosteriorStates()[k], x_post + k * n, n, "x_post");
            if (P_post) Copy(filter.PosteriorCovariances()[k], P_post + k * n * n, n, n, "P_post");
            if (innovations) Copy(filter.Innovations()[k], innovations + k * m, m, "innovations");
            if (transitions) Copy(filter.Transitions()[k], transitions + k * n * n, n, n, "transitions");
            if (smooth && x_smooth) Copy(filter.SmoothedStates()[k], x_smooth + k * n, n, "x_smooth");
            if (smooth && P_smooth) Copy(filter.SmoothedCovariances()[k], P_smooth + k * n * n, n, n, "P_smooth");
        }
        return 0;
    });
}

} // extern "C"
