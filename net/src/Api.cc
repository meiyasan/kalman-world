/**
 * \file Api.cc
 * \brief The JSON verbs of kalman.api for C++ models (Kalman::Net::Api).
 *
 * Each verb mirrors its namesake in modules/api/__init__.py: the argument
 * names, the answer's keys and shapes, and the error messages, so that a
 * client cannot tell the servers apart on the verbs both serve.
 */
#include "Kalman/Net/Api.h"

#include "Registry.h"

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
#include "Kalman/Utils.h"

#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <map>
#include <mutex>
#include <random>
#include <regex>
#include <set>
#include <sstream>

namespace fs = std::filesystem;

namespace Kalman::Net
{
    namespace
    {
        // ------------------------------------------------------------------
        // JSON <-> Vector / Matrix
        // ------------------------------------------------------------------
        [[noreturn]] void Bad(const std::string &what) { throw Error(400, what); }

        json ToJson(const Vector &v)
        {
            json out = json::array();
            for (std::size_t i = 0; i < v.Size(); ++i) out.push_back(v(i));
            return out;
        }

        json ToJson(const Matrix &M)
        {
            json out = json::array();
            for (std::size_t i = 0; i < M.Rows(); ++i)
            {
                json row = json::array();
                for (std::size_t j = 0; j < M.Cols(); ++j) row.push_back(M(i, j));
                out.push_back(row);
            }
            return out;
        }

        template <typename T>
        json ToJson(const std::vector<T> &items)
        {
            json out = json::array();
            for (const auto &x : items) out.push_back(ToJson(x));
            return out;
        }

        double Number(const json &x, const std::string &what)
        {
            if (x.is_null()) return std::nan("");             // a missing channel
            if (!x.is_number()) Bad(what + " must hold numbers.");
            return x.get<double>();
        }

        // Every number of a (possibly nested) array, in row-major order.
        void Flatten(const json &x, std::vector<double> &out, const std::string &what)
        {
            if (x.is_array())
                for (const auto &e : x) Flatten(e, out, what);
            else out.push_back(Number(x, what));
        }

        Vector ParseVector(const json &x, std::size_t n, const std::string &what)
        {
            std::vector<double> flat;
            Flatten(x, flat, what);
            if (n && flat.size() != n)
                Bad(what + " has " + std::to_string(flat.size()) + " elements, expected " + std::to_string(n) + ".");
            Vector v(flat.size());
            for (std::size_t i = 0; i < flat.size(); ++i) v(i) = flat[i];
            return v;
        }

        // A matrix as a list of rows, a flat list or a scalar. rows/cols = 0 are
        // read from the nesting.
        Matrix ParseMatrix(const json &x, std::size_t rows, std::size_t cols, const std::string &what)
        {
            if (rows == 0 || cols == 0)
            {
                if (x.is_number()) { rows = rows ? rows : 1; cols = cols ? cols : 1; }
                else if (x.is_array() && !x.empty() && x[0].is_array())
                {
                    rows = rows ? rows : x.size();
                    cols = cols ? cols : x[0].size();
                }
                else if (x.is_array())
                {
                    // A flat list is one row (numpy's 1-d array as a matrix)...
                    if (!rows) rows = 1;
                    if (!cols) cols = x.size() / std::max<std::size_t>(rows, 1);
                }
                else Bad(what + " must be a matrix: a list of rows.");
            }
            std::vector<double> flat;
            Flatten(x, flat, what);
            if (flat.size() == 1 && rows * cols > 1 && rows == cols)
            {
                // A scalar for a square matrix: numpy would refuse it too, say so.
                Bad(what + " is a scalar, expected (" + std::to_string(rows) + ", " + std::to_string(cols) + ").");
            }
            if (flat.size() != rows * cols)
                Bad(what + " has " + std::to_string(flat.size()) + " elements, expected (" + std::to_string(rows)
                    + ", " + std::to_string(cols) + ").");
            Matrix M(rows, cols);
            for (std::size_t i = 0; i < flat.size(); ++i) M.Data()[i] = flat[i];
            return M;
        }

        // Measurements: N rows of m values (a flat list when m == 1).
        std::vector<Vector> ParseRows(const json &data, std::size_t m, const std::string &what)
        {
            if (!data.is_array()) Bad(what + " must be a list of rows: [[z1, z2, ...], ...].");
            std::vector<Vector> rows;
            rows.reserve(data.size());
            for (const auto &row : data)
            {
                if (!row.is_array() && m != 1)
                    Bad(what + ": a row of " + std::to_string(m) + " values was expected, got a number.");
                rows.push_back(ParseVector(row, m, what + " row " + std::to_string(rows.size())));
            }
            return rows;
        }

        // ------------------------------------------------------------------
        // Spec checking: the messages of _check_spec in Python
        // ------------------------------------------------------------------
        const std::map<std::string, std::string> kSections = {
            {"F", "dynamics"}, {"H", "dynamics"}, {"B", "dynamics"}, {"L", "dynamics"},
            {"M", "dynamics"}, {"A", "dynamics"}, {"Qc", "dynamics"}, {"Lc", "dynamics"},
            {"params", "dynamics"}, {"Q", "noise"}, {"R", "noise"}, {"x0", "initial"}, {"P0", "initial"},
            {"states", "labels"}, {"observables", "labels"}, {"engine", "bindings"},
            {"integrator", "bindings"}, {"solver", "bindings"}, {"cost", "control"},
            {"Qcost", "control"}, {"Rcost", "control"}, {"x_ref", "control"}};

        const std::set<std::string> kTopLevel = {"base", "name", "sensor_id", "fs", "dynamics", "noise",
                                                 "initial", "labels", "control", "bindings", "description"};

        const std::map<std::string, std::string> kAliases = {
            {"model", "base"}, {"class", "base"}, {"model_name", "base"},
            {"state_names", "labels.states"}, {"observable_names", "labels.observables"},
            {"measurements", "(pass measurements to run_filter's data argument)"},
            {"data", "(pass measurements to run_filter's data argument)"}};

        std::string Repr(const std::string &s) { return "'" + s + "'"; }

        void CheckSpec(const json &spec)
        {
            if (!spec.is_object()) Bad("spec must be a dict; see catalog()/describe() for the shape.");
            std::vector<std::string> problems;
            std::vector<std::string> keys;
            for (auto it = spec.begin(); it != spec.end(); ++it) keys.push_back(it.key());
            std::sort(keys.begin(), keys.end());
            for (const auto &k : keys)
                if (kSections.count(k)) problems.push_back(Repr(k) + " belongs in " + kSections.at(k) + "[" + Repr(k) + "]");
            for (const auto &k : keys)
                if (kAliases.count(k)) problems.push_back(Repr(k) + " is named " + Repr(kAliases.at(k)) + " in a spec");
            for (const auto &k : keys)
                if (!kTopLevel.count(k) && !kSections.count(k) && !kAliases.count(k))
                    problems.push_back(Repr(k) + " is not a spec key");
            if (!problems.empty())
            {
                std::string joined;
                for (std::size_t i = 0; i < problems.size(); ++i) joined += (i ? "; " : "") + problems[i];
                Bad("malformed KalmanSpec -- " + joined + ". A spec is nested: "
                    "{\"base\": \"matrices\", \"fs\": 1.0, \"dynamics\": {\"F\": .., \"H\": ..}, "
                    "\"noise\": {\"Q\": .., \"R\": ..}, \"initial\": {\"x0\": .., \"P0\": ..}}; for a "
                    "catalog model use {\"base\": \"<ClassName>\", \"dynamics\": {\"params\": {..}}}.");
            }
            const std::string base = spec.value("base", std::string("matrices"));
            if (base == "matrices" || base == "continuous")
            {
                const json dyn = spec.contains("dynamics") && spec["dynamics"].is_object() ? spec["dynamics"] : json::object();
                bool any = false;
                for (const char *k : {"F", "H", "A", "Qc", "B", "L", "M", "Lc"})
                    any = any || (dyn.contains(k) && !dyn[k].is_null());
                if (!any)
                    Bad("a " + Repr(base) + " spec carries its own dynamics, but none were given. "
                        "Provide dynamics.F and dynamics.H (base \"matrices\"), or dynamics.A and "
                        "dynamics.Qc (base \"continuous\"). To use a catalog model instead, set base "
                        "to its class name -- see catalog()['models'].");
            }
        }

        json Section(const json &spec, const char *name)
        {
            if (spec.contains(name) && spec[name].is_object()) return spec[name];
            return json::object();
        }

        bool Present(const json &obj, const char *key) { return obj.contains(key) && !obj[key].is_null(); }

        // ------------------------------------------------------------------
        // Engines
        // ------------------------------------------------------------------
        const std::vector<std::string> kEngines = {"linear", "extended", "unscented",
                                                   "particle", "ensemble", "combinatorial"};

        std::string EngineName(EngineKind kind) { return kEngines.at(static_cast<std::size_t>(kind)); }

        EngineKind EngineOf(const std::string &name)
        {
            std::string key = name;
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
            for (std::size_t i = 0; i < kEngines.size(); ++i)
                if (kEngines[i] == key) return static_cast<EngineKind>(i);
            Bad("unknown engine " + Repr(name) + "; choose from linear, extended, unscented, particle, "
                "ensemble, combinatorial.");
        }

        std::unique_ptr<Engine> MakeEngine(EngineKind kind, const json &params)
        {
            const bool has = params.is_object() && !params.empty();
            switch (kind)
            {
                case EngineKind::Linear:
                    if (has) Bad("engine 'linear' takes no parameters.");
                    return std::make_unique<Engines::Linear>();
                case EngineKind::Extended:
                    if (has) Bad("engine 'extended' takes no parameters.");
                    return std::make_unique<Engines::Extended>();
                case EngineKind::Unscented:
                    return std::make_unique<Engines::Unscented>(params.value("alpha", 1e-3), params.value("beta", 2.),
                                                                params.value("kappa", 0.));
                case EngineKind::Particle:
                    return std::make_unique<Engines::Particle>(params.value("n_particles", 500),
                                                               params.value("threshold", 0.5),
                                                               params.value("seed", 0ULL));
                case EngineKind::Ensemble:
                    return std::make_unique<Engines::Ensemble>(params.value("n_members", 50),
                                                               params.value("inflation", 1.),
                                                               params.value("seed", 0ULL));
                case EngineKind::Combinatorial:
                    Bad("the combinatorial engine consumes a whole pass, not a time series.");
            }
            Bad("no such engine.");
        }

        std::string LibraryPath()
        {
            Dl_info info{};
            if (dladdr(reinterpret_cast<void *>(&kalman_version), &info) && info.dli_fname)
                return fs::weakly_canonical(info.dli_fname).string();
            return "";
        }

        std::string Env(const char *name)
        {
            const char *s = std::getenv(name);
            return s ? std::string(s) : std::string();
        }

#if defined(__APPLE__)
        const std::vector<std::string> kExts = {".dylib", ".so"};
#else
        const std::vector<std::string> kExts = {".so", ".dylib"};
#endif

        bool EndsWith(const std::string &s, const std::string &tail)
        {
            return s.size() >= tail.size() && s.compare(s.size() - tail.size(), tail.size(), tail) == 0;
        }

        std::string Real(const std::string &p)
        {
            std::error_code ec;
            const fs::path r = fs::canonical(p, ec);
            return ec ? fs::weakly_canonical(p).string() : r.string();
        }

        // A run kept for diagnostics.
        struct Run
        {
            std::unique_ptr<Model> model;
            std::vector<Vector> measurements;   // empty: none kept
            std::vector<Vector> truth;          // a sample() run's own truth
            bool hasMeasurements = false;
        };

        std::string NewRunId()
        {
            static std::mt19937_64 rng{std::random_device{}()};
            std::ostringstream os;
            os << "run-" << std::hex;
            const std::uint64_t v = rng();
            char buf[17];
            std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
            return "run-" + std::string(buf).substr(0, 12);
        }
    }

    // ======================================================================
    // Impl
    // ======================================================================
    struct Api::Impl
    {
        Options options;
        Registry registry;
        Auth auth;
        std::mutex mutex;                               // one call at a time
        std::map<std::string, Run> runs;
        std::deque<std::string> order;                  // run ids, oldest first
        std::map<std::string, std::string> pluginErrors;

        explicit Impl(const Options &o) : options(o), registry(o.registry)
        {
            auth.SetRegistry(&registry);
            if (const std::string e = Env("KALMAN_MAX_RUNS"); !e.empty()) options.maxRuns = std::stoul(e);
        }

        // ---- plugins ------------------------------------------------------
        std::vector<std::string> PluginDirs() const
        {
            std::vector<std::string> dirs;
            const std::string env = Env("KALMAN_PLUGIN_PATH");
            if (!env.empty())
            {
                std::stringstream ss(env);
                std::string d;
                while (std::getline(ss, d, ':'))
                    if (!d.empty()) dirs.push_back(d);
            }
            else
            {
                dirs = {KALMAN_NET_INSTALL_LIBDIR, KALMAN_NET_BUILD_LIBDIR};
                const std::string lib = LibraryPath();
                if (!lib.empty()) dirs.push_back(fs::path(lib).parent_path().string());
            }
            std::vector<std::string> out;
            for (const auto &d : dirs)
            {
                const std::string r = Real(d);
                if (std::find(out.begin(), out.end(), r) == out.end()) out.push_back(r);
            }
            return out;
        }

        std::string ResolvePlugin(const std::string &name) const
        {
            if (name.empty()) Bad("plugin: give a plugin name or path.");
            const auto dirs = PluginDirs();
            bool isPath = name.find('/') != std::string::npos;
            for (const auto &e : kExts) isPath = isPath || EndsWith(name, e);
            if (isPath)
            {
                const std::string real = Real(name);
                if (!fs::is_regular_file(real)) throw Error(404, "plugin: '" + name + "' does not exist.");
                const std::string dir = fs::path(real).parent_path().string();
                if (std::find(dirs.begin(), dirs.end(), dir) == dirs.end())
                    throw Error(403, "plugin: '" + name + "' is outside the allowed directories; "
                                     "set KALMAN_PLUGIN_PATH to allow another.");
                return real;
            }
            static const std::regex kName("^[A-Za-z0-9_.+-]+$");
            if (!std::regex_match(name, kName)) Bad("plugin: '" + name + "' is not a plugin name.");
            std::size_t tried = 0;
            for (const auto &d : dirs)
            {
                std::vector<std::string> candidates;
                for (const auto &e : kExts) candidates.push_back("lib" + name + e);
                for (const auto &e : kExts) candidates.push_back(name + e);
                candidates.push_back(name);
                for (const auto &c : candidates)
                {
                    ++tried;
                    const fs::path p = fs::path(d) / c;
                    if (fs::is_regular_file(p)) return Real(p.string());
                }
            }
            std::string list;
            for (std::size_t i = 0; i < dirs.size(); ++i) list += (i ? ", " : "") + Repr(dirs[i]);
            throw Error(404, "plugin: no '" + name + "' in [" + list + "] (tried " + std::to_string(tried) + " names).");
        }

        std::vector<std::string> LoadPath(const std::string &path)
        {
            auto &factory = Factory::Instance();
            try { return factory.LoadPlugin(path); }
            catch (const std::exception &e) { throw Error(400, e.what()); }
        }

        void LoadApps()
        {
            try
            {
                const std::string path = ResolvePlugin("KalmanApps");
                LoadPath(path);
            }
            catch (const std::exception &) {}              // not built: nothing to serve from it
        }

        // _sync_plugins: what any process recorded, loaded here by *name*.
        void SyncPlugins()
        {
            json recorded;
            try { recorded = registry.ListPlugins(); }
            catch (const std::exception &) { return; }
            const auto loaded = Factory::Instance().Plugins();
            for (const auto &rec : recorded)
            {
                const std::string name = rec.value("name", "");
                if (name.empty() || pluginErrors.count(name)) continue;
                try
                {
                    const std::string path = ResolvePlugin(name);
                    if (std::find(loaded.begin(), loaded.end(), path) == loaded.end()) LoadPath(path);
                }
                catch (const std::exception &e) { pluginErrors[name] = e.what(); }
            }
        }

        // ---- spec -> model -------------------------------------------------
        std::unique_ptr<Model> Build(const json &specIn, int depth = 0)
        {
            if (depth > 8) Bad("authored specs refer to each other in a loop.");
            CheckSpec(specIn);
            const json &spec = specIn;
            const std::string base = spec.value("base", std::string("matrices"));

            // An authored spec referenced by name: that spec, as in make_model().
            if (base != "matrices" && base != "continuous" && !Factory::Instance().Has(base))
            {
                if (auto rec = registry.GetModel(base)) return Build((*rec)["spec"], depth + 1);
            }

            const std::string sensor = spec.contains("sensor_id") && spec["sensor_id"].is_string()
                                       ? spec["sensor_id"].get<std::string>() : std::string("api");
            const double fs = Present(spec, "fs") ? Number(spec["fs"], "fs") : 1.;
            const json noise = Section(spec, "noise");
            const json initial = Section(spec, "initial");
            const json dyn = Section(spec, "dynamics");

            std::unique_ptr<Model> model;
            if (base == "matrices" || base == "continuous")
            {
                for (const char *k : {"M"})
                    if (Present(dyn, k)) throw Error(501, std::string("dynamics.") + k + " (noise entering h) is served by the Python server.");
                const bool continuous = Present(dyn, "A");
                if (continuous && Present(dyn, "F")) Bad("Pass either F= or A=, not both.");
                if (!Present(dyn, "H")) Bad("a matrices spec needs dynamics.H.");
                const Matrix Fm = continuous ? ParseMatrix(dyn["A"], 0, 0, "dynamics.A")
                                             : (Present(dyn, "F") ? ParseMatrix(dyn["F"], 0, 0, "dynamics.F") : Matrix());
                const Matrix Hm = ParseMatrix(dyn["H"], 0, 0, "dynamics.H");
                const std::size_t n = Fm.Empty() ? Hm.Cols() : Fm.Rows();
                const std::size_t m = Hm.Rows();
                const Matrix Bm = Present(dyn, "B") ? ParseMatrix(dyn["B"], n, 0, "dynamics.B") : Matrix();
                const std::size_t l = Bm.Empty() ? 0 : Bm.Cols();
                auto M = std::make_unique<Model>(sensor, n, m, l);
                M->SetSampleRate(fs);
                M->SetDefaultEngine(EngineKind::Linear);
                M->SetH(ParseMatrix(dyn["H"], m, n, "dynamics.H"));
                if (!Bm.Empty()) M->SetB(Bm);
                if (Present(dyn, "L")) M->SetL(ParseMatrix(dyn["L"], n, n, "dynamics.L"));
                if (Present(noise, "R")) M->SetR(ParseMatrix(noise["R"], m, m, "noise.R"));
                if (continuous)
                {
                    if (Present(noise, "Q") && Present(dyn, "Qc")) Bad("Pass either Q= or Qc=, not both.");
                    if (Present(noise, "Q")) M->SetQ(ParseMatrix(noise["Q"], n, n, "noise.Q"));
                    M->SetGenerator(ParseMatrix(dyn["A"], n, n, "dynamics.A"));
                    if (Present(dyn, "Qc"))
                    {
                        const Matrix Qc = ParseMatrix(dyn["Qc"], 0, 0, "dynamics.Qc");
                        M->SetNoiseIntensity(Qc, Present(dyn, "Lc") ? ParseMatrix(dyn["Lc"], n, Qc.Rows(), "dynamics.Lc")
                                                                    : Matrix());
                    }
                }
                else
                {
                    if (Present(dyn, "Qc")) Bad("Qc= needs a continuous model: pass A=.");
                    if (!Fm.Empty()) M->SetF(ParseMatrix(dyn["F"], n, n, "dynamics.F"));
                    if (Present(noise, "Q")) M->SetQ(ParseMatrix(noise["Q"], n, n, "noise.Q"));
                }
                model = std::move(M);
            }
            else
            {
                auto &factory = Factory::Instance();
                if (!factory.Has(base)) SyncPlugins();            // a plugin another process loaded
                if (!factory.Has(base))
                    Bad("unknown model base " + Repr(base) + "; see catalog()['models']. A model written "
                        "in Python is served by the Python server.");
                model = factory.Create(base);
                model->SetName(sensor);
                // make_model() always binds the spec's fs (default 1.0) to a native
                // model's constructor; the same here, so both servers step alike.
                model->SetSampleRate(fs);
                if (dyn.contains("params") && dyn["params"].is_object())
                    for (auto it = dyn["params"].begin(); it != dyn["params"].end(); ++it)
                    {
                        if (it.key() == "fs" || it.key() == "sample_rate") { model->SetSampleRate(Number(it.value(), it.key())); continue; }
                        if (!model->HasParameter(it.key()))
                        {
                            std::string known;
                            for (const auto &p : model->ParameterNames()) known += (known.empty() ? "" : ", ") + p;
                            Bad(Repr(base) + " has no parameter " + Repr(it.key()) + " (it declares: "
                                + (known.empty() ? "none" : known) + ").");
                        }
                        model->SetParameter(it.key(), Number(it.value(), "dynamics.params." + it.key()));
                    }
                const std::size_t n = model->NState(), m = model->NMeasure();
                if (Present(noise, "Q")) model->SetQ(ParseMatrix(noise["Q"], n, n, "noise.Q"));
                if (Present(noise, "R")) model->SetR(ParseMatrix(noise["R"], m, m, "noise.R"));
            }

            const std::size_t n = model->NState();
            if (Present(initial, "x0")) model->SetInitialState(ParseVector(initial["x0"], n, "initial.x0"));
            if (Present(initial, "P0")) model->SetInitialCovariance(ParseMatrix(initial["P0"], n, n, "initial.P0"));
            else
            {
                // A P0 never set (all zeros) is "infer from Q", as on the Python side.
                const Matrix &P0 = model->P0();
                bool zero = P0.Rows() != n || P0.Cols() != n;
                if (!zero)
                {
                    zero = true;
                    for (std::size_t i = 0; i < n * n; ++i) zero = zero && P0.Data()[i] == 0.;
                }
                if (zero) model->SetInitialCovariance(model->Q(0.));
            }
            if (model->X0().Size() != n) model->SetInitialState(Vector(n, 0.));
            try { model->CheckShapes(); }
            catch (const std::exception &e) { Bad(e.what()); }
            return model;
        }

        std::unique_ptr<Engine> EngineFor(const json &spec, const Model &model)
        {
            const json binds = Section(spec, "bindings");
            const EngineKind kind = binds.contains("engine") && binds["engine"].is_string()
                                    ? EngineOf(binds["engine"].get<std::string>()) : model.DefaultEngine();
            return MakeEngine(kind, binds.contains("engine_params") ? binds["engine_params"] : json::object());
        }

        std::string Stash(Run run)
        {
            std::string id = NewRunId();
            while (runs.count(id)) id = NewRunId();
            while (runs.size() >= std::max<std::size_t>(options.maxRuns, 1) && !order.empty())
            {
                runs.erase(order.front());
                order.pop_front();
            }
            runs.emplace(id, std::move(run));
            order.push_back(id);
            return id;
        }

        // ---- verbs ----------------------------------------------------------
        json Native()
        {
            auto &factory = Factory::Instance();
            json out = {{"library", LibraryPath()}, {"version", KALMAN_VERSION},
                        {"models", factory.Names()}, {"plugins", factory.Plugins()},
                        {"plugin_dirs", PluginDirs()}, {"server", "c++"}};
            if (!pluginErrors.empty()) out["unavailable_plugins"] = pluginErrors;
            return out;
        }

        json Catalog()
        {
            SyncPlugins();
            json sources = json::array();
            for (const auto &s : registry.ListSources()) sources.push_back(s["name"]);
            return {{"models", Factory::Instance().Names()},
                    {"controllers", json::array()},
                    {"engines", {"combinatorial", "ensemble", "extended", "linear", "particle", "unscented"}},
                    {"integrators", {"dormand_prince", "euler", "rk4"}},
                    {"solvers", {"finite_difference", "spectral"}},
                    {"providers", json::array()},
                    {"authored", registry.ModelNames()},
                    {"sources", sources},
                    {"native", Native()}};
        }

        json Describe(const std::string &name)
        {
            SyncPlugins();
            if (auto rec = registry.GetModel(name))
            {
                const json &spec = (*rec)["spec"];
                return {{"name", name}, {"kind", "authored"},
                        {"base", spec.value("base", std::string("matrices"))},
                        {"description", spec.value("description", std::string(""))},
                        {"n", rec->value("n", json(nullptr))}, {"m", rec->value("m", json(nullptr))},
                        {"l", rec->value("l", json(nullptr))}, {"owner", (*rec)["owner"]},
                        {"spec", spec}, {"backend", "spec"}};
            }
            auto &factory = Factory::Instance();
            if (!factory.Has(name))
                throw Error(404, "unknown catalog name " + Repr(name) + "; see catalog().");
            const Factory::Entry e = factory.Describe(name);
            const auto instance = factory.Create(name);
            json engines = json::array();
            for (const auto &k : {"combinatorial", "ensemble", "extended", "linear", "particle", "unscented"})
                engines.push_back(k);
            json params = json::object();
            for (const auto &p : instance->ParameterNames()) params[p] = instance->Parameter(p);
            return {{"name", e.name}, {"title", e.title}, {"description", e.description}, {"version", e.version},
                    {"params", instance->ParameterNames()}, {"parameters", params}, {"kind", "model"},
                    {"default_engine", EngineName(e.engine)}, {"default_integrator", nullptr},
                    {"default_solver", nullptr}, {"allowed_engines", engines},
                    {"allowed_integrators", nullptr}, {"allowed_solvers", nullptr},
                    {"backend", "native"}, {"plugin", e.plugin}, {"library", LibraryPath()},
                    {"n", e.n}, {"m", e.m}, {"l", e.l}, {"fs", e.fs}, {"analytic", e.analytic}};
        }

        json Sample(const json &args)
        {
            if (!args.contains("spec")) Bad("missing field(s) ['spec'] in the request body.");
            if (!args.contains("duration") || !args["duration"].is_number()) Bad("sample needs a duration (seconds).");
            auto model = Build(args["spec"]);
            const double duration = args["duration"].get<double>();
            const std::size_t N = static_cast<std::size_t>(std::max(0., std::floor(duration * model->SampleRate() + 1e-12)));
            if (N > 10000000) Bad("at most 10,000,000 samples per call.");
            const bool processNoise = args.value("process_noise", true);
            const bool measurementNoise = args.value("measurement_noise", true);
            std::mt19937_64 rng(args.contains("seed") && args["seed"].is_number_integer()
                                ? args["seed"].get<std::uint64_t>() : std::random_device{}());
            std::normal_distribution<double> gauss(0., 1.);
            auto draw = [&](const Matrix &factor) {
                Vector w(factor.Cols());
                for (std::size_t i = 0; i < w.Size(); ++i) w(i) = gauss(rng);
                return factor * w;
            };

            const std::size_t n = model->NState(), l = model->NControl();
            const Vector u(l, 0.);
            const double dt = model->DeltaT(), t0 = model->T0();
            Vector x = model->X0();
            std::vector<Vector> xs, zs;
            std::vector<double> times;
            for (std::size_t k = 0; k < N; ++k)
            {
                const double t = t0 + static_cast<double>(k) * dt;
                // Step first, then record (the order the filter assumes).
                x = model->f(x, u, t);
                if (processNoise)
                {
                    const Vector w = draw(Utils::CholeskyFactor(model->Q(t)));
                    const Matrix Lm = model->L(t);
                    x += Lm.Empty() ? w : Lm * w;
                }
                Vector z = model->h(x, t);
                if (measurementNoise) z += draw(Utils::CholeskyFactor(model->R(t)));
                xs.push_back(x);
                zs.push_back(z);
                times.push_back(t);
            }
            (void)n;
            json out = {{"time", times}, {"xk", ToJson(xs)}, {"zk", ToJson(zs)}};
            Run run;
            run.model = std::move(model);
            run.measurements = zs;
            run.truth = xs;
            run.hasMeasurements = true;
            out["run_id"] = Stash(std::move(run));
            return out;
        }

        std::vector<Vector> FetchSource(const std::string &name, const json &query)
        {
            const auto rec = registry.GetSource(name);
            if (!rec) throw Error(404, "no source named " + Repr(name) + "; see sources().");
            if (rec->value("kind", "") != "push")
                throw Error(501, "source " + Repr(name) + " is a " + Repr(rec->value("kind", "")) +
                                 " source; its provider runs in the Python server.");
            long long since = 0, limit = 10000;
            if (query.is_object())
                for (auto it = query.begin(); it != query.end(); ++it)
                {
                    if (it.key() == "since") since = it.value().is_number() ? it.value().get<long long>() : 0;
                    else if (it.key() == "limit") limit = it.value().is_number() ? it.value().get<long long>() : 10000;
                    else Bad("a push source takes only since= and limit=, not ['" + it.key() + "'].");
                }
            std::vector<Vector> rows;
            for (const auto &r : registry.ReadMeasurements(name, since, limit))
                rows.push_back(ParseVector(r["row"], 0, "source row"));
            return rows;
        }

        json RunFilter(const json &args)
        {
            if (!args.contains("spec")) Bad("missing field(s) ['spec'] in the request body.");
            const bool hasData = args.contains("data") && !args["data"].is_null();
            const bool hasSource = args.contains("source") && !args["source"].is_null();
            if (hasData && hasSource) Bad("give data or source, not both.");
            if (!hasData && !hasSource) Bad("run_filter needs measurements: data=[[...], ...] or source=<name>.");
            auto model = Build(args["spec"]);
            const std::size_t m = model->NMeasure(), n = model->NState();
            std::vector<Vector> zk;
            if (hasSource)
            {
                zk = FetchSource(args["source"].get<std::string>(), args.value("query", json::object()));
                for (const auto &z : zk)
                    if (z.Size() != m) Bad("source rows have " + std::to_string(z.Size()) + " values, the model measures " + std::to_string(m) + ".");
            }
            else zk = ParseRows(args["data"], m, "data");
            if (zk.empty()) Bad("no measurements to filter.");

            const bool smooth = args.value("smooth", false);
            auto engine = EngineFor(args["spec"], *model);
            if (smooth && !engine->HasSmoother())
                Bad("engine " + Repr(EngineName(engine->Kind())) + " has no smoother.");

            std::vector<Vector> xs;
            std::vector<Matrix> Ps;
            std::vector<double> times;
            {
                Filter filter(model.get(), engine.get());
                const Record rec = filter.Filtering(zk, nullptr, smooth);
                xs = smooth ? filter.SmoothedStates() : filter.PosteriorStates();
                Ps = smooth ? filter.SmoothedCovariances() : filter.PosteriorCovariances();
                times = filter.Times();
                (void)rec;
            }
            (void)n;
            json out = {{"xk", ToJson(xs)}, {"Pk", ToJson(Ps)}, {"time", times}};
            Run run;
            run.model = std::move(model);
            run.measurements = std::move(zk);
            run.hasMeasurements = true;
            out["run_id"] = Stash(std::move(run));
            return out;
        }

        json Diagnostics(const std::string &id)
        {
            auto it = runs.find(id);
            if (it == runs.end()) throw Error(404, "unknown run_id " + Repr(id) + ".");
            Run &run = it->second;
            if (!run.hasMeasurements)
                Bad("run " + Repr(id) + " carries no stored measurements to diagnose; re-run it through "
                    "run_filter (which now keeps them) or pass a sample() run.");
            Model &model = *run.model;
            const std::size_t N = run.measurements.size(), n = model.NState(), m = model.NMeasure();

            // As diagnostics() in Python: a fresh filter with the model's default engine.
            Filter filter(&model);
            filter.Forward(run.measurements);
            const auto &xpr = filter.PriorStates();
            const auto &Ppr = filter.PriorCovariances();
            const auto &xpo = filter.PosteriorStates();
            const auto &Ppo = filter.PosteriorCovariances();
            const auto &times = filter.Times();

            json nu = json::array(), S = json::array(), nis = json::array();
            double sumNis = 0.;
            for (std::size_t k = 0; k < N; ++k)
            {
                const double t = times[k];
                // nu = z - h(x_prior), S = H P_prior H^T + R, H at the prior
                // (KalmanDiagnostics.innovations).
                const Matrix Hk = model.HasAnalyticJacobian() ? model.H(t) : model.JacobianH(xpr[k], t);
                Vector v = run.measurements[k] - model.h(xpr[k], t);
                Matrix Sk = Hk * Ppr[k] * Hk.Transpose() + model.R(t);
                const double e = [&] {
                    const Vector s = Utils::Solve(Sk, v);
                    double acc = 0.;
                    for (std::size_t i = 0; i < m; ++i) acc += v(i) * s(i);
                    return acc;
                }();
                nu.push_back(ToJson(v));
                S.push_back(ToJson(Sk));
                nis.push_back(e);
                sumNis += e;
            }
            json out = {{"nis", nis}, {"innovation", nu}, {"innovation_covariance", S},
                        {"mean_nis", N ? sumNis / static_cast<double>(N) : 0.}};
            if (!run.truth.empty())
            {
                // NEES against the sampled truth: (x - x_post)^T P_post^-1 (x - x_post).
                json nees = json::array();
                double sum = 0.;
                for (std::size_t k = 0; k < N; ++k)
                {
                    const Vector e = run.truth[k] - xpo[k];
                    const Vector s = Utils::Solve(Ppo[k], e);
                    double acc = 0.;
                    for (std::size_t i = 0; i < n; ++i) acc += e(i) * s(i);
                    nees.push_back(acc);
                    sum += acc;
                }
                out["nees"] = nees;
                out["mean_nees"] = N ? sum / static_cast<double>(N) : 0.;
            }
            return out;
        }

        json Forecast(const json &args)
        {
            if (!args.contains("spec") || !args.contains("x0"))
            {
                std::vector<std::string> missing;
                for (const char *k : {"spec", "x0"}) if (!args.contains(k)) missing.push_back(k);
                std::string list;
                for (std::size_t i = 0; i < missing.size(); ++i) list += (i ? ", " : "") + Repr(missing[i]);
                Bad("missing field(s) [" + list + "] in the request body.");
            }
            auto model = Build(args["spec"]);
            const std::size_t n = model->NState(), l = model->NControl();
            Vector x = ParseVector(args["x0"], n, "x0");
            Matrix P = Present(args, "P0") ? ParseMatrix(args["P0"], n, n, "P0") : model->P0();
            const long long steps = args.contains("steps") && args["steps"].is_number() ? args["steps"].get<long long>() : 1;
            if (steps < 0 || steps > 1000000) Bad("steps must be between 0 and 1,000,000.");
            const Vector u(l, 0.);
            const double dt = model->DeltaT(), t0 = model->T0();
            json xs = json::array(), Ps = json::array();
            for (long long k = 0; k < steps; ++k)
            {
                const double t = t0 + static_cast<double>(k) * dt;
                const Matrix F = model->HasAnalyticJacobian() ? model->F(t) : model->Jacobian(x, u, t);
                x = model->f(x, u, t);
                Matrix Q = model->Q(t);
                const Matrix Lm = model->L(t);
                if (!Lm.Empty()) Q = Utils::Similarity(Lm, Q);
                P = F * P * F.Transpose() + Q;
                // Each state as a column (n, 1), as the Python facade answers.
                json col = json::array();
                for (std::size_t i = 0; i < n; ++i) col.push_back(json::array({x(i)}));
                xs.push_back(col);
                Ps.push_back(ToJson(P));
            }
            return {{"xk", xs}, {"Pk", Ps}};
        }

        json LoadPlugin(const std::string &name, const std::string &token)
        {
            const std::string path = ResolvePlugin(name);
            auto &factory = Factory::Instance();
            const auto before = factory.Names();
            const auto owned = LoadPath(path);
            std::vector<std::string> added;
            for (const auto &x : factory.Names())
                if (std::find(before.begin(), before.end(), x) == before.end()) added.push_back(x);
            if (added.empty()) added = owned;
            std::string key = fs::path(name).filename().string();
            for (const auto &e : kExts)
                if (EndsWith(key, e)) key = key.substr(0, key.size() - e.size());
            const std::string who = auth.Identity(token);
            try { registry.PutPlugin(key, path, added, who.empty() ? json(nullptr) : json(who)); }
            catch (const std::exception &) {}       // no registry: loaded here only
            pluginErrors.erase(key);
            return {{"plugin", path}, {"models", added}, {"native", factory.Names()}};
        }

        json Plugins()
        {
            SyncPlugins();
            json recorded = registry.ListPlugins();
            if (!Registry::Available())
            {
                // No shared registry: what this process loaded.
                recorded = json::array();
                for (const auto &p : Factory::Instance().Plugins())
                    recorded.push_back({{"name", fs::path(p).stem().string()}, {"path", p}});
            }
            return {{"plugins", recorded}, {"unavailable", pluginErrors}};
        }

        json Whoami(const std::string &token)
        {
            if (!auth.Enabled())
                return {{"auth", false}, {"user", nullptr}, {"scopes", {"admin"}}, {"exp", nullptr}};
            const Claims c = auth.Verify(token);
            const json payload = json::parse(c.json);
            return {{"auth", true}, {"user", c.subject}, {"scopes", payload.value("scopes", json::array())},
                    {"exp", payload["exp"]}};
        }

        json Events(const json &args)
        {
            const long long since = args.contains("since") && args["since"].is_number() ? args["since"].get<long long>() : 0;
            long long limit = args.contains("limit") && args["limit"].is_number() ? args["limit"].get<long long>() : 100;
            limit = std::max(1LL, std::min(limit, 1000LL));
            if (since < 0) return {{"events", json::array()}, {"last", registry.LastEvent()}};
            const json evs = registry.Events(since, limit);
            return {{"events", evs}, {"last", evs.empty() ? json(std::max(since, 0LL)) : evs.back()["id"]}};
        }

        json DescribeSource(const std::string &name)
        {
            for (const auto &rec : registry.ListSources())
                if (rec["name"] == name) return rec;
            throw Error(404, "no source named " + Repr(name) + "; see sources().");
        }
    };

    // ======================================================================
    // The verb table
    // ======================================================================
    namespace
    {
        const std::vector<std::pair<std::string, std::string>> kVerbs = {
            {"catalog", "read"}, {"describe", "read"}, {"models", "read"}, {"sources", "read"},
            {"describe_source", "read"}, {"events", "read"}, {"plugins", "read"}, {"native", "read"},
            {"diagnostics", "read"}, {"whoami", ""}, {"sample", "send_measurement"},
            {"run_filter", "send_measurement"}, {"forecast", "send_measurement"},
            {"load_plugin", "load_plugins"}};

        const std::vector<std::pair<std::string, std::string>> kPythonOnly = {
            {"render_plot", "matplotlib draws the figures"},
            {"run_osse", "the twin experiment (kalman.apps.weather.TwinExperiment) is Python"},
            {"run_control", "the controllers' training (LQR fit, neural) is Python"},
            {"fetch", "the providers (ERA5, files, sensors, remote APIs) are Python"},
            {"providers", "the providers are Python"},
            {"describe_provider", "the providers are Python"},
            {"register_model", "a spec is validated by building it with the Python framework"},
            {"save_spec", "authoring writes Python-side files"},
            {"load_spec", "authoring reads Python-side files"},
            {"scaffold", "it emits Python source"},
            {"write_scaffold", "it emits Python source"},
            {"delete_model", "the registry's writers (ownership, events) are Python"},
            {"register_source", "a source is checked by building its Python provider"},
            {"delete_source", "the registry's writers (ownership, events) are Python"},
            {"push_measurements", "the registry's writers (ownership, events) are Python"},
            {"login", "accounts are bcrypt in the htpasswd file, with the lockout in Python"},
            {"revoke", "the revocation list is written by Python (this server reads it)"},
            {"unlock", "the lockout is Python's"}};

        std::string Arg(const json &args, const char *key)
        {
            if (!args.contains(key) || !args[key].is_string())
                Bad(std::string("missing field(s) ['") + key + "'] in the request body.");
            return args[key].get<std::string>();
        }
    }

    Api::Api() : Api(Options()) {}

    Api::Api(const Options &options) : fImpl(std::make_unique<Impl>(options))
    {
        if (options.loadApps) fImpl->LoadApps();
    }

    Api::~Api() = default;

    std::vector<std::string> Api::Verbs()
    {
        std::vector<std::string> out;
        for (const auto &v : kVerbs) out.push_back(v.first);
        return out;
    }

    std::string Api::Scope(const std::string &verb)
    {
        for (const auto &v : kVerbs)
            if (v.first == verb) return v.second;
        throw Error(404, "unknown verb " + Repr(verb) + ".");
    }

    std::vector<std::pair<std::string, std::string>> Api::PythonOnly() { return kPythonOnly; }

    const Auth &Api::GetAuth() const { return fImpl->auth; }

    long long Api::LastEvent() const
    {
        try { return fImpl->registry.LastEvent(); }
        catch (const std::exception &) { return 0; }
    }

    std::vector<std::string> Api::PluginDirs() const { return fImpl->PluginDirs(); }

    std::string Api::ResolvePlugin(const std::string &nameOrPath) const { return fImpl->ResolvePlugin(nameOrPath); }

    std::string Api::Call(const std::string &verb, const std::string &arguments, const std::string &token)
    {
        for (const auto &p : kPythonOnly)
            if (p.first == verb)
                throw Error(501, Repr(verb) + " is served by the Python server (python -m kalman.rest, "
                                 "python -m kalman.mcp): " + p.second + ".");
        const std::string scope = Scope(verb);

        json args;
        try { args = arguments.empty() ? json::object() : json::parse(arguments); }
        catch (const std::exception &e) { Bad(std::string("the arguments are not JSON: ") + e.what()); }
        if (args.is_null()) args = json::object();
        if (!args.is_object()) Bad("the arguments are a JSON object.");

        std::lock_guard<std::mutex> lock(fImpl->mutex);
        if (!scope.empty()) fImpl->auth.Authorize(token, scope);

        try
        {
            json out;
            if (verb == "catalog") out = fImpl->Catalog();
            else if (verb == "describe") out = fImpl->Describe(Arg(args, "name"));
            else if (verb == "models") out = {{"models", fImpl->registry.ListModels()}};
            else if (verb == "sources") out = {{"sources", fImpl->registry.ListSources()}};
            else if (verb == "describe_source") out = fImpl->DescribeSource(Arg(args, "name"));
            else if (verb == "events") out = fImpl->Events(args);
            else if (verb == "plugins") out = fImpl->Plugins();
            else if (verb == "native") out = fImpl->Native();
            else if (verb == "diagnostics") out = fImpl->Diagnostics(Arg(args, "run_id"));
            else if (verb == "whoami") out = fImpl->Whoami(token);
            else if (verb == "sample") out = fImpl->Sample(args);
            else if (verb == "run_filter") out = fImpl->RunFilter(args);
            else if (verb == "forecast") out = fImpl->Forecast(args);
            else if (verb == "load_plugin") out = fImpl->LoadPlugin(Arg(args, "name"), token);
            return out.dump(-1, ' ', false, json::error_handler_t::replace);
        }
        catch (const Error &) { throw; }
        catch (const json::exception &e) { throw Error(400, e.what()); }
        catch (const std::out_of_range &e) { throw Error(404, e.what()); }
        catch (const std::exception &e) { throw Error(400, e.what()); }
    }
}
