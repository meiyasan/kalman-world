/**
 * \file kalman.cc
 * \brief `kalman`: what this installation of libKalman can run.
 *
 *     kalman                      the models, one per line
 *     kalman models [PATTERN]     the same, names containing PATTERN only
 *     kalman model NAME           everything one model declares: the state vector and
 *                                 the measurement component by component (name, unit,
 *                                 noise), the sample rate, dynamics, parameters
 *     kalman model FILE           the same for a model file
 *     kalman export NAME|FILE     write a model out (see Kalman::IO::ModelFile): to -o FILE
 *                                 (.kw: one archive; another name ending in .json: plain
 *                                 JSON), or its model.json on the standard output
 *     kalman settings FILE.kw     write FILE.kw.json, the settings file beside a model
 *                                 file (what is changed from it), or update it with --set
 *     kalman plugins              the libraries the models came from
 *     kalman version
 *
 * Export options: --linearize (the Jacobians at the initial state, as a
 * matrices model), --steps N (samples to tabulate for a time-varying model),
 * --set NAME=VALUE (a parameter, repeatable).
 *
 * A model file is opened with the settings file beside it (FILE.kw.json) when
 * there is one; --no-settings leaves it out.
 *
 * Options: --json (machine-readable), --plugin FILE|NAME (load a plugin first,
 * repeatable), --no-apps (do not load libKalmanApps), --color auto|always|never
 * (auto: colours on a terminal, none in a pipe or when NO_COLOR is set).
 *
 * The program links libKalman only. The application models are a plugin like
 * any other: libKalmanApps is looked for beside libKalman and loaded when it is
 * there. A plugin given by name is looked for in KALMAN_PLUGIN_PATH, or beside
 * libKalman when that is unset.
 */
#include "Kalman/CAPI.h"
#include "Kalman/Config.h"
#include "Kalman/Factory.h"
#include "Kalman/IO/ModelFile.h"

#include <dlfcn.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
    // What one model declares, read from a fresh instance.
    struct Sheet
    {
        std::string name, title, description, version, plugin, error;
        std::size_t n = 0, m = 0, l = 0;
        double fs = 0.;
        std::string dynamics, structure, engine, differentiator, integrator;
        std::string file, kind;         // the file it was read from and what that file holds, if any
        std::string settings;           // the settings file applied, and
        std::vector<std::string> changed;   // what it changed
        std::size_t substeps = 1;
        bool processThroughMap = false, measurementThroughMap = false;
        bool noiseIntensity = false;    // Q stated per unit time and decoded for the sample rate
        bool coupling = false;          // a process-noise coupling L other than the identity
        std::vector<double> q, r, x0;   // diagonals of Q and R, and the initial state
        bool qFull = false, rFull = false;
        std::vector<std::pair<std::string, double>> parameters;
        std::vector<std::pair<std::string, std::vector<double>>> arrays;
        std::vector<Kalman::Model::Component> state, measurement;
    };

    std::vector<double> Diagonal(const Kalman::Matrix &M, bool &full)
    {
        std::vector<double> d;
        full = false;
        for (std::size_t i = 0; i < M.Rows(); ++i)
            for (std::size_t j = 0; j < M.Cols(); ++j)
            {
                if (i == j) d.push_back(M(i, j));
                else if (M(i, j) != 0.) full = true;
            }
        return d;
    }

    bool IsIdentity(const Kalman::Matrix &M)
    {
        for (std::size_t i = 0; i < M.Rows(); ++i)
            for (std::size_t j = 0; j < M.Cols(); ++j)
                if (M(i, j) != (i == j ? 1. : 0.)) return false;
        return true;
    }

    Sheet Read(const std::string &name, const Kalman::Model *given = nullptr)
    {
        Sheet s;
        s.name = name;
        try
        {
            auto &factory = Kalman::Factory::Instance();
            std::unique_ptr<Kalman::Model> created;
            if (!given)
            {
                s.plugin = factory.Describe(name).plugin;
                created = factory.Create(name);
            }
            const Kalman::Model *model = given ? given : created.get();
            s.structure = Kalman::ToString(model->Structure());
            s.title = model->Title();
            s.description = model->Description();
            s.version = model->Version();
            s.n = model->NState();
            s.m = model->NMeasure();
            s.l = model->NControl();
            s.fs = model->SampleRate();
            s.dynamics = Kalman::ToString(model->Dynamics());
            s.engine = Kalman::ToString(model->DefaultEngine());
            s.differentiator = Kalman::ToString(model->Differentiator());
            if (model->HasContinuousDynamics())
            {
                s.integrator = Kalman::ToString(model->DefaultIntegrator());
                s.substeps = model->Substeps();
            }
            else if (!model->Generator().Empty()) s.integrator = "exact (matrix exponential)";
            s.processThroughMap = model->HasProcessNoiseInput();
            s.measurementThroughMap = model->HasMeasurementNoiseInput();
            s.noiseIntensity = !model->NoiseIntensity().Empty();
            const Kalman::Matrix L = model->L(model->T0());
            s.coupling = !L.Empty() && !IsIdentity(L);
            s.q = Diagonal(model->Q(model->T0()), s.qFull);
            s.r = Diagonal(model->R(model->T0()), s.rFull);
            s.state = model->StateComponents();
            s.measurement = model->MeasurementComponents();
            const Kalman::Vector &x0 = model->X0();
            for (std::size_t i = 0; i < x0.Size(); ++i) s.x0.push_back(x0(i));
            for (const auto &p : model->ParameterNames()) s.parameters.emplace_back(p, model->Parameter(p));
            for (const auto &a : model->ArrayNames()) s.arrays.emplace_back(a, model->Array(a));
        }
        catch (const std::exception &e) { s.error = e.what(); }
        return s;
    }

    // "additive", or where the noise goes when it is not simply added.
    std::string ProcessNoise(const Sheet &s)
    {
        std::string out = s.processThroughMap ? "in f" : (s.coupling ? "through L" : "additive");
        if (s.noiseIntensity) out += ", per unit time";
        return out;
    }

    std::string MeasurementNoise(const Sheet &s) { return s.measurementThroughMap ? "in h" : "additive"; }

    std::string Number(double v)
    {
        std::ostringstream os;
        os << std::setprecision(6) << v;
        return os.str();
    }

    std::string List(const std::vector<double> &v)
    {
        std::string out;
        for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + Number(v[i]);
        return out;
    }

    // ---- JSON, by hand: the core has no JSON dependency ------------------
    std::string Quote(const std::string &s)
    {
        std::ostringstream os;
        os << '"';
        for (const unsigned char c : s)
        {
            if (c == '"' || c == '\\') os << '\\' << c;
            else if (c == '\n') os << "\\n";
            else if (c == '\t') os << "\\t";
            else if (c < 0x20) os << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
            else os << c;
        }
        os << '"';
        return os.str();
    }

    std::string QuoteList(const std::vector<std::string> &v)
    {
        std::string out = "[";
        for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + Quote(v[i]);
        return out + "]";
    }

    std::string JsonNumber(double v)
    {
        if (!std::isfinite(v)) return "null";
        std::ostringstream os;
        os << std::setprecision(17) << v;
        return os.str();
    }

    std::string JsonList(const std::vector<double> &v)
    {
        std::string out = "[";
        for (std::size_t i = 0; i < v.size(); ++i) out += (i ? ", " : "") + JsonNumber(v[i]);
        return out + "]";
    }

    std::string JsonComponents(const std::vector<Kalman::Model::Component> &components)
    {
        std::string out = "[";
        for (std::size_t i = 0; i < components.size(); ++i)
            out += std::string(i ? ", " : "") + "{\"name\": " + Quote(components[i].name) + ", \"unit\": "
                 + Quote(components[i].unit) + "}";
        return out + "]";
    }

    std::string Json(const Sheet &s, bool detail)
    {
        std::ostringstream os;
        os << "{\"name\": " << Quote(s.name);
        if (!s.error.empty())
        {
            os << ", \"error\": " << Quote(s.error) << "}";
            return os.str();
        }
        os << ", \"title\": " << Quote(s.title)
           << ", \"n\": " << s.n << ", \"m\": " << s.m << ", \"l\": " << s.l
           << ", \"fs\": " << JsonNumber(s.fs)
           << ", \"dynamics\": " << Quote(s.dynamics)
           << ", \"structure\": " << Quote(s.structure)
           << ", \"engine\": " << Quote(s.engine)
           << ", \"differentiator\": " << Quote(s.differentiator)
           << ", \"integrator\": " << (s.integrator.empty() ? "null" : Quote(s.integrator))
           << ", \"substeps\": " << s.substeps
           << ", \"process_noise\": " << Quote(s.processThroughMap ? "in f" : (s.coupling ? "through L" : "additive"))
           << ", \"process_noise_per_unit_time\": " << (s.noiseIntensity ? "true" : "false")
           << ", \"measurement_noise\": " << Quote(MeasurementNoise(s))
           << ", \"plugin\": " << Quote(s.plugin);
        if (detail)
        {
            os << ", \"description\": " << Quote(s.description)
               << ", \"version\": " << Quote(s.version)
               << ", \"Q_diagonal\": " << JsonList(s.q) << ", \"Q_is_diagonal\": " << (s.qFull ? "false" : "true")
               << ", \"R_diagonal\": " << JsonList(s.r) << ", \"R_is_diagonal\": " << (s.rFull ? "false" : "true")
               << ", \"file\": " << (s.file.empty() ? "null" : Quote(s.file))
               << ", \"settings\": " << (s.settings.empty() ? "null" : Quote(s.settings))
               << ", \"settings_changed\": " << QuoteList(s.changed)
               << ", \"x0\": " << JsonList(s.x0) << ", \"dt\": " << JsonNumber(s.fs > 0. ? 1. / s.fs : 1.)
               << ", \"state\": " << JsonComponents(s.state)
               << ", \"measurement\": " << JsonComponents(s.measurement) << ", \"parameters\": {";
            for (std::size_t i = 0; i < s.parameters.size(); ++i)
                os << (i ? ", " : "") << Quote(s.parameters[i].first) << ": " << JsonNumber(s.parameters[i].second);
            os << "}";
            os << ", \"arrays\": {";
            for (std::size_t i = 0; i < s.arrays.size(); ++i)
                os << (i ? ", " : "") << Quote(s.arrays[i].first) << ": " << JsonList(s.arrays[i].second);
            os << "}";
        }
        os << "}";
        return os.str();
    }

    // ---- colour -------------------------------------------------------------
    // Off unless the output is a terminal: a pipe, a file and --json get plain text.
    bool gColor = false;

    const char *const kBold = "1", *const kDim = "2", *const kRed = "31", *const kGreen = "32",
                *const kYellow = "33", *const kBlue = "34", *const kMagenta = "35", *const kCyan = "36",
                *const kTitle = "1;36", *const kHead = "1;4";

    /// \p text padded to \p width (counted on the text, not on the escape codes), in \p style.
    std::string Paint(const std::string &text, const char *style, std::size_t width = 0)
    {
        const std::string pad(width > text.size() ? width - text.size() : 0, ' ');
        if (!gColor || !style || text.empty()) return text + pad;
        return std::string("\033[") + style + "m" + text + "\033[0m" + pad;
    }

    const char *DynamicsStyle(const std::string &dynamics) { return dynamics == "continuous" ? kGreen : kYellow; }
    /// Additive noise is the plain case; noise that goes through a map stands out.
    const char *NoiseStyle(const std::string &noise) { return noise.rfind("additive", 0) == 0 ? nullptr : kMagenta; }

    // ---- text ---------------------------------------------------------------
    void Table(const std::vector<Sheet> &sheets)
    {
        const std::vector<std::string> head = {"MODEL", "DYNAMICS", "STRUCTURE", "ENGINE", "n", "m", "l", "fs [Hz]", "JACOBIAN",
                                               "PROCESS NOISE", "MEAS. NOISE"};
        std::vector<std::vector<std::string>> rows;
        for (const auto &s : sheets)
        {
            if (!s.error.empty())
            {
                rows.push_back({s.name, "error: " + s.error});
                continue;
            }
            rows.push_back({s.name, s.dynamics, s.structure, s.engine, std::to_string(s.n), std::to_string(s.m),
                            std::to_string(s.l), Number(s.fs), s.differentiator, ProcessNoise(s),
                            MeasurementNoise(s)});
        }
        std::vector<std::size_t> width(head.size());
        for (std::size_t c = 0; c < head.size(); ++c) width[c] = head[c].size();
        for (const auto &r : rows)
            if (r.size() == head.size())
                for (std::size_t c = 0; c < r.size(); ++c) width[c] = std::max(width[c], r[c].size());
        // The last column is not padded, so no line ends in spaces.
        const auto line = [&width](const std::vector<std::string> &r, const std::vector<const char *> &style)
        {
            for (std::size_t c = 0; c < r.size(); ++c)
                std::cout << Paint(r[c], c < style.size() ? style[c] : nullptr, c + 1 == r.size() ? 0 : width[c] + 2);
            std::cout << '\n';
        };
        line(head, std::vector<const char *>(head.size(), kBold));
        for (const auto &r : rows)
        {
            if (r.size() != head.size()) line(r, {kTitle, kRed});
            else line(r, {kTitle, DynamicsStyle(r[1]), nullptr, kBlue, nullptr, nullptr, nullptr, nullptr, nullptr,
                          NoiseStyle(r[9]), NoiseStyle(r[10])});
        }
    }

    // "128 Hz, one sample every 7.8125 ms": the step in the unit that reads best.
    std::string Rate(double fs)
    {
        if (!(fs > 0.)) return "not set";
        const double dt = 1. / fs;
        std::string step;
        if (dt < 1e-3) step = Number(dt * 1e6) + " us";
        else if (dt < 1.) step = Number(dt * 1e3) + " ms";
        else if (dt < 120.) step = Number(dt) + " s";
        else if (dt < 7200.) step = Number(dt / 60.) + " min";
        else step = Number(dt / 3600.) + " h";
        return Number(fs) + " Hz, one sample every " + step;
    }

    // One line per component: index, name, unit, then the columns given.
    void Components(const std::vector<Kalman::Model::Component> &components,
                    const std::vector<std::pair<std::string, const std::vector<double> *>> &columns)
    {
        std::size_t name = 4, unit = 4;
        for (const auto &c : components)
        {
            name = std::max(name, c.name.size());
            unit = std::max(unit, c.unit.size());
        }
        const std::size_t index = std::to_string(components.empty() ? 0 : components.size() - 1).size() + 2;
        // Padded between columns only, so no line ends in spaces.
        const auto tail = [](const std::vector<std::string> &cells, const char *style)
        {
            for (std::size_t c = 0; c < cells.size(); ++c)
                std::cout << Paint(cells[c], style, c + 1 == cells.size() ? 0 : 14);
            std::cout << '\n';
        };
        std::vector<std::string> cells;
        for (const auto &col : columns) cells.push_back(col.first);
        std::cout << "    " << Paint("", nullptr, index + 2) << Paint("name", kBold, name + 2)
                  << Paint("unit", kBold, unit + 2);
        tail(cells, kBold);
        for (std::size_t i = 0; i < components.size(); ++i)
        {
            std::cout << "    " << Paint("[" + std::to_string(i) + "]", kDim, index + 2)
                      << Paint(components[i].name, kCyan, name + 2)
                      << Paint(components[i].unit.empty() ? "-" : components[i].unit, kYellow, unit + 2);
            cells.clear();
            for (const auto &col : columns) cells.push_back(i < col.second->size() ? Number((*col.second)[i]) : "-");
            tail(cells, nullptr);
        }
    }

    void Detail(const Sheet &s)
    {
        const auto row = [](const char *key, const std::string &value, const char *style = nullptr)
        { std::cout << "  " << Paint(key, kDim, 20) << Paint(value, style) << '\n'; };
        std::vector<double> qSigma, rSigma;
        for (const double q : s.q) qSigma.push_back(q >= 0. ? std::sqrt(q) : std::nan(""));
        for (const double r : s.r) rSigma.push_back(r >= 0. ? std::sqrt(r) : std::nan(""));

        std::cout << Paint(s.name, kTitle) << '\n';
        row("title", s.title + " (version " + s.version + ")");
        row("description", s.description);
        if (!s.file.empty()) row("file", s.file + " (" + s.kind + ")");
        if (!s.settings.empty())
        {
            std::string what;
            for (const auto &c : s.changed) what += (what.empty() ? "" : ", ") + c;
            row("settings", s.settings + (what.empty() ? " (changes nothing)" : " (changes " + what + ")"),
                what.empty() ? nullptr : kMagenta);
        }
        else row("from", s.plugin.empty() ? "this program" : s.plugin);
        row("sample rate", Rate(s.fs));
        row("dynamics", s.dynamics, DynamicsStyle(s.dynamics));
        row("structure", s.structure);
        if (!s.integrator.empty())
            row("integrator", s.integrator + (s.substeps > 1 ? ", " + std::to_string(s.substeps) + " substeps" : ""));
        row("engine", s.engine, kBlue);
        row("Jacobian", s.differentiator);
        row("control l", std::to_string(s.l));

        std::cout << "\n  " << Paint("state vector, n = " + std::to_string(s.n), kHead) << " (process noise "
                  << Paint(ProcessNoise(s), NoiseStyle(ProcessNoise(s))) << (s.qFull ? "; Q has off-diagonal terms, its diagonal is shown" : "") << ")\n";
        Components(s.state, {{"x0", &s.x0}, {"Q", &s.q}, {"sqrt(Q)", &qSigma}});

        std::cout << "\n  " << Paint("measurement, m = " + std::to_string(s.m), kHead) << " (noise "
                  << Paint(MeasurementNoise(s), NoiseStyle(MeasurementNoise(s))) << (s.rFull ? "; R has off-diagonal terms, its diagonal is shown" : "") << ")\n";
        Components(s.measurement, {{"R", &s.r}, {"sqrt(R)", &rSigma}});

        std::cout << "\n  " << Paint("parameters", kHead)
                  << (s.parameters.empty() && s.arrays.empty() ? ": none" : "") << '\n';
        std::size_t width = 0;
        for (const auto &p : s.parameters) width = std::max(width, p.first.size());
        for (const auto &a : s.arrays) width = std::max(width, a.first.size());
        for (const auto &p : s.parameters)
            std::cout << "    " << Paint(p.first, kCyan, width + 2) << Number(p.second) << '\n';
        // An array: its length, then its first values.
        for (const auto &a : s.arrays)
        {
            std::string values;
            for (std::size_t i = 0; i < a.second.size() && i < 4; ++i) values += (i ? ", " : "") + Number(a.second[i]);
            if (a.second.size() > 4) values += ", ...";
            std::cout << "    " << Paint(a.first, kCyan, width + 2) << Paint("[" + std::to_string(a.second.size()) + "]", kDim)
                      << "  " << values << '\n';
        }
    }

    // ---- plugins ----------------------------------------------------------
    std::string LibraryDir()
    {
        Dl_info info{};
        if (dladdr(reinterpret_cast<void *>(&kalman_version), &info) && info.dli_fname)
        {
            std::error_code ec;
            const fs::path p = fs::weakly_canonical(info.dli_fname, ec);
            if (!ec) return p.parent_path().string();
        }
        return "";
    }

    std::vector<std::string> PluginDirs()
    {
        std::vector<std::string> dirs;
        if (const char *env = std::getenv("KALMAN_PLUGIN_PATH"); env && *env)
        {
            std::stringstream ss(env);
            std::string d;
            while (std::getline(ss, d, ':'))
                if (!d.empty()) dirs.push_back(d);
        }
        else if (const std::string lib = LibraryDir(); !lib.empty()) dirs.push_back(lib);
        return dirs;
    }

    // A file as given, or a name looked for in the plugin directories. "" when not found.
    std::string Resolve(const std::string &what)
    {
        if (what.find('/') != std::string::npos || fs::is_regular_file(what)) return what;
        for (const auto &d : PluginDirs())
            for (const char *ext : {".dylib", ".so", ".dll"})
                for (const std::string &file : {"lib" + what + ext, what + ext})
                    if (fs::is_regular_file(fs::path(d) / file)) return (fs::path(d) / file).string();
        return "";
    }

    int Usage(int code)
    {
        (code ? std::cerr : std::cout)
            << "usage: kalman [models [PATTERN] | model NAME|FILE | plugins | version] [--json]\n"
               "              [--plugin FILE|NAME]... [--no-apps] [--color auto|always|never]\n"
               "       kalman export NAME|FILE [-o FILE] [--linearize] [--steps N] [--set NAME=VALUE]...\n"
               "       kalman settings FILE.kw [--set NAME=VALUE]...\n"
               "  models     the models this installation can run, one per line (the default)\n"
               "  model      everything one model declares: dimensions, dynamics, noise, parameters\n"
               "  export     write a model out: -o m.kw (one archive: model.json and its tables) or\n"
               "             -o m.json (plain JSON); without -o, model.json on the standard output.\n"
               "             A model of matrices is written whole; a model given by code is written as a\n"
               "             reference to its name and settings, or with --linearize as its Jacobians at\n"
               "             the initial state; a time-varying one needs --steps N\n"
               "  settings   write m.kw.json beside m.kw: the sample rate, parameters, noise and initial\n"
               "             state to edit, applied whenever m.kw is opened; --set changes a parameter in it\n"
               "  plugins    the libraries the models came from\n"
               "  version    the version of libKalman\n"
               "  --json     machine-readable output\n"
               "  --plugin   load a plugin first: a file, or a name looked for in KALMAN_PLUGIN_PATH\n"
               "             (beside libKalman when unset)\n"
               "  --no-apps  do not load libKalmanApps\n"
               "  --no-settings  open a model file without the settings file beside it\n"
               "  --color    auto (the default: on a terminal, unless NO_COLOR is set), always or never\n";
        return code;
    }
}

int main(int argc, char **argv)
{
    bool json = false, apps = true, linearize = false, useSettings = true;
    std::size_t steps = 0;
    std::string color = "auto", output;
    std::vector<std::pair<std::string, double>> settings;
    std::vector<std::string> plugins, words;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--json") json = true;
        else if (a == "--no-apps") apps = false;
        else if (a == "--linearize") linearize = true;
        else if (a == "--no-settings") useSettings = false;
        else if (a == "-o" || a == "--output")
        {
            if (i + 1 >= argc) return Usage(2);
            output = argv[++i];
        }
        else if (a == "--steps")
        {
            if (i + 1 >= argc) return Usage(2);
            char *end = nullptr;
            const long value = std::strtol(argv[++i], &end, 10);
            if (!end || *end || value <= 0)
            {
                std::cerr << "kalman: --steps takes a positive number of samples\n";
                return 2;
            }
            steps = static_cast<std::size_t>(value);
        }
        else if (a == "--set")
        {
            if (i + 1 >= argc) return Usage(2);
            const std::string pair = argv[++i];
            const std::size_t eq = pair.find('=');
            char *end = nullptr;
            const double value = eq == std::string::npos ? 0. : std::strtod(pair.c_str() + eq + 1, &end);
            if (eq == std::string::npos || eq == 0 || !end || *end || end == pair.c_str() + eq + 1)
            {
                std::cerr << "kalman: --set takes NAME=VALUE with a number, not '" << pair << "'\n";
                return 2;
            }
            settings.emplace_back(pair.substr(0, eq), value);
        }
        else if (a == "--color" || a.rfind("--color=", 0) == 0)
        {
            if (a == "--color" && i + 1 >= argc) return Usage(2);
            color = a == "--color" ? argv[++i] : a.substr(8);
            if (color != "auto" && color != "always" && color != "never")
            {
                std::cerr << "kalman: --color takes auto, always or never, not '" << color << "'\n";
                return 2;
            }
        }
        else if (a == "--plugin")
        {
            if (i + 1 >= argc) return Usage(2);
            plugins.push_back(argv[++i]);
        }
        else if (a == "-h" || a == "--help") return Usage(0);
        else if (a == "--version") words = {"version"};
        else if (!a.empty() && a[0] == '-')
        {
            std::cerr << "kalman: unknown option '" << a << "'\n";
            return Usage(2);
        }
        else words.push_back(a);
    }
    const std::string command = words.empty() ? "models" : words[0];

    // Colours on a terminal only, and never when the user said no (https://no-color.org).
    if (color == "always") gColor = true;
    else if (color == "auto")
    {
        const char *no = std::getenv("NO_COLOR"), *term = std::getenv("TERM");
        gColor = isatty(STDOUT_FILENO) && !(no && *no) && !(term && std::string(term) == "dumb");
    }

    if (command == "version")
    {
        if (json) std::cout << "{\"version\": " << Quote(KALMAN_VERSION) << "}\n";
        else std::cout << "kalman " << KALMAN_VERSION << '\n';
        return 0;
    }

    auto &factory = Kalman::Factory::Instance();
    if (apps)
        if (const std::string path = Resolve("KalmanApps"); !path.empty())
        {
            try { factory.LoadPlugin(path); }
            catch (const std::exception &e) { std::cerr << "kalman: libKalmanApps not loaded: " << e.what() << '\n'; }
        }
    for (const auto &p : plugins)
    {
        const std::string path = Resolve(p);
        if (path.empty())
        {
            std::cerr << "kalman: no plugin '" << p << "' in";
            for (const auto &d : PluginDirs()) std::cerr << ' ' << d;
            std::cerr << '\n';
            return 1;
        }
        try { factory.LoadPlugin(path); }
        catch (const std::exception &e)
        {
            std::cerr << "kalman: " << e.what() << '\n';
            return 1;
        }
    }

    if (command == "models")
    {
        if (words.size() > 2) return Usage(2);
        const std::string pattern = words.size() == 2 ? words[1] : "";
        std::vector<Sheet> sheets;
        for (const auto &name : factory.Names())
            if (pattern.empty() || name.find(pattern) != std::string::npos) sheets.push_back(Read(name));
        if (json)
        {
            std::cout << "[";
            for (std::size_t i = 0; i < sheets.size(); ++i) std::cout << (i ? ",\n " : "") << Json(sheets[i], false);
            std::cout << "]\n";
            return 0;
        }
        if (sheets.empty())
        {
            std::cerr << (pattern.empty() ? "kalman: no model is registered (libKalmanApps not found; give --plugin)."
                                          : "kalman: no model matches '" + pattern + "'.")
                      << '\n';
            return 1;
        }
        Table(sheets);
        return 0;
    }

    if (command == "model")
    {
        if (words.size() != 2) return Usage(2);
        Sheet s;
        if (factory.Has(words[1])) s = Read(words[1]);
        else if (fs::is_regular_file(words[1]))
        {
            try
            {
                const auto opened = Kalman::IO::ModelFile::Open(words[1], useSettings);
                s = Read(opened.model->Name(), opened.model.get());
                s.file = words[1];
                s.kind = opened.kind + (opened.tables ? ", with tables" : "");
                s.settings = opened.settings;
                s.changed = opened.changed;
            }
            catch (const std::exception &e)
            {
                std::cerr << "kalman: " << words[1] << ": " << e.what() << '\n';
                return 1;
            }
        }
        else
        {
            std::cerr << "kalman: no model and no file '" << words[1] << "'. Run `kalman models` for the list.\n";
            return 1;
        }
        if (json) std::cout << Json(s, true) << '\n';
        else if (!s.error.empty()) std::cerr << "kalman: " << s.name << ": " << s.error << '\n';
        else Detail(s);
        return s.error.empty() ? 0 : 1;
    }

    if (command == "export")
    {
        if (words.size() != 2) return Usage(2);
        try
        {
            Kalman::IO::ModelFile::Options options;
            std::unique_ptr<Kalman::Model> model;
            if (factory.Has(words[1]))
            {
                model = factory.Create(words[1]);
                options.registered = words[1];
                // The plugin by name ("KalmanApps"), not by the path it has on this machine.
                std::string stem = fs::path(factory.Describe(words[1]).plugin).stem().string();
                if (stem.rfind("lib", 0) == 0) stem = stem.substr(3);
                options.plugin = stem.substr(0, stem.find('.'));
            }
            else if (fs::is_regular_file(words[1]))
            {
                const Kalman::IO::ModelFile file = Kalman::IO::ModelFile::Load(words[1]);
                model = std::move(Kalman::IO::ModelFile::Open(words[1], useSettings).model);   // its settings go in
                if (file.Kind() == "reference") options.registered = file.Spec()["base"].AsString("spec.base");
                options.origin = file.Document()["origin"];     // where it first came from
            }
            else
            {
                std::cerr << "kalman: no model and no file '" << words[1] << "'. Run `kalman models` for the list.\n";
                return 1;
            }
            for (const auto &setting : settings) model->SetParameter(setting.first, setting.second);
            options.linearize = linearize;
            options.steps = steps;
            const Kalman::IO::ModelFile file = Kalman::IO::ModelFile::Export(*model, options);
            if (!output.empty())
            {
                file.Save(output);
                std::cerr << "kalman: wrote " << output << " (" << file.Kind()
                          << (file.HasTables() ? ", with tables" : "") << ")\n";
            }
            else if (file.HasTables())
            {
                std::cerr << "kalman: this model has tables, so its JSON is not the whole model: give -o FILE"
                          << Kalman::IO::ModelFile::kExtension << '\n';
                return 1;
            }
            else std::cout << file.Text();
        }
        catch (const std::exception &e)
        {
            std::cerr << "kalman: " << e.what() << '\n';
            return 1;
        }
        return 0;
    }

    if (command == "settings")
    {
        if (words.size() != 2) return Usage(2);
        try
        {
            using Kalman::IO::ModelFile;
            const std::string beside = ModelFile::SettingsPath(words[1]);
            const bool existed = fs::is_regular_file(beside);
            const ModelFile file = ModelFile::Load(words[1]);
            // What is there is kept: the model is opened with it, then --set goes on top.
            const auto opened = ModelFile::Open(words[1]);
            for (const auto &setting : settings) opened.model->SetParameter(setting.first, setting.second);
            const std::string text = ModelFile::Settings(*opened.model, file.Identity()).Dump(2) + "\n";
            std::ofstream out(beside, std::ios::binary | std::ios::trunc);
            out << text;
            if (!out) throw std::runtime_error("cannot write '" + beside + "'.");
            out.close();
            const auto now = ModelFile::Open(words[1]);
            std::string what;
            for (const auto &c : now.changed) what += (what.empty() ? "" : ", ") + c;
            std::cerr << "kalman: " << (existed ? "updated " : "wrote ") << beside
                      << (what.empty() ? " (the values of the model file: edit it to change them)"
                                       : " (changes " + what + ")") << '\n';
        }
        catch (const std::exception &e)
        {
            std::cerr << "kalman: " << e.what() << '\n';
            return 1;
        }
        return 0;
    }

    if (command == "plugins")
    {
        if (words.size() != 1) return Usage(2);
        const auto loaded = factory.Plugins();
        if (json)
        {
            std::cout << "[";
            for (std::size_t i = 0; i < loaded.size(); ++i) std::cout << (i ? ", " : "") << Quote(loaded[i]);
            std::cout << "]\n";
            return 0;
        }
        for (const auto &p : loaded)
        {
            std::size_t count = 0;
            for (const auto &name : factory.Names())
                if (factory.Describe(name).plugin == p) ++count;
            std::cout << p << "  (" << count << " models)\n";
        }
        return 0;
    }

    std::cerr << "kalman: unknown command '" << command << "'\n";
    return Usage(2);
}
