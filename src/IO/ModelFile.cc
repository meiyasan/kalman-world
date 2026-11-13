/**
 * *********************************************
 *
 * \file ModelFile.cc
 * \brief Implementation of the Kalman::IO::ModelFile class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/IO/ModelFile.h"

#include "Kalman/Config.h"
#include "Kalman/Factory.h"
#include "Kalman/Models/Tabulated.h"

#include <algorithm>
#include <bit>
#include <cctype>
#include <cstring>
#include <functional>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Kalman::IO
{
    namespace
    {
        // The tables are little-endian float64, which is also how the hosts
        // this library is built for keep a double.
        static_assert(std::endian::native == std::endian::little, "ModelFile tables are little-endian.");

        [[noreturn]] void Bad(const std::string &message) { throw std::runtime_error("Kalman::IO::ModelFile: " + message); }

        bool IsIdentity(const Matrix &M)
        {
            if (M.Rows() != M.Cols()) return false;
            for (std::size_t i = 0; i < M.Rows(); ++i)
                for (std::size_t j = 0; j < M.Cols(); ++j)
                    if (M(i, j) != (i == j ? 1. : 0.)) return false;
            return true;
        }

        bool Same(const Matrix &A, const Matrix &B)
        {
            if (A.Rows() != B.Rows() || A.Cols() != B.Cols()) return false;
            for (std::size_t i = 0; i < A.Rows(); ++i)
                for (std::size_t j = 0; j < A.Cols(); ++j)
                    if (A(i, j) != B(i, j)) return false;
            return true;
        }

        std::string Lower(std::string s)
        {
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
            return s;
        }

        Json Components(const std::vector<Model::Component> &components)
        {
            Json out = Json::Array();
            for (const auto &c : components) out.Add(Json::Object().Set("name", c.name).Set("unit", c.unit));
            return out;
        }

        Json Names(const std::vector<Model::Component> &components)
        {
            Json out = Json::Array();
            for (const auto &c : components) out.Add(c.name);
            return out;
        }

        std::vector<Model::Component> ReadComponents(const Json &list, const std::string &what)
        {
            std::vector<Model::Component> out;
            if (!list.IsArray()) return out;
            for (const auto &item : list.Items(what))
            {
                Model::Component c;
                c.name = item["name"].AsString(what + ".name");
                if (item.Has("unit")) c.unit = item["unit"].AsString(what + ".unit");
                out.push_back(std::move(c));
            }
            return out;
        }

        std::string Pack(const std::vector<Matrix> &table)
        {
            std::string bytes;
            for (const auto &M : table)
                for (std::size_t i = 0; i < M.Rows(); ++i)
                    for (std::size_t j = 0; j < M.Cols(); ++j)
                    {
                        const double v = M(i, j);
                        bytes.append(reinterpret_cast<const char *>(&v), sizeof v);
                    }
            return bytes;
        }

        std::vector<Matrix> Unpack(const std::string &bytes, std::size_t steps, std::size_t rows, std::size_t cols,
                                   const std::string &what)
        {
            if (bytes.size() != steps * rows * cols * sizeof(double))
                Bad(what + " holds " + std::to_string(bytes.size()) + " bytes, its shape asks for "
                    + std::to_string(steps * rows * cols * sizeof(double)) + ".");
            std::vector<Matrix> table(steps, Matrix(rows, cols, 0.));
            const char *at = bytes.data();
            for (auto &M : table)
                for (std::size_t i = 0; i < rows; ++i)
                    for (std::size_t j = 0; j < cols; ++j)
                    {
                        std::memcpy(&M(i, j), at, sizeof(double));
                        at += sizeof(double);
                    }
            return table;
        }

        std::string TablePath(const std::string &which) { return "tables/" + which + ".f64"; }
    }

    // -------------------------------------------------------------------------
    //  Export
    // -------------------------------------------------------------------------
    ModelFile ModelFile::Export(const Model &model) { return Export(model, Options()); }

    ModelFile ModelFile::Export(const Model &model, const Options &options)
    {
        const StructureKind structure = model.Structure();
        const double t0 = model.T0();
        const std::size_t n = model.NState(), m = model.NMeasure(), l = model.NControl();

        std::string kind = "complete";
        if (options.linearize) kind = "linearization";
        else if (structure == StructureKind::Maps) kind = "reference";

        if (kind == "reference" && options.registered.empty())
            Bad("'" + model.Name() + "' is given by maps (code), which a file cannot hold. It can be written as a "
                "reference to its registered name (give Options::registered), or as a linearization "
                "(Options::linearize).");
        if (kind == "complete" && structure == StructureKind::TimeVarying && options.steps == 0)
            Bad("'" + model.Name() + "' has matrices that change with time: say over how many samples to "
                "tabulate them (Options::steps), or ask for the matrices at the first sample only "
                "(Options::linearize).");

        Json document = Json::Object();
        document.Set("format", kFormat).Set("format_version", kVersion).Set("kind", kind)
                .Set("generator", std::string("libKalman ") + KALMAN_VERSION);

        Json origin = Json::Object();
        if (!options.registered.empty()) origin.Set("model", options.registered);
        if (!options.plugin.empty()) origin.Set("plugin", options.plugin);
        origin.Set("title", model.Title()).Set("description", model.Description()).Set("version", model.Version());
        document.Set("origin", options.origin.IsObject() ? options.origin : origin);

        document.Set("structure", kind == "linearization" ? ToString(StructureKind::Matrices) : ToString(structure));
        document.Set("dynamics", ToString(model.Dynamics()));
        document.Set("state", Components(model.StateComponents()));
        document.Set("measurement", Components(model.MeasurementComponents()));

        // ---- the spec, in the form the JSON API takes ----------------------
        Json spec = Json::Object(), dynamics = Json::Object(), noise = Json::Object(), initial = Json::Object();
        ModelFile file;

        if (kind == "reference")
        {
            spec.Set("base", options.registered);
            Json parameters = Json::Object();
            for (const auto &name : model.ParameterNames()) parameters.Set(name, model.Parameter(name));
            dynamics.Set("params", parameters);
        }
        else
        {
            const bool continuous = kind == "complete" && !model.Generator().Empty() && !model.NoiseIntensity().Empty();
            spec.Set("base", continuous ? "continuous" : "matrices");
            spec.Set("name", options.registered.empty() ? model.Name() : options.registered);

            if (kind == "linearization")
            {
                const Vector x = options.at.Size() ? options.at : model.X0();
                if (x.Size() != n) Bad("the linearization state has " + std::to_string(x.Size()) + " components, the "
                                       "model has " + std::to_string(n) + ".");
                dynamics.Set("F", Json::From(model.Jacobian(x, Vector(l, 0.), t0)));
                dynamics.Set("H", Json::From(model.JacobianH(x, t0)));
                document.Set("linearized_at", Json::Object().Set("x", Json::From(x)).Set("t", t0));
            }
            else if (continuous)
            {
                dynamics.Set("A", Json::From(model.Generator()));
                dynamics.Set("H", Json::From(model.H(t0)));
                dynamics.Set("Qc", Json::From(model.NoiseIntensity()));
            }
            else
            {
                dynamics.Set("F", Json::From(model.F(t0)));
                dynamics.Set("H", Json::From(model.H(t0)));
            }
            if (l > 0)
                if (const Matrix B = model.B(t0); !B.Empty()) dynamics.Set("B", Json::From(B));
            if (const Matrix L = model.L(t0); !L.Empty() && !IsIdentity(L)) dynamics.Set("L", Json::From(L));
        }

        const bool continuousNoise = spec["base"].AsString() == "continuous";
        if (!continuousNoise) noise.Set("Q", Json::From(model.Q(t0)));
        noise.Set("R", Json::From(model.R(t0)));
        initial.Set("x0", Json::From(model.X0())).Set("P0", Json::From(model.P0()));

        spec.Set("fs", model.SampleRate()).Set("dynamics", dynamics).Set("noise", noise).Set("initial", initial);
        spec.Set("labels", Json::Object().Set("states", Names(model.StateComponents()))
                                         .Set("observables", Names(model.MeasurementComponents())));
        spec.Set("bindings", Json::Object().Set("engine", kind == "reference" ? Lower(ToString(model.DefaultEngine()))
                                                                               : std::string("linear")));
        document.Set("spec", spec);

        // ---- the tables: the matrices that change with time ------------------
        if (kind == "complete" && structure == StructureKind::TimeVarying)
        {
            const double dt = model.DeltaT();
            Json entries = Json::Object();
            const auto tabulate = [&](const char *which, const std::function<Matrix(double)> &at)
            {
                std::vector<Matrix> table;
                bool varies = false;
                for (std::size_t k = 0; k < options.steps; ++k)
                {
                    table.push_back(at(t0 + static_cast<double>(k) * dt));
                    varies = varies || !Same(table.back(), table.front());
                }
                if (!varies) return;   // constant: the spec already holds it
                Json shape = Json::Array();
                shape.Add(options.steps).Add(table.front().Rows()).Add(table.front().Cols());
                entries.Set(which, Json::Object().Set("file", TablePath(which)).Set("shape", shape));
                file.fTables.Set(TablePath(which), Pack(table));
            };
            tabulate("F", [&model](double t) { return model.F(t); });
            tabulate("H", [&model](double t) { return model.H(t); });
            tabulate("Q", [&model](double t) { return model.Q(t); });
            tabulate("R", [&model](double t) { return model.R(t); });
            document.Set("tables", Json::Object().Set("steps", options.steps).Set("t0", t0)
                                                 .Set("dtype", "float64, little-endian, row-major")
                                                 .Set("entries", entries));
        }
        (void) m;

        file.fDocument = std::move(document);
        return file;
    }

    // -------------------------------------------------------------------------
    //  Import
    // -------------------------------------------------------------------------
    std::string ModelFile::Kind() const
    { return fDocument.Has("kind") ? fDocument["kind"].AsString("kind") : std::string("complete"); }

    std::unique_ptr<Model> ModelFile::Import() const
    {
        // A bare spec (no envelope) is accepted: it is what the JSON API takes.
        const bool envelope = fDocument.Has("spec");
        if (envelope)
        {
            if (!fDocument.Has("format") || fDocument["format"].AsString("format") != kFormat)
                Bad(std::string("not a model file: \"format\" is not \"") + kFormat + "\".");
            const double version = fDocument["format_version"].AsNumber("format_version");
            if (version > kVersion)
                Bad("the file is of format version " + std::to_string(static_cast<int>(version)) + ", this library "
                    "reads up to " + std::to_string(kVersion) + ".");
        }
        const Json &spec = envelope ? fDocument["spec"] : fDocument;
        if (!spec.IsObject()) Bad("the document has no spec.");

        const std::string base = spec.Has("base") ? spec["base"].AsString("spec.base") : std::string("matrices");
        const Json &dynamics = spec["dynamics"], &noise = spec["noise"], &initial = spec["initial"];
        std::unique_ptr<Model> model;

        if (base == "matrices" || base == "continuous")
        {
            const bool continuous = dynamics.Has("A");
            if (continuous && dynamics.Has("F")) Bad("the spec gives both dynamics.F and dynamics.A.");
            if (!continuous && !dynamics.Has("F")) Bad("a matrices spec needs dynamics.F (or dynamics.A).");
            if (!dynamics.Has("H")) Bad("a matrices spec needs dynamics.H.");
            const Matrix first = (continuous ? dynamics["A"] : dynamics["F"]).AsMatrix(continuous ? "dynamics.A" : "dynamics.F");
            const Matrix H = dynamics["H"].AsMatrix("dynamics.H");
            const Matrix B = dynamics.Has("B") ? dynamics["B"].AsMatrix("dynamics.B") : Matrix();
            const std::size_t n = first.Rows();

            auto tabulated = std::make_unique<Models::Tabulated>(
                spec.Has("name") ? spec["name"].AsString("spec.name") : std::string("model"), n, H.Rows(), B.Cols());
            if (spec.Has("fs")) tabulated->SetSampleRate(spec["fs"].AsNumber("spec.fs"));
            tabulated->SetH(H);
            if (!B.Empty()) tabulated->SetB(B);
            if (dynamics.Has("L")) tabulated->SetL(dynamics["L"].AsMatrix("dynamics.L"));
            if (continuous)
            {
                tabulated->SetGenerator(first);
                if (dynamics.Has("Qc"))
                    tabulated->SetNoiseIntensity(dynamics["Qc"].AsMatrix("dynamics.Qc"),
                                                 dynamics.Has("Lc") ? dynamics["Lc"].AsMatrix("dynamics.Lc") : Matrix());
            }
            else tabulated->SetF(first);

            if (envelope)
                tabulated->SetComponents(ReadComponents(fDocument["state"], "state"),
                                         ReadComponents(fDocument["measurement"], "measurement"));

            if (envelope && fDocument.Has("tables"))
            {
                const Json &tables = fDocument["tables"];
                const std::size_t steps = static_cast<std::size_t>(tables["steps"].AsNumber("tables.steps"));
                if (tables.Has("t0")) tabulated->SetT0(tables["t0"].AsNumber("tables.t0"));
                for (const auto &entry : tables["entries"].Members("tables.entries"))
                {
                    const std::string where = entry.second["file"].AsString("tables.entries." + entry.first + ".file");
                    const auto &shape = entry.second["shape"].Items("tables.entries." + entry.first + ".shape");
                    if (shape.size() != 3 || static_cast<std::size_t>(shape[0].AsNumber("shape")) != steps)
                        Bad("the table of " + entry.first + " does not have the shape [steps, rows, columns].");
                    if (!fTables.Has(where))
                        Bad("the document refers to '" + where + "', which is not in the file. A model with tables "
                            "is one archive (" + kExtension + "); its JSON alone is not the whole model.");
                    tabulated->SetTable(entry.first, Unpack(fTables.Get(where), steps,
                                                           static_cast<std::size_t>(shape[1].AsNumber("shape")),
                                                           static_cast<std::size_t>(shape[2].AsNumber("shape")), where));
                }
            }
            model = std::move(tabulated);
        }
        else
        {
            auto &factory = Factory::Instance();
            if (!factory.Has(base))
            {
                std::string hint;
                if (envelope && fDocument["origin"].Has("plugin"))
                    hint = " It was written from the plugin '" + fDocument["origin"]["plugin"].AsString("origin.plugin")
                         + "': load it first.";
                Bad("the file refers to the model '" + base + "', which is not registered here." + hint
                    + " A reference holds a model's settings, not its code.");
            }
            model = factory.Create(base);
            if (dynamics.Has("params"))
                for (const auto &p : dynamics["params"].Members("dynamics.params"))
                    model->SetParameter(p.first, p.second.AsNumber("dynamics.params." + p.first));
            if (spec.Has("fs"))
                if (const double fs = spec["fs"].AsNumber("spec.fs"); fs != model->SampleRate()) model->SetSampleRate(fs);
        }

        if (noise.Has("Q")) model->SetQ(noise["Q"].AsMatrix("noise.Q"));
        if (noise.Has("R")) model->SetR(noise["R"].AsMatrix("noise.R"));
        if (initial.Has("x0")) model->SetInitialState(initial["x0"].AsVector("initial.x0"));
        if (initial.Has("P0")) model->SetInitialCovariance(initial["P0"].AsMatrix("initial.P0"));
        model->CheckShapes();
        return model;
    }

    // -------------------------------------------------------------------------
    //  Files
    // -------------------------------------------------------------------------
    std::string ModelFile::Bytes() const
    {
        Archive archive;
        archive.Set(kDocument, Text());     // first, so the readable part opens the file
        for (const auto &e : fTables.Entries()) archive.Set(e.first, e.second);
        return archive.Bytes();
    }

    void ModelFile::Save(const std::string &path) const
    {
        const std::string json = ".json";
        const bool plain = path.size() >= json.size() && path.compare(path.size() - json.size(), json.size(), json) == 0;
        if (plain && HasTables())
            Bad("this model has tables, so its JSON is not the whole model: write it as one archive ('"
                + std::string(kExtension) + "'), not as '" + path + "'.");
        const std::string bytes = plain ? Text() : Bytes();
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) Bad("cannot write '" + path + "'.");
    }

    ModelFile ModelFile::FromBytes(const std::string &bytes)
    {
        ModelFile file;
        if (Archive::Looks(bytes))
        {
            const Archive archive = Archive::FromBytes(bytes);
            file.fDocument = Json::Parse(archive.Get(kDocument));
            for (const auto &e : archive.Entries())
                if (e.first.rfind("tables/", 0) == 0) file.fTables.Set(e.first, e.second);
        }
        else file.fDocument = Json::Parse(bytes);
        if (!file.fDocument.IsObject()) Bad("the document is not a JSON object.");
        return file;
    }

    ModelFile ModelFile::Load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) Bad("cannot read '" + path + "'.");
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return FromBytes(buffer.str());
    }
}
