/**
 * \file 17_modelfile.cc
 * \brief Kalman::IO::ModelFile: a model written to one file and read back.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Kalman;

KALMAN_REGISTER_MODEL(Fixture::Pendulum, "file-pendulum", 1., 0.1, 50.);

namespace
{
    // A linear model whose measurement noise grows with time.
    class Fading : public Models::Linear
    {
        public:
            Fading() : Models::Linear("fading", 2, 1)
            {
                SetSampleRate(10.);
                Matrix F = Matrix::Identity(2);
                F(0, 1) = 0.1;
                SetF(F);
                Matrix H(1, 2, 0.);
                H(0, 0) = 1.;
                SetH(H);
                Matrix Q(2, 2, 0.);
                Q(0, 0) = 1e-4;
                Q(1, 1) = 1e-3;
                SetQ(Q);
                SetStructure(StructureKind::TimeVarying);
            }
            Matrix R(double t) const override { return Matrix(1, 1, 0.04 * (1. + t)); }
    };

    std::vector<Vector> Measurements(std::size_t N)
    {
        std::vector<Vector> zk;
        for (std::size_t k = 0; k < N; ++k)
        {
            Vector z(1);
            z(0) = 0.1 * static_cast<double>(k) + 0.3 * std::sin(1.7 * static_cast<double>(k));
            zk.push_back(z);
        }
        return zk;
    }

    // Largest difference between the estimates of two models on the same data.
    double Apart(Model &a, Model &b, std::size_t N = 60)
    {
        const std::vector<Vector> zk = Measurements(N);
        Filter fa(&a), fb(&b);
        const IO::Record ra = fa.Forward(zk), rb = fb.Forward(zk);
        double worst = 0.;
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t i = 0; i < a.NState(); ++i)
            {
                worst = std::max(worst, std::fabs(ra.State(k)(i) - rb.State(k)(i)));
                worst = std::max(worst, std::fabs(ra.Covariance(k)(i, i) - rb.Covariance(k)(i, i)));
            }
        return worst;
    }

    template <class F>
    std::string Thrown(F &&body)
    {
        try { body(); }
        catch (const std::exception &e) { return e.what(); }
        return "";
    }

    bool Mentions(const std::string &text, const char *word) { return text.find(word) != std::string::npos; }
}

int main()
{
    std::printf("17_modelfile\n");
    using IO::ModelFile;

    // --- JSON: what goes out comes back, to the bit ----------------------------
    {
        IO::Json j = IO::Json::Object();
        j.Set("pi", 3.141592653589793).Set("tiny", 1e-300).Set("third", 1. / 3.).Set("count", 3)
         .Set("text", "a \"quoted\"\nline").Set("flag", true);
        const IO::Json back = IO::Json::Parse(j.Dump());
        CHECK(back["pi"].AsNumber() == 3.141592653589793 && back["tiny"].AsNumber() == 1e-300
              && back["third"].AsNumber() == 1. / 3., "json: doubles read back unchanged");
        CHECK(back["text"].AsString() == "a \"quoted\"\nline" && back["flag"].AsBoolean(), "json: strings and booleans");
        CHECK(Mentions(j.Dump(), "\"count\": 3,"), "json: a count is written without a decimal point");
        CHECK(Mentions(Thrown([] { IO::Json::Parse("{\"a\": [1, 2}"); }), "line 1"), "json: an error gives its position");
    }

    // --- archive: read back, and damage is noticed -----------------------------
    {
        IO::Archive archive;
        archive.Set("model.json", "{}");
        archive.Set("tables/R.f64", std::string(64, '\x01'));
        std::string bytes = archive.Bytes();
        const IO::Archive back = IO::Archive::FromBytes(bytes);
        CHECK(back.Entries().size() == 2 && back.Get("tables/R.f64").size() == 64, "archive: entries read back");
        bytes[bytes.find(std::string(8, '\x01'))] = '\x02';
        CHECK(Mentions(Thrown([&bytes] { IO::Archive::FromBytes(bytes); }), "checksum"), "archive: a changed byte fails its checksum");
    }

    // --- a model of constant matrices: complete ---------------------------------
    {
        Fixture::ConstantVelocity model(1, 100.);
        const ModelFile file = ModelFile::Export(model);
        CHECK(file.Kind() == "complete" && !file.HasTables(), "matrices: written whole, no tables");
        CHECK(file.Spec()["base"].AsString() == "matrices", "matrices: the spec is a matrices spec");
        const std::unique_ptr<Model> back = ModelFile::FromBytes(file.Bytes()).Import();
        CHECK(back->Structure() == StructureKind::Matrices && back->NState() == model.NState(), "matrices: read back as matrices");
        CHECK_CLOSE(Apart(model, *back), 0., 0., "matrices: the same estimates, to the bit");
        const std::unique_ptr<Model> plain = ModelFile::FromBytes(file.Text()).Import();
        CHECK_CLOSE(Apart(model, *plain), 0., 0., "matrices: the same from the plain JSON");
        CHECK(ModelFile::Export(*back).Spec().Dump() == file.Spec().Dump(), "matrices: writing it again gives the same spec");
    }

    // --- matrices that change with time: complete, with tables -------------------
    {
        Fading model;
        CHECK(Mentions(Thrown([&model] { ModelFile::Export(model); }), "how many samples"), "time-varying: asks for the samples");
        ModelFile::Options options;
        options.steps = 60;
        const ModelFile file = ModelFile::Export(model, options);
        CHECK(file.Kind() == "complete" && file.HasTables(), "time-varying: written whole, with tables");
        CHECK(file.Document()["tables"]["entries"].Has("R") && !file.Document()["tables"]["entries"].Has("F"),
              "time-varying: only the matrices that change are tabulated");
        const std::unique_ptr<Model> back = ModelFile::FromBytes(file.Bytes()).Import();
        CHECK(back->Structure() == StructureKind::TimeVarying, "time-varying: read back as time-varying");
        // Not to the bit: the table is taken at t0 + k dt, the filter's clock adds dt k times.
        CHECK_CLOSE(Apart(model, *back), 0., 1e-12, "time-varying: the same estimates, to rounding");
        CHECK(Mentions(Thrown([&file] { file.Save("fading.kalman.json"); }), "not the whole model"),
              "time-varying: refuses to be written as JSON alone");
        CHECK(Mentions(Thrown([&file] { ModelFile::FromBytes(file.Text()).Import(); }), "not in the file"),
              "time-varying: the JSON without its tables is refused");
    }

    // --- a model given by code: a reference, or a linearization ------------------
    {
        Fixture::Pendulum model(1., 0.1, 50.);
        CHECK(Mentions(Thrown([&model] { ModelFile::Export(model); }), "registered name"), "maps: says what it can be written as");
        ModelFile::Options options;
        options.registered = "file-pendulum";
        options.plugin = "SomePlugin";
        const ModelFile file = ModelFile::Export(model, options);
        CHECK(file.Kind() == "reference" && file.Spec()["base"].AsString() == "file-pendulum", "maps: written as a reference");
        const std::unique_ptr<Model> back = ModelFile::FromBytes(file.Bytes()).Import();
        CHECK_CLOSE(Apart(model, *back), 0., 0., "maps: the reference gives the same estimates");

        std::string text = file.Text();
        text.replace(text.find("\"base\": \"file-pendulum\""), 23, "\"base\": \"not-installed\"");
        const std::string why = Thrown([&text] { ModelFile::FromBytes(text).Import(); });
        CHECK(Mentions(why, "not registered here") && Mentions(why, "SomePlugin"), "maps: a missing model names its plugin");

        ModelFile::Options linear;
        linear.linearize = true;
        const ModelFile jacobians = ModelFile::Export(model, linear);
        CHECK(jacobians.Kind() == "linearization" && jacobians.Document().Has("linearized_at"), "maps: linearization on request");
        const std::unique_ptr<Model> tangent = jacobians.Import();
        const Matrix F = model.Jacobian(model.X0(), Vector(), model.T0());
        CHECK_CLOSE(tangent->F(0.)(1, 0), F(1, 0), 0., "maps: the linearization holds the Jacobian at x0");
    }

    // --- files -----------------------------------------------------------------
    {
        Fixture::ConstantVelocity model(1, 100.);
        const ModelFile file = ModelFile::Export(model);
        const std::string path = "17_modelfile_test.kalman";
        file.Save(path);
        const std::unique_ptr<Model> back = ModelFile::Load(path).Import();
        CHECK_CLOSE(Apart(model, *back), 0., 0., "file: saved and loaded");
        std::remove(path.c_str());
        CHECK(Mentions(Thrown([] { ModelFile::FromBytes("{\"format\": \"other\", \"spec\": {}}").Import(); }), "not a model file"),
              "file: another format is refused");
        CHECK(Mentions(Thrown([] { ModelFile::FromBytes("{\"format\": \"kalman-model\", \"format_version\": 99, \"spec\": {}}").Import(); }),
                       "reads up to"), "file: a later version is refused");
    }

    return Check::Summary("17_modelfile");
}
