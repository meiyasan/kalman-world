/**
 * \file 21_fields_apps.cc
 * \brief Kalman::Apps::Weather::KolmogorovFlow / BarotropicFlow: a field on the ensemble engine.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Apps/Apps.h"
#include "Kalman/Filters/EnKF.h"
#include "Check.h"

#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

using namespace Kalman;

namespace
{
    double RMSE(const Vector &a, const Vector &b)
    {
        double e = 0.;
        for (std::size_t i = 0; i < a.Size(); ++i) e += (a(i) - b(i)) * (a(i) - b(i));
        return std::sqrt(e / static_cast<double>(a.Size()));
    }

    // A small flow: 16 x 16, 10 steps of 0.01 a cycle, a probe on every other cell of every 4th row.
    void Configure(Apps::Weather::KolmogorovFlow &flow, bool gradient = false)
    {
        flow.SetParameter("grid", 16.);
        flow.SetParameter("cycle", 10.);
        flow.SetParameter("dt", 0.01);
        flow.SetParameter("nu", 1. / 40.);
        flow.SetParameter("loc_radius", 4.);
        flow.SetParameter("gradient", gradient ? 1. : 0.);
        std::vector<double> probes;
        for (std::size_t i = 0; i < 16; i += 4)
            for (std::size_t j = 0; j < 16; j += 2) probes.push_back(static_cast<double>(i * 16 + j));
        flow.SetArray("probes", probes);
    }
}

int main()
{
    std::printf("21_fields_apps\n");
    auto &factory = Factory::Instance();
    CHECK(factory.Has("kolmogorov-flow") && factory.Has("barotropic-flow"), "fields: registered");
    {
        const Factory::Entry e = factory.Describe("kolmogorov-flow");
        CHECK(e.n == 4096 && e.m == 100 && e.engine == EngineKind::Ensemble,
              "kolmogorov-flow: 64 x 64 cells, 100 probes, the ensemble engine");
        CHECK_CLOSE(e.fs, 1. / (0.005 * 30.), 1e-12, "kolmogorov-flow: one sample per cycle of 30 x 0.005");
        Apps::Weather::BarotropicFlow b;
        CHECK_CLOSE(b.Parameter("beta"), 2. * 7.292e-5 * std::cos(50. * M_PI / 180.) / 6.371e6, 1e-25,
                    "barotropic-flow: beta at 50 N");
        CHECK(b.StateComponents()[0].unit == "1/s" && b.Array("forcing_re").empty(), "barotropic-flow: SI, unforced");
    }

    Apps::Weather::KolmogorovFlow flow;
    Configure(flow);
    CHECK(flow.NState() == 256 && flow.NMeasure() == 32, "resized: grid 16, 32 probes");
    CHECK(flow.StateComponents()[17].name == "w(1,1)" && flow.MeasurementComponents()[1].name == "w(0,2)",
          "components: w(row,column)");
    CHECK(flow.Array("forcing_re").size() == 256, "resized: the default forcing follows the grid");

    // A state on the attractor, from the solver's own random field.
    Field w0 = flow.Solver().Advance(flow.Solver().RandomIC(), 300);
    Vector truth(256);
    for (std::size_t i = 0; i < 256; ++i) truth(i) = w0.Data()[i];

    // --- h, H and the hooks against the one-member maps ------------------------
    {
        const Matrix H = flow.H();
        const Vector z = flow.h(truth);
        double d = 0.;
        const Vector Hx = H * truth;
        for (std::size_t k = 0; k < z.Size(); ++k) d = std::max(d, std::abs(Hx(k) - z(k)));
        CHECK(d == 0., "vorticity probes: H x is h(x)");

        Apps::Weather::KolmogorovFlow grad;
        Configure(grad, true);
        const Vector zg = grad.h(truth);
        const Vector Hg = grad.H() * truth;
        d = 0.;
        for (std::size_t k = 0; k < zg.Size(); ++k) d = std::max(d, std::abs(Hg(k) - zg(k)));
        CHECK(grad.NMeasure() == 64 && d < 1e-12, "gradient probes: two per probe, H x is h(x)");

        std::vector<Vector> members(5, truth), Y;
        for (std::size_t s = 0; s < members.size(); ++s) members[s](s) += 0.1 * static_cast<double>(s);
        std::vector<Vector> serial = members;
        for (Vector &x : serial) x = flow.f(x, Vector());
        flow.SetParameter("threads", 3.);
        flow.Forecast(flow, members, Vector(), 0.);
        double fd = 0.;
        for (std::size_t s = 0; s < members.size(); ++s) fd = std::max(fd, RMSE(members[s], serial[s]));
        CHECK(fd == 0., "forecast: the members in parallel are f, member by member");
        grad.Observe(grad, members, 0., Y);
        CHECK(Y.size() == 5 && RMSE(Y[2], grad.h(members[2])) == 0., "observe: h of every member");
    }

    // --- the tapers ------------------------------------------------------------
    {
        Matrix rxy, ryy;
        CHECK(flow.Localization(rxy, ryy) && rxy.Rows() == 256 && rxy.Cols() == 32 && ryy.Rows() == 32,
              "localization: (n, m) and (m, m)");
        const std::size_t p0 = flow.Probes()[0];
        bool unit = true, compact = true, bounded = true;
        for (std::size_t k = 0; k < 32; ++k) unit = unit && ryy(k, k) == 1.;
        for (std::size_t c = 0; c < 256; ++c)
        {
            bounded = bounded && rxy(c, 0) >= 0. && rxy(c, 0) <= 1.;
            if (flow.TorusDistance(c, p0) > 8.) compact = compact && rxy(c, 0) == 0.;
        }
        CHECK(unit && bounded && compact, "localization: 1 at the probe, in [0, 1], zero beyond 2 L");
        CHECK_CLOSE(flow.TorusDistance(0, 15), 1., 0., "localization: the grid is periodic");
        CHECK_CLOSE(Engines::GaspariCohn(4., 4.), 5. / 24., 1e-15, "Gaspari-Cohn: 5/24 at r = L");
    }

    // --- a twin experiment: the analysis beats a free run -----------------------
    {
        const std::size_t N = 16, T = 20;
        std::mt19937_64 random(7);
        std::normal_distribution<double> gauss(0., 1.);
        std::vector<Vector> members;
        Field w = w0;
        for (std::size_t s = 0; s < N; ++s)
        {
            w = flow.Solver().Advance(w, 40);              // climatology-like: other states on the attractor
            Vector x(256);
            for (std::size_t i = 0; i < 256; ++i) x(i) = w.Data()[i];
            members.push_back(x);
        }
        std::vector<Vector> zk, truths;
        Vector x = truth;
        for (std::size_t k = 0; k < T; ++k)
        {
            x = flow.f(x, Vector());
            truths.push_back(x);
            Vector z = flow.h(x);
            for (std::size_t j = 0; j < z.Size(); ++j) z(j) += 0.3 * gauss(random);
            if (k == 4) for (std::size_t j = 0; j < z.Size(); j += 3) z(j) = std::nan("");
            zk.push_back(z);
        }

        Engines::Ensemble enkf(N, 1.05, 3), free(N, 1., 3);
        enkf.SetMembers(members);
        free.SetMembers(members);
        Filters::EnKF assimilate(&flow, &enkf), drift(&flow, &free);
        const auto a = assimilate.Cycle(zk);
        const auto b = drift.Cycle(zk, false);
        double ea = 0., eb = 0.;
        for (std::size_t k = T - 5; k < T; ++k) { ea += RMSE(a.analysis[k], truths[k]); eb += RMSE(b.analysis[k], truths[k]); }
        std::printf("    last 5 cycles: analysis RMSE %.3f, free run %.3f, spread %.3f\n", ea / 5., eb / 5., a.spread.back());
        CHECK(ea < 0.5 * eb, "cycle: the analysis error is under half the free run's");
        CHECK(a.spread.back() > 0.02 && a.spread.back() < 3. * ea / 5., "cycle: the spread is of the error's size");

        // The same run through Kalman::Filter, with the draws handed in: Cycle() and
        // Forward() take the same steps.
        Engines::Ensemble e1(N, 1.05), e2(N, 1.05);
        e1.SetMembers(members);
        e2.SetMembers(members);
        std::vector<double> normals;
        for (std::size_t i = 0; i < T * 32 * N; ++i) normals.push_back(gauss(random));
        e1.SetNormals(normals);
        e2.SetNormals(normals);
        Filters::EnKF c1(&flow, &e1);
        const auto cycles = c1.Cycle(zk);
        Filter f2(&flow, &e2);
        const Record rec = f2.Forward(zk);
        double d = 0.;
        for (std::size_t k = 0; k < T; ++k)
            for (std::size_t i = 0; i < 256; ++i) d = std::max(d, std::abs(rec.State(k)(i) - cycles.analysis[k](i)));
        CHECK(d < 1e-12, "draws handed in: Cycle() and Forward() agree");
        // Every analysis takes (p, N): 32 channels, 21 at the cycle where 11 were silent.
        CHECK(e1.NormalsLeft() == 11 * N && e2.NormalsLeft() == 11 * N,
              "draws handed in: one (p, N) block per analysis, p the channels present");
        bool threw = false;
        Engines::Ensemble short_(N, 1.);
        short_.SetMembers(members);
        short_.SetNormals(std::vector<double>(10, 0.));
        try { short_.Analyze(flow, zk[0], 0.); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "draws handed in: running out throws");
    }

    // --- shapes ---------------------------------------------------------------
    {
        Apps::Weather::KolmogorovFlow bad;
        Configure(bad);
        bad.SetArray("probes", {0., 300.});
        bool threw = false;
        try { bad.CheckShapes(); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "shapes: a probe outside the grid is refused");
        bad.SetArray("probes", {0., 3.});
        bad.SetArray("process_variance", {1., 2., 3.});
        threw = false;
        try { bad.CheckShapes(); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "shapes: a process_variance of another length is refused");
    }
    return Check::Summary("21_fields_apps");
}
