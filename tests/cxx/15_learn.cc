/**
 * \file 15_learn.cc
 * \brief libKalmanLearn: the ONNX and TorchScript backends and the libTorch field solver, with or without their runtime.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Learn/Learn.h"
#include "Check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include <stdexcept>

using namespace Kalman;
using namespace Kalman::Routers;

int main()
{
    // Which runtimes this build carries. The classes exist and link either way:
    // an absent runtime is an answer at run time, not a link error.
    std::printf("  [backends: ONNX=%s  Torch=%s]\n",
                NN::Onnx::HasRuntime() ? "yes" : "no",
                NN::Torch::HasRuntime() ? "yes" : "no");

    // --- the optional backends behave the same way whether present or not -----
    {
        // Asking for a specific artifact that cannot be executed is always loud,
        // whether the file is missing or the runtime is.
        bool threw = false;
        try { NN::Onnx missing("definitely-not-here.onnx"); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "NN::Onnx: a missing graph or absent runtime throws");

        threw = false;
        try { NN::Torch missing("definitely-not-here.pt"); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "NN::Torch: a missing module or absent runtime throws");

        NN::Onnx idleOnnx;
        NN::Torch idleTorch;
        CHECK(!idleOnnx.IsReady(), "NN::Onnx: default-constructed is not ready");
        CHECK(!idleTorch.IsReady(), "NN::Torch: default-constructed is not ready");

        // ... and an unready model leaves the router on its Bayesian fallback.
        Neural router(&idleOnnx, 5);
        CHECK(!router.IsReady(), "Neural: an unready model means an unready router");
        const Matrix nis(3, 2, 1.);
        Vector c{0.5, 0.5}, L{3.0, 1.0};
        CHECK_CLOSE(router.Route(nis, c, L)(0), 0.75, 1e-12,
                    "Neural: unready model still routes, via Bayes");
    }

    // --- the libTorch field solver ---------------------------------------------
    std::printf("  [field solver: libTorch=%s]\n", Learn::Solvers::Vorticity::HasRuntime() ? "yes" : "no");
    {
        // w = cos(x) cos(y) is an eigenfunction of the Laplacian, so its stream
        // function is proportional to it and advection vanishes: the field only
        // decays, as exp(-2 nu t). An exact solution to step against.
        const std::size_t n = 16;
        const double pi = 3.14159265358979323846, nu = 0.05, dt = 0.01;
        const std::size_t steps = 50;
        Learn::Solvers::Vorticity::Grid grid;
        grid.ny = grid.nx = n;
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i)
            {
                const double kx = i < n / 2 ? double(i) : double(i) - double(n);
                const double ky = j < n / 2 ? double(j) : double(j) - double(n);
                const double k2 = kx * kx + ky * ky;
                grid.kx.push_back(kx);
                grid.ky.push_back(ky);
                grid.k2.push_back(k2);
                grid.k2inv.push_back(k2 > 0. ? 1. / k2 : 0.);
                grid.dealias.push_back(std::fabs(kx) <= double(n / 3) && std::fabs(ky) <= double(n / 3) ? 1. : 0.);
            }
        // Two fields in one call: the mode, and twice the mode.
        std::vector<double> w(2 * n * n), out(2 * n * n);
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = 0; i < n; ++i)
            {
                const double value = std::cos(2. * pi * i / n) * std::cos(2. * pi * j / n);
                w[j * n + i] = value;
                w[n * n + j * n + i] = 2. * value;
            }
        Learn::Solvers::Vorticity::Tableau rk4;
        rk4.c = {0., 0.5, 0.5, 1.};
        rk4.A = {0., 0., 0., 0.,   0.5, 0., 0., 0.,   0., 0.5, 0., 0.,   0., 0., 1., 0.};
        rk4.b = {1. / 6., 1. / 3., 1. / 3., 1. / 6.};

        if (Learn::Solvers::Vorticity::HasRuntime())
        {
            const Learn::Solvers::Vorticity solver(grid, nu, 0., {}, {}, "cpu");
            CHECK(solver.Device() == "cpu", "Vorticity: runs on the device asked for");
            solver.Advance(w.data(), 2, steps, dt, rk4, out.data());
            const double decay = std::exp(-2. * nu * dt * double(steps));
            double worst = 0.;
            for (std::size_t p = 0; p < 2 * n * n; ++p) worst = std::max(worst, std::fabs(out[p] - decay * w[p]));
            CHECK(worst < 1e-10, "Vorticity: a Laplacian eigenmode decays as exp(-2 nu t), both fields of the batch");

            const Learn::Solvers::Vorticity fallback(grid, nu, 0., {}, {}, "no-such-device");
            CHECK(fallback.Device() == "cpu", "Vorticity: an unavailable device falls back to the CPU");

            bool threw = false;
            Learn::Solvers::Vorticity::Tableau broken = rk4;
            broken.A.pop_back();
            try { solver.Advance(w.data(), 2, 1, dt, broken, out.data()); }
            catch (const std::invalid_argument &) { threw = true; }
            CHECK(threw, "Vorticity: a malformed tableau is refused");
        }
        else
        {
            bool threw = false;
            try { Learn::Solvers::Vorticity solver(grid, nu, 0., {}, {}, "cpu"); }
            catch (const std::runtime_error &) { threw = true; }
            CHECK(threw, "Vorticity: without libTorch the constructor says so");
        }
    }

    return Check::Summary("15_learn");
}
