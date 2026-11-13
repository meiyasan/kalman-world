/**
 * *********************************************
 *
 * \file KolmogorovFlow.cc
 * \brief Implementation of the Kalman::Apps::Weather::KolmogorovFlow class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Weather/KolmogorovFlow.h"

#include <Kalman/FFT.h>
#include <Kalman/Factory.h>
#include <Kalman/Integrator.h>

#include <algorithm>
#include <cmath>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace Kalman::Apps::Weather
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;

        std::string Cell(const char *what, std::size_t flat, std::size_t grid)
        {
            return std::string(what) + "(" + std::to_string(flat / grid) + "," + std::to_string(flat % grid) + ")";
        }

        // The registered model's probes: a regular 10 x 10 lattice (the Python
        // class draws its own and pushes them).
        std::vector<double> Lattice(std::size_t grid)
        {
            std::vector<double> out;
            const std::size_t side = std::min<std::size_t>(10, grid);
            for (std::size_t a = 0; a < side; ++a)
                for (std::size_t b = 0; b < side; ++b)
                    out.push_back(static_cast<double>((a * grid / side) * grid + b * grid / side));
            return out;
        }

        // The transform of F = chi cos(kf y), as NS2D builds it (y = Ly i / ny).
        void KolmogorovForcing(std::size_t grid, double ly, double chi, double kf,
                               std::vector<double> &re, std::vector<double> &im)
        {
            Field F(grid, grid);
            for (std::size_t i = 0; i < grid; ++i)
            {
                const double y = ly * static_cast<double>(i) / static_cast<double>(grid);
                for (std::size_t j = 0; j < grid; ++j) F(i, j) = chi * std::cos(kf * y);
            }
            const SpectralField Fh = FFT::Forward(F);
            re.resize(Fh.Size());
            im.resize(Fh.Size());
            for (std::size_t i = 0; i < Fh.Size(); ++i)
            {
                re[i] = Fh.Data()[i].real();
                im[i] = Fh.Data()[i].imag();
            }
        }

        template <typename Job>
        void Parallel(std::size_t jobs, std::size_t workers, Job job)
        {
            if (workers <= 1)
            {
                for (std::size_t s = 0; s < jobs; ++s) job(s);
                return;
            }
            std::vector<std::exception_ptr> failed(workers);
            std::vector<std::thread> pool;
            pool.reserve(workers - 1);
            const auto run = [&](std::size_t w) {
                try { for (std::size_t s = w; s < jobs; s += workers) job(s); }
                catch (...) { failed[w] = std::current_exception(); }
            };
            for (std::size_t w = 1; w < workers; ++w) pool.emplace_back(run, w);
            run(0);
            for (std::thread &t : pool) t.join();
            for (const auto &e : failed)
                if (e) std::rethrow_exception(e);
        }
    }

    KolmogorovFlow::KolmogorovFlow() : KolmogorovFlow("kolmogorov-flow") {}

    KolmogorovFlow::KolmogorovFlow(const std::string &name)
        : Kalman::Apps::Model(name, 64 * 64, 100)
    {
        SetDefaultEngine(EngineKind::Ensemble);
        // F is the identity placeholder and H the linear observation map: both are
        // what this class states, not differences of f and h.
        SetDifferentiator(DifferentiatorKind::Analytic);

        DeclareParameter("grid", 64.);
        DeclareParameter("cycle", 30.);
        DeclareParameter("integrator", static_cast<double>(IntegratorKind::RK4));
        DeclareParameter("loc_radius", 8.);
        DeclareParameter("gradient", 0.);
        DeclareParameter("threads", 0.);
        DeclareArray("probes", Lattice(64));
        DeclareArray("process_variance", {});
        Configure(1. / 150., 0., 0.005, 30, 2. * kPi, 2. * kPi, true);
    }

    void KolmogorovFlow::Configure(double nu, double beta, double dt, std::size_t cycle, double ly, double lx,
                                   bool forcing)
    {
        fParameters["nu"] = nu;
        fParameters["beta"] = beta;
        fParameters["dt"] = dt;
        fParameters["cycle"] = static_cast<double>(cycle);
        fParameters["length_y"] = ly;
        fParameters["length_x"] = lx;
        std::vector<double> re, im;
        if (forcing) KolmogorovForcing(static_cast<std::size_t>(Parameter("grid")), ly, 2., 4., re, im);
        fArrays["forcing_re"] = re;
        fArrays["forcing_im"] = im;
        fGrid = 0;                                        // state everything again (the unit may differ)
        ParametersChanged();
        SetSampleRate(1. / (dt * static_cast<double>(cycle)));
    }

    void KolmogorovFlow::ParametersChanged()
    {
        const double gridValue = Parameter("grid");
        if (!(gridValue >= 4.) || gridValue != std::floor(gridValue))
            throw std::invalid_argument("KolmogorovFlow: grid must be a whole number of at least 4 cells.");
        const std::size_t grid = static_cast<std::size_t>(gridValue);
        const std::size_t n = grid * grid;

        // A grid of another size: the default probes and forcing follow it, arrays
        // that were set are the caller's to set again.
        if (fGrid != 0 && grid != fGrid)
        {
            const std::vector<double> lattice = Lattice(fGrid);
            if (Array("probes") == lattice) fArrays["probes"] = Lattice(grid);
            std::vector<double> re, im;
            KolmogorovForcing(fGrid, Parameter("length_y"), 2., 4., re, im);
            if (Array("forcing_re") == re && Array("forcing_im") == im)
            {
                KolmogorovForcing(grid, Parameter("length_y"), 2., 4., re, im);
                fArrays["forcing_re"] = re;
                fArrays["forcing_im"] = im;
            }
        }

        fCycle = static_cast<std::size_t>(std::max(0., Parameter("cycle")));
        fGradient = Parameter("gradient") != 0.;
        fLocRadius = Parameter("loc_radius");
        fThreads = static_cast<std::size_t>(std::max(0., Parameter("threads")));

        fProbes.clear();
        for (const double p : Array("probes"))
            fProbes.push_back(p >= 0. ? static_cast<std::size_t>(p) : n);   // out of range: CheckShapes() says so
        const std::size_t m = std::max<std::size_t>(fProbes.size() * (fGradient ? 2 : 1), 1);

        const bool resized = grid != fGrid || m != NMeasure();
        if (grid != fGrid)
        {
            fGrid = grid;
            Allocate(n, m, 0);
            // F and Q are stated by F() and Q(); keep no n x n copy of either.
            fF = Matrix();
            fQ = Matrix();
            std::vector<Component> state;
            state.reserve(n);
            for (std::size_t i = 0; i < n; ++i) state.push_back({Cell("w", i, grid), fUnit});
            DeclareState(std::move(state));
        }
        else if (m != NMeasure()) ResizeMeasurement(m);
        if (resized)
        {
            Kalman::Model::fH = Matrix();
            fR = Matrix::Identity(m);
            fR *= 0.3 * 0.3;                              // sig_o = 0.3, the Python default
        }
        if (fProbes.size() * (fGradient ? 2 : 1) == m)
        {
            std::vector<Component> names;
            for (const std::size_t p : fProbes) names.push_back({Cell(fGradient ? "dw/dx" : "w", p, grid), fGradient ? "" : fUnit});
            if (fGradient)
                for (const std::size_t p : fProbes) names.push_back({Cell("dw/dy", p, grid), ""});
            DeclareMeasurement(std::move(names));
        }

        const std::vector<double> &variance = Array("process_variance");
        fVariance = Vector(n, 0.);
        if (variance.size() == 1) fVariance = Vector(n, variance[0]);
        else if (variance.size() == n)
            for (std::size_t i = 0; i < n; ++i) fVariance(i) = variance[i];

        const std::vector<double> &re = Array("forcing_re");
        const std::vector<double> &im = Array("forcing_im");
        fSolver.reset();
        if (re.size() == im.size() && (re.empty() || re.size() == n))
        {
            SpectralField forcing;
            if (!re.empty())
            {
                forcing = SpectralField(grid, grid);
                for (std::size_t i = 0; i < n; ++i) forcing.Data()[i] = {re[i], im[i]};
            }
            const auto kind = static_cast<IntegratorKind>(static_cast<int>(Parameter("integrator")));
            fSolver = std::make_shared<const Kalman::Solvers::Vorticity>(
                grid, grid, Parameter("dt"), Parameter("length_y"), Parameter("length_x"),
                Parameter("nu"), Parameter("beta"), forcing,
                std::shared_ptr<Integrator>(Integrators::Make(kind)));
        }

        std::lock_guard<std::mutex> lock(fCacheMutex);
        fObserve = Matrix();
        fRhoXY = fRhoYY = Matrix();
    }

    void KolmogorovFlow::CheckShapes() const
    {
        const std::size_t n = NState();
        for (const std::size_t p : fProbes)
            if (p >= n)
                throw std::runtime_error("KolmogorovFlow: a probe lies outside the " + std::to_string(fGrid) + " x "
                                         + std::to_string(fGrid) + " grid.");
        if (fProbes.empty()) throw std::runtime_error("KolmogorovFlow: no probes.");
        const std::size_t variances = Array("process_variance").size();
        if (variances > 1 && variances != n)
            throw std::runtime_error("KolmogorovFlow: process_variance has " + std::to_string(variances)
                                     + " values; give n = " + std::to_string(n) + ", one, or none.");
        if (!fSolver)
            throw std::runtime_error("KolmogorovFlow: forcing_re and forcing_im must both be empty or n = "
                                     + std::to_string(n) + " long.");
        const Matrix Rm = R(T0());
        if (Rm.Rows() != NMeasure() || Rm.Cols() != NMeasure())
            throw std::runtime_error("KolmogorovFlow: R does not match the measurement.");
    }

    std::size_t KolmogorovFlow::Workers(std::size_t jobs) const
    {
        std::size_t workers = fThreads ? fThreads : std::max(1u, std::thread::hardware_concurrency());
        return std::max<std::size_t>(1, std::min(workers, jobs));
    }

    Field KolmogorovFlow::AsField(const Vector &x) const
    {
        if (x.Size() != NState())
            throw std::invalid_argument("KolmogorovFlow: a state of " + std::to_string(x.Size()) + " values, expected "
                                        + std::to_string(NState()) + ".");
        Field w(fGrid, fGrid);
        std::copy(x.Data(), x.Data() + x.Size(), w.Data());
        return w;
    }

    Vector KolmogorovFlow::f(const Vector &x, const Vector &, double t) const
    {
        if (!fSolver) CheckShapes();
        const Field next = fSolver->Advance(AsField(x), fCycle, t);
        Vector out(NState());
        std::copy(next.Data(), next.Data() + next.Size(), out.Data());
        return out;
    }

    void KolmogorovFlow::Measure(const Field &w, Vector &z) const
    {
        const std::size_t p = fProbes.size();
        z = Vector(NMeasure());
        if (!fGradient)
        {
            for (std::size_t k = 0; k < p; ++k) z(k) = w.Data()[fProbes[k]];
            return;
        }
        Field wx, wy;
        fSolver->Gradient(w, wx, wy);
        for (std::size_t k = 0; k < p; ++k)
        {
            z(k) = wx.Data()[fProbes[k]];
            z(p + k) = wy.Data()[fProbes[k]];
        }
    }

    Vector KolmogorovFlow::h(const Vector &x, double) const
    {
        CheckShapes();
        Vector z;
        Measure(AsField(x), z);
        return z;
    }

    Matrix KolmogorovFlow::F(double) const { return Matrix::Identity(NState()); }

    Matrix KolmogorovFlow::Q(double) const { return Matrix::Diagonal(fVariance); }

    Matrix KolmogorovFlow::H(double) const
    {
        CheckShapes();
        std::lock_guard<std::mutex> lock(fCacheMutex);
        if (!fObserve.Empty()) return fObserve;
        const std::size_t n = NState(), p = fProbes.size(), g = fGrid;
        Matrix H(NMeasure(), n, 0.);
        if (!fGradient)
            for (std::size_t k = 0; k < p; ++k) H(k, fProbes[k]) = 1.;
        else
        {
            // The derivative is a convolution: its kernel is the gradient of a unit
            // impulse at cell 0, read at the offset between the probe and each cell.
            Field delta(g, g, 0.), kx, ky;
            delta(0, 0) = 1.;
            fSolver->Gradient(delta, kx, ky);
            for (std::size_t k = 0; k < p; ++k)
            {
                const std::size_t pi = fProbes[k] / g, pj = fProbes[k] % g;
                for (std::size_t c = 0; c < n; ++c)
                {
                    const std::size_t ci = c / g, cj = c % g;
                    const std::size_t di = (pi + g - ci) % g, dj = (pj + g - cj) % g;
                    H(k, c) = kx(di, dj);
                    H(p + k, c) = ky(di, dj);
                }
            }
        }
        fObserve = H;
        return H;
    }

    void KolmogorovFlow::Forecast(const Kalman::Model &, std::vector<Vector> &members, const Vector &, double t) const
    {
        CheckShapes();
        Parallel(members.size(), Workers(members.size()), [&](std::size_t s) {
            const Field next = fSolver->Advance(AsField(members[s]), fCycle, t);
            std::copy(next.Data(), next.Data() + next.Size(), members[s].Data());
        });
    }

    void KolmogorovFlow::Observe(const Kalman::Model &, const std::vector<Vector> &members, double,
                                 std::vector<Vector> &out) const
    {
        CheckShapes();
        out.assign(members.size(), Vector());
        if (!fGradient)
        {
            for (std::size_t s = 0; s < members.size(); ++s)
            {
                if (members[s].Size() != NState()) throw std::invalid_argument("KolmogorovFlow: a member of the wrong size.");
                out[s] = Vector(NMeasure());
                for (std::size_t k = 0; k < fProbes.size(); ++k) out[s](k) = members[s](fProbes[k]);
            }
            return;
        }
        Parallel(members.size(), Workers(members.size()), [&](std::size_t s) { Measure(AsField(members[s]), out[s]); });
    }

    double KolmogorovFlow::TorusDistance(std::size_t a, std::size_t b) const
    {
        const auto wrap = [g = static_cast<double>(fGrid)](double p, double q) {
            const double d = std::abs(p - q);
            return std::min(d, g - d);
        };
        const double ay = static_cast<double>(a / fGrid), ax = static_cast<double>(a % fGrid);
        const double by = static_cast<double>(b / fGrid), bx = static_cast<double>(b % fGrid);
        const double dx = wrap(ax, bx), dy = wrap(ay, by);
        return std::sqrt(dx * dx + dy * dy);
    }

    bool KolmogorovFlow::Localization(Matrix &rhoXY, Matrix &rhoYY) const
    {
        CheckShapes();
        std::lock_guard<std::mutex> lock(fCacheMutex);
        if (fRhoXY.Empty())
        {
            const std::size_t n = NState(), p = fProbes.size(), copies = fGradient ? 2 : 1;
            Matrix xy(n, p * copies), yy(p * copies, p * copies);
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t k = 0; k < p; ++k)
                {
                    const double rho = Engines::GaspariCohn(TorusDistance(c, fProbes[k]), fLocRadius);
                    for (std::size_t a = 0; a < copies; ++a) xy(c, a * p + k) = rho;
                }
            for (std::size_t k = 0; k < p; ++k)
                for (std::size_t l = 0; l < p; ++l)
                {
                    const double rho = Engines::GaspariCohn(TorusDistance(fProbes[k], fProbes[l]), fLocRadius);
                    for (std::size_t a = 0; a < copies; ++a)
                        for (std::size_t b = 0; b < copies; ++b) yy(a * p + k, b * p + l) = rho;
                }
            fRhoXY = xy;
            fRhoYY = yy;
        }
        rhoXY = fRhoXY;
        rhoYY = fRhoYY;
        return true;
    }

    bool KolmogorovFlow::ProcessNoiseVariances(double, Vector &variances) const
    {
        variances = fVariance;
        return true;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Weather::KolmogorovFlow, "kolmogorov-flow");
