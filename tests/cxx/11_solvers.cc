/**
 * \file 11_solvers.cc
 * \brief The FFT, the spectral and finite-difference solvers, and the reserved kinds.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include <algorithm>
#include <vector>
#include <complex>

#include <cmath>
#include <stdexcept>

using namespace Kalman;

namespace
{
    constexpr double kPi = 3.14159265358979323846;
    using cd = std::complex<double>;

    double MaxAbsDiff(const Field &a, const Field &b)
    {
        double d = 0.;
        for (std::size_t i = 0; i < a.Size(); ++i) d = std::max(d, std::abs(a.Data()[i] - b.Data()[i]));
        return d;
    }

    // Direct O(n^2) DFT, the reference the fast transforms are checked against.
    std::vector<cd> DirectDFT(const std::vector<cd> &x)
    {
        const std::size_t n = x.size();
        std::vector<cd> out(n);
        for (std::size_t k = 0; k < n; ++k)
            for (std::size_t j = 0; j < n; ++j)
                out[k] += x[j] * std::exp(cd(0., -2. * kPi * static_cast<double>(k * j) / static_cast<double>(n)));
        return out;
    }

    Field SinCos(std::size_t n, double L)
    {
        Field f(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                f(i, j) = std::sin(L * j / n) * std::cos(L * i / n);   // sin(x) cos(y), x along columns
        return f;
    }

    /// Spectral diffusion: d(w_hat)/dt = -nu k^2 w_hat — the smallest physics a spectral solver can carry.
    class Diffusion : public Solvers::Spectral
    {
        private:
            double fNu;
        public:
            Diffusion(std::size_t n, double dt, double nu) : Solvers::Spectral(n, n, dt), fNu(nu) {}
            SpectralField RHSHat(const SpectralField &wh, double) const override
            {
                SpectralField out = wh;
                for (std::size_t i = 0; i < out.Size(); ++i) out.Data()[i] *= -fNu * K2().Data()[i];
                return out;
            }
    };
}

int main()
{
    // --- the transform itself, on every kind of length -------------------------
    {
        const double pi = 3.14159265358979323846;
        for (std::size_t n : {8, 12, 15, 48, 100, 7, 22})     // 2^k, {2,3,5}-smooth, and neither
        {
            std::vector<std::complex<double>> x(n), direct(n);
            for (std::size_t k = 0; k < n; ++k) x[k] = std::complex<double>(std::sin(0.7 * k) + 0.1 * k, std::cos(1.3 * k));
            for (std::size_t k = 0; k < n; ++k)
            {
                std::complex<double> sum(0., 0.);
                for (std::size_t j = 0; j < n; ++j)
                    sum += x[j] * std::polar(1., -2. * pi * double(j * k % n) / double(n));
                direct[k] = sum;
            }
            std::vector<std::complex<double>> y = x;
            FFT::Transform(y, false);
            double worst = 0.;
            for (std::size_t k = 0; k < n; ++k) worst = std::max(worst, std::abs(y[k] - direct[k]));
            CHECK(worst < 1e-10 * double(n), "FFT: matches the definition at this length");
            FFT::Transform(y, true);
            worst = 0.;
            for (std::size_t k = 0; k < n; ++k) worst = std::max(worst, std::abs(y[k] - x[k]));
            CHECK(worst < 1e-12 * double(n), "FFT: inverse of forward is the identity");
        }
    }

    // --- the transform ------------------------------------------------------------
    {
        for (std::size_t n : {8u, 16u, 12u, 15u, 30u, 33u})
        {
            std::vector<cd> x(n);
            for (std::size_t i = 0; i < n; ++i) x[i] = cd(std::sin(0.3 * i) + 0.1 * i, std::cos(1.7 * i));
            const std::vector<cd> ref = DirectDFT(x);
            std::vector<cd> fast = x;
            FFT::Transform(fast, false);
            double err = 0.;
            for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(fast[i] - ref[i]));
            FFT::Transform(fast, true);
            double back = 0.;
            for (std::size_t i = 0; i < n; ++i) back = std::max(back, std::abs(fast[i] - x[i]));
            const std::string tag = "FFT n = " + std::to_string(n);
            CHECK(err < 1e-9, (tag + ": matches the direct DFT").c_str());
            CHECK(back < 1e-12, (tag + ": inverse round trip").c_str());
        }
        const std::vector<double> k = FFT::Frequencies(8);
        CHECK(k[0] == 0. && k[3] == 3. && k[4] == -4. && k[7] == -1., "frequencies: numpy fftfreq ordering (even n)");
        const std::vector<double> k5 = FFT::Frequencies(5);
        CHECK(k5[2] == 2. && k5[3] == -2., "frequencies: odd n");

        Field f(6, 10);
        for (std::size_t i = 0; i < f.Size(); ++i) f.Data()[i] = std::cos(0.4 * i);
        CHECK(MaxAbsDiff(Real(FFT::Inverse(FFT::Forward(f))), f) < 1e-12, "FFT 2-D: rectangular round trip");
    }

    // --- spectral operators ----------------------------------------------------------
    {
        const std::size_t n = 32;
        Solvers::Spectral s(n, n);
        CHECK(s.Kind() == SolverKind::Spectral, "spectral: kind");
        CHECK(s.GetIntegrator().Kind() == IntegratorKind::RK4, "spectral: composes RK4 by default");

        const Field w = SinCos(n, 2. * kPi);
        Field wx, wy;
        s.Gradient(w, wx, wy);
        Field ex(n, n), ey(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
                const double x = 2. * kPi * j / n, y = 2. * kPi * i / n;
                ex(i, j) = std::cos(x) * std::cos(y);
                ey(i, j) = -std::sin(x) * std::sin(y);
            }
        CHECK(MaxAbsDiff(wx, ex) < 1e-10, "spectral: d/dx of sin(x)cos(y) is exact");
        CHECK(MaxAbsDiff(wy, ey) < 1e-10, "spectral: d/dy of sin(x)cos(y) is exact");

        // Wavenumbers scale with the physical length.
        const double Lx = 4.;
        Solvers::Spectral r(n, n, 0.01, 2. * kPi, Lx);
        Field g(n, n), gx(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
                const double x = Lx * j / n;
                g(i, j) = std::sin(2. * kPi * x / Lx);
                gx(i, j) = (2. * kPi / Lx) * std::cos(2. * kPi * x / Lx);
            }
        Field rx, ry;
        r.Gradient(g, rx, ry);
        CHECK(MaxAbsDiff(rx, gx) < 1e-10, "spectral: wavenumbers scale as 2 pi / L");

        // Velocity from a vorticity: w = sin(x) => psi = -sin(x), u = 0, v = -cos(x)... check divergence-free instead.
        Field u, v;
        s.Velocity(w, u, v);
        Field ux, uy, vx, vy;
        s.Gradient(u, ux, uy);
        s.Gradient(v, vx, vy);
        Field div = ux + vy;
        double maxdiv = 0.;
        for (std::size_t i = 0; i < div.Size(); ++i) maxdiv = std::max(maxdiv, std::abs(div.Data()[i]));
        CHECK(maxdiv < 1e-10, "spectral: the velocity is divergence-free");

        bool threw = false;
        try { s.Advance(Field(8, 8), 1); } catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "spectral: a mis-gridded field is refused");
        threw = false;
        try { s.Advance(w, 1); } catch (const std::runtime_error &e)
        { threw = std::string(e.what()).find("RHSHat") != std::string::npos; }
        CHECK(threw, "spectral: the base class ships no physics and says so");

        const Field ic = s.RandomIC(2.);
        double mean = 0., var = 0.;
        for (std::size_t i = 0; i < ic.Size(); ++i) mean += ic.Data()[i];
        mean /= ic.Size();
        for (std::size_t i = 0; i < ic.Size(); ++i) var += std::pow(ic.Data()[i] - mean, 2);
        CHECK_CLOSE(std::sqrt(var / ic.Size()), 2., 1e-9, "spectral: RandomIC has the requested amplitude");
    }

    // --- spectral diffusion decays analytically -------------------------------------
    {
        const std::size_t n = 32;
        const double nu = 0.1, dt = 1e-3, tEnd = 0.5;
        Diffusion d(n, dt, nu);
        const Field w = SinCos(n, 2. * kPi);                 // |k|^2 = 2: decays as e^{-2 nu t}
        const Field out = d.Advance(w, static_cast<std::size_t>(tEnd / dt));
        Field expect = w; expect *= std::exp(-2. * nu * tEnd);
        CHECK(MaxAbsDiff(out, expect) < 1e-6, "spectral diffusion: matches e^{-2 nu t} decay");

        Diffusion e(n, dt, nu);
        e.SetIntegrator(IntegratorKind::Euler);
        const Field outE = e.Advance(w, static_cast<std::size_t>(tEnd / dt));
        CHECK(MaxAbsDiff(outE, expect) < 1e-3 && MaxAbsDiff(outE, expect) > 1e-7,
              "spectral diffusion: Euler is close but measurably worse");
    }

    // --- finite differences ------------------------------------------------------------
    {
        const std::size_t n = 64;
        const double L = 2. * kPi;
        Solvers::FiniteDifference s(n, n, 1e-4, L, L, Solvers::FiniteDifference::Boundary::Periodic, 0., 0.1);
        CHECK(s.Kind() == SolverKind::FiniteDifference, "fd: kind");

        const Field f = SinCos(n, L);
        Field fx, fy;
        s.Gradient(f, fx, fy);
        Field ex(n, n), ey(n, n), lap(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
            {
                const double x = L * j / n, y = L * i / n;
                ex(i, j) = std::cos(x) * std::cos(y);
                ey(i, j) = -std::sin(x) * std::sin(y);
                lap(i, j) = -2. * f(i, j);
            }
        CHECK(MaxAbsDiff(fx, ex) < 5e-3, "fd: second-order gradient in x");
        CHECK(MaxAbsDiff(fy, ey) < 5e-3, "fd: second-order gradient in y");
        CHECK(MaxAbsDiff(s.Laplacian(f), lap) < 5e-3, "fd: 5-point Laplacian");

        const double tEnd = 0.5;
        const Field out = s.Advance(f, static_cast<std::size_t>(tEnd / s.Dt()));
        Field expect = f; expect *= std::exp(-2. * 0.1 * tEnd);
        CHECK(MaxAbsDiff(out, expect) < 1e-3, "fd diffusion: matches the analytic decay");

        using B = Solvers::FiniteDifference::Boundary;
        const Field ones(16, 16, 1.);
        const Field per = Solvers::FiniteDifference(16, 16, 1e-3, L, L, B::Periodic, 0., 0.5).Advance(ones, 20);
        const Field neu = Solvers::FiniteDifference(16, 16, 1e-3, L, L, B::Neumann, 0., 0.5).Advance(ones, 20);
        const Field dir = Solvers::FiniteDifference(16, 16, 1e-3, L, L, B::Dirichlet, 0., 0.5).Advance(ones, 20);
        CHECK(MaxAbsDiff(per, ones) < 1e-9, "fd: a uniform field is stationary (periodic)");
        CHECK(MaxAbsDiff(neu, ones) < 1e-9, "fd: a uniform field is stationary (neumann)");
        CHECK(dir(0, 0) < 0.99, "fd: a Dirichlet wall pulls the edge toward the boundary value");
        CHECK_CLOSE(dir(8, 8), 1., 1e-6, "fd: the interior is untouched after 20 short steps");

        Solvers::FiniteDifference rect(8, 16, 1e-3);
        CHECK(rect.Ny() == 8 && rect.Nx() == 16, "fd: rectangular grid");
        const Field ic = rect.RandomIC();
        CHECK(rect.Advance(ic, 3).Rows() == 8 && rect.Advance(ic, 3).Cols() == 16, "fd: advances a rectangular field");

        Solvers::FiniteDifference eul(16, 16, 1e-3);
        eul.SetIntegrator(IntegratorKind::Euler);
        CHECK(eul.GetIntegrator().Kind() == IntegratorKind::Euler, "fd: the integrator is swappable");
        CHECK(eul.Solve(Field(16, 16), 2).Size() == 256, "fd: Solve() is Advance()");
    }

    // --- resolution by kind and the reserved solvers ---------------------------------
    {
        CHECK(Solvers::Make(SolverKind::Spectral, 16)->Kind() == SolverKind::Spectral, "make: spectral");
        CHECK(Solvers::Make(SolverKind::FiniteDifference, 16)->Kind() == SolverKind::FiniteDifference, "make: finite difference");
        for (SolverKind k : {SolverKind::FiniteVolume, SolverKind::FiniteElement})
        {
            bool threw = false;
            try { Solvers::Make(k); } catch (const std::runtime_error &e)
            { threw = std::string(e.what()).find("reserved") != std::string::npos; }
            CHECK(threw, "make: the reserved kinds throw and say so");
        }
        CHECK(std::string(ToString(SolverKind::FiniteElement)) == "FiniteElement", "kinds: nameable");
    }

    return Check::Summary("11_solvers");
}
