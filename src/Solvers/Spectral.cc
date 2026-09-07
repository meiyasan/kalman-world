/**
 * *********************************************
 *
 * \file Spectral.cc
 * \brief Implementation of the Kalman::Solvers::Spectral class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Solvers/Spectral.h"
#include "Kalman/FFT.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace Kalman::Solvers
{
    namespace
    {
        constexpr double kPi = 3.14159265358979323846;
        using cd = std::complex<double>;
    }

    Spectral::Spectral(std::size_t ny, std::size_t nx, double dt, double Ly, double Lx,
                       std::shared_ptr<Integrator> integrator, std::uint64_t seed)
        : Solver(std::move(integrator)), fNy(ny), fNx(nx), fLy(Ly), fLx(Lx), fDt(dt), fRandom(seed)
    {
        if (ny < 2 || nx < 2) throw std::runtime_error("Solvers::Spectral: the grid needs at least 2 cells a side.");
        if (!(Ly > 0.) || !(Lx > 0.)) throw std::runtime_error("Solvers::Spectral: domain lengths must be positive.");

        const std::vector<double> ix = FFT::Frequencies(nx);
        const std::vector<double> iy = FFT::Frequencies(ny);

        fIx = Field(ny, nx); fIy = Field(ny, nx);
        fKx = Field(ny, nx); fKy = Field(ny, nx);
        fK2 = Field(ny, nx); fK2Inv = Field(ny, nx);
        fDealias = Field(ny, nx);

        const double sx = 2. * kPi / Lx, sy = 2. * kPi / Ly;
        for (std::size_t i = 0; i < ny; ++i)
            for (std::size_t j = 0; j < nx; ++j)
            {
                fIx(i, j) = ix[j];
                fIy(i, j) = iy[i];
                fKx(i, j) = sx * ix[j];
                fKy(i, j) = sy * iy[i];
                fK2(i, j) = fKx(i, j) * fKx(i, j) + fKy(i, j) * fKy(i, j);
                fK2Inv(i, j) = fK2(i, j) > 0. ? 1. / std::max(fK2(i, j), 1e-12) : 0.;
                fDealias(i, j) = (std::abs(ix[j]) <= static_cast<double>(nx / 3)
                                  && std::abs(iy[i]) <= static_cast<double>(ny / 3)) ? 1. : 0.;
            }
    }

    void Spectral::CheckShape(const Field &w) const
    {
        if (w.Rows() != fNy || w.Cols() != fNx)
        {
            std::ostringstream os;
            os << "Solvers::Spectral: field grid (" << w.Rows() << ", " << w.Cols()
               << ") != solver grid (" << fNy << ", " << fNx << ").";
            throw std::runtime_error(os.str());
        }
    }

    SpectralField Spectral::RHSHat(const SpectralField &, double) const
    {
        throw std::runtime_error("Solvers::Spectral: a subclass must define RHSHat(wh, t) — "
                                 "the solver is domain-agnostic and ships no physics.");
    }

    SpectralField Spectral::DerivativeX(const SpectralField &wh) const
    {
        SpectralField out = wh;
        for (std::size_t i = 0; i < out.Size(); ++i) out.Data()[i] *= cd(0., fKx.Data()[i]);
        return out;
    }

    SpectralField Spectral::DerivativeY(const SpectralField &wh) const
    {
        SpectralField out = wh;
        for (std::size_t i = 0; i < out.Size(); ++i) out.Data()[i] *= cd(0., fKy.Data()[i]);
        return out;
    }

    void Spectral::Gradient(const Field &w, Field &wx, Field &wy) const
    {
        CheckShape(w);
        const SpectralField wh = FFT::Forward(w);
        wx = Real(FFT::Inverse(DerivativeX(wh)));
        wy = Real(FFT::Inverse(DerivativeY(wh)));
    }

    void Spectral::Velocity(const Field &w, Field &u, Field &v) const
    {
        CheckShape(w);
        SpectralField psih = FFT::Forward(w);
        for (std::size_t i = 0; i < psih.Size(); ++i) psih.Data()[i] *= fK2Inv.Data()[i];
        u = Real(FFT::Inverse(DerivativeY(psih)));
        SpectralField vh = DerivativeX(psih);
        vh *= -1.;
        v = Real(FFT::Inverse(vh));
    }

    SpectralField Spectral::Step(const SpectralField &wh, double t) const
    {
        return RungeKuttaStep(ExplicitTableau(),
                              [this](const SpectralField &f, double tt) { return RHSHat(f, tt); },
                              wh, t, fDt);
    }

    Field Spectral::Advance(const Field &w, std::size_t nsteps, double t) const
    {
        CheckShape(w);
        SpectralField wh = FFT::Forward(w);
        for (std::size_t k = 0; k < nsteps; ++k)
        {
            wh = Step(wh, t);
            t += fDt;
        }
        return Real(FFT::Inverse(wh));
    }

    Field Spectral::Solve(const Field &field, std::size_t steps, double t) const
    {
        return Advance(field, steps, t);
    }

    Field Spectral::RandomIC(double amp) const
    {
        std::normal_distribution<double> gauss(0., 1.);
        SpectralField wh(fNy, fNx);
        for (std::size_t i = 0; i < fNy; ++i)
            for (std::size_t j = 0; j < fNx; ++j)
            {
                const double kmag = std::sqrt(fIx(i, j) * fIx(i, j) + fIy(i, j) * fIy(i, j));
                const double taper = std::exp(-0.5 * std::pow(kmag / 6., 2)) * fDealias(i, j);
                wh(i, j) = cd(gauss(fRandom), gauss(fRandom)) * taper;
            }
        Field w = Real(FFT::Inverse(wh));

        double mean = 0., var = 0.;
        for (std::size_t i = 0; i < w.Size(); ++i) mean += w.Data()[i];
        mean /= static_cast<double>(w.Size());
        for (std::size_t i = 0; i < w.Size(); ++i) var += std::pow(w.Data()[i] - mean, 2);
        const double sd = std::sqrt(var / static_cast<double>(w.Size()));
        w *= amp / (sd + 1e-12);
        return w;
    }
}
