/**
 * *********************************************
 *
 * \file FiniteDifference.cc
 * \brief Implementation of the Kalman::Solvers::FiniteDifference class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Solvers/FiniteDifference.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace Kalman::Solvers
{
    FiniteDifference::FiniteDifference(std::size_t ny, std::size_t nx, double dt, double Ly, double Lx,
                                       Boundary boundary, double boundaryValue, double nu,
                                       std::shared_ptr<Integrator> integrator, std::uint64_t seed)
        : Solver(std::move(integrator)), fNy(ny), fNx(nx), fLy(Ly), fLx(Lx),
          fDy(Ly / static_cast<double>(ny)), fDx(Lx / static_cast<double>(nx)), fDt(dt),
          fBoundary(boundary), fBoundaryValue(boundaryValue), fNu(nu), fRandom(seed)
    {
        if (ny < 3 || nx < 3) throw std::runtime_error("Solvers::FiniteDifference: the grid needs at least 3 cells a side.");
        if (!(Ly > 0.) || !(Lx > 0.)) throw std::runtime_error("Solvers::FiniteDifference: domain lengths must be positive.");
    }

    void FiniteDifference::CheckShape(const Field &w) const
    {
        if (w.Rows() != fNy || w.Cols() != fNx)
        {
            std::ostringstream os;
            os << "Solvers::FiniteDifference: field grid (" << w.Rows() << ", " << w.Cols()
               << ") != solver grid (" << fNy << ", " << fNx << ").";
            throw std::runtime_error(os.str());
        }
    }

    Field FiniteDifference::Discretize(const Field &w) const
    {
        CheckShape(w);
        Field p(fNy + 2, fNx + 2, fBoundaryValue);
        for (std::size_t i = 0; i < fNy; ++i)
            for (std::size_t j = 0; j < fNx; ++j) p(i + 1, j + 1) = w(i, j);

        auto wrap = [](std::size_t i, std::size_t n) { return (i + n) % n; };
        for (std::size_t i = 0; i < fNy + 2; ++i)
            for (std::size_t j = 0; j < fNx + 2; ++j)
            {
                const bool edge = i == 0 || j == 0 || i == fNy + 1 || j == fNx + 1;
                if (!edge) continue;
                if (fBoundary == Boundary::Periodic)
                    p(i, j) = w(wrap(i + fNy - 1, fNy), wrap(j + fNx - 1, fNx));
                else if (fBoundary == Boundary::Neumann)
                    p(i, j) = w(std::min(std::max<std::size_t>(i, 1), fNy) - 1,
                                std::min(std::max<std::size_t>(j, 1), fNx) - 1);
                // Dirichlet: already fBoundaryValue
            }
        return p;
    }

    void FiniteDifference::Gradient(const Field &w, Field &wx, Field &wy) const
    {
        const Field p = Discretize(w);
        wx = Field(fNy, fNx);
        wy = Field(fNy, fNx);
        for (std::size_t i = 0; i < fNy; ++i)
            for (std::size_t j = 0; j < fNx; ++j)
            {
                wx(i, j) = (p(i + 1, j + 2) - p(i + 1, j)) / (2. * fDx);
                wy(i, j) = (p(i + 2, j + 1) - p(i, j + 1)) / (2. * fDy);
            }
    }

    Field FiniteDifference::Laplacian(const Field &w) const
    {
        const Field p = Discretize(w);
        Field out(fNy, fNx);
        for (std::size_t i = 0; i < fNy; ++i)
            for (std::size_t j = 0; j < fNx; ++j)
            {
                const double c = p(i + 1, j + 1);
                out(i, j) = (p(i + 1, j + 2) - 2. * c + p(i + 1, j)) / (fDx * fDx)
                          + (p(i + 2, j + 1) - 2. * c + p(i, j + 1)) / (fDy * fDy);
            }
        return out;
    }

    Field FiniteDifference::RHS(const Field &w, double) const
    {
        Field out = Laplacian(w);
        out *= fNu;
        return out;
    }

    Field FiniteDifference::Step(const Field &w, double t) const
    {
        return RungeKuttaStep(ExplicitTableau(),
                              [this](const Field &f, double tt) { return RHS(f, tt); },
                              w, t, fDt);
    }

    Field FiniteDifference::Advance(const Field &w, std::size_t nsteps, double t) const
    {
        CheckShape(w);
        Field f = w;
        for (std::size_t k = 0; k < nsteps; ++k)
        {
            f = Step(f, t);
            t += fDt;
        }
        return f;
    }

    Field FiniteDifference::Solve(const Field &field, std::size_t steps, double t) const
    {
        return Advance(field, steps, t);
    }

    Field FiniteDifference::RandomIC(double amp) const
    {
        std::normal_distribution<double> gauss(0., 1.);
        Field f(fNy, fNx);
        for (std::size_t i = 0; i < f.Size(); ++i) f.Data()[i] = gauss(fRandom);

        const double h2 = std::pow(std::min(fDx, fDy), 2);
        for (int sweep = 0; sweep < 8; ++sweep)              // smooth by diffusion sweeps
        {
            Field lap = Laplacian(f);
            lap *= 0.2 * h2;
            f += lap;
        }

        double mean = 0., var = 0.;
        for (std::size_t i = 0; i < f.Size(); ++i) mean += f.Data()[i];
        mean /= static_cast<double>(f.Size());
        for (std::size_t i = 0; i < f.Size(); ++i) var += std::pow(f.Data()[i] - mean, 2);
        f *= amp / (std::sqrt(var / static_cast<double>(f.Size())) + 1e-12);
        return f;
    }
}
