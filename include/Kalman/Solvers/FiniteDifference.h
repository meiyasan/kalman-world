/**
 * *********************************************
 *
 * \file FiniteDifference.h
 * \brief Header of the Kalman::Solvers::FiniteDifference class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Solvers::FiniteDifference
 * \brief Second-order finite-difference 2-D field solver — the bounded-domain kind.
 *
 * \details Where the spectral solver needs a regular *periodic* mesh, this one
 * works on any regular rectangular grid with a choice of boundary condition —
 * periodic, Dirichlet (pinned to a value outside) or Neumann (zero gradient) —
 * so walls and fixed values are representable. Central stencils for the
 * gradient and the 5-point Laplacian; the time march is the composed
 * Kalman::Integrator, through the same tableau path as the spectral solver.
 *
 * A subclass supplies the physics through RHS() in *real* space. The built-in
 * default is diffusion, `df/dt = nu Laplacian(f)`, which makes the class usable
 * on its own. Mirrors `modules/framework/solvers/finite_difference.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_SOLVERS_FINITEDIFFERENCE_H
#define KALMAN_SOLVERS_FINITEDIFFERENCE_H

#include "Kalman/Solver.h"

#include <cstdint>
#include <random>

namespace Kalman::Solvers
{
    class FiniteDifference : public Solver
    {
        public:
            enum class Boundary { Periodic, Dirichlet, Neumann };

        protected:
            std::size_t fNy = 0, fNx = 0;
            double fLy = 0., fLx = 0.;
            double fDy = 0., fDx = 0.;
            double fDt = 0.01;
            Boundary fBoundary = Boundary::Periodic;
            double fBoundaryValue = 0.;
            double fNu = 0.01;
            mutable std::mt19937_64 fRandom;

        public:
            FiniteDifference(std::size_t ny, std::size_t nx, double dt = 0.01,
                             double Ly = 6.283185307179586, double Lx = 6.283185307179586,
                             Boundary boundary = Boundary::Periodic, double boundaryValue = 0.,
                             double nu = 0.01, std::shared_ptr<Integrator> integrator = nullptr,
                             std::uint64_t seed = 0);

            SolverKind Kind() const override { return SolverKind::FiniteDifference; }
            const char *Title() const override { return "Finite-Difference Solver"; }
            const char *Description() const override
            { return "Second-order finite-difference 2D field solver (periodic / dirichlet / neumann); explicit RK time march."; }

            std::size_t Ny() const { return fNy; }
            std::size_t Nx() const { return fNx; }
            double Dx() const { return fDx; }
            double Dy() const { return fDy; }
            double Dt() const { return fDt; }
            void SetDt(double dt) { fDt = dt; }
            Boundary GetBoundary() const { return fBoundary; }
            double Nu() const { return fNu; }
            void SetNu(double nu) { fNu = nu; }

            /// \throws std::runtime_error when \p w is not (ny, nx).
            void CheckShape(const Field &w) const;

            /// One ghost cell on every side, filled per the boundary condition — the discretization.
            Field Discretize(const Field &w) const;

            // --- operators ------------------------------------------------------------
            void Gradient(const Field &w, Field &wx, Field &wy) const;
            Field Laplacian(const Field &w) const;

            // --- the physics hook: default diffusion ---------------------------------
            virtual Field RHS(const Field &w, double t) const;

            // --- the time march -------------------------------------------------------
            Field Step(const Field &w, double t = 0.) const;
            Field Advance(const Field &w, std::size_t nsteps, double t = 0.) const;
            Field Solve(const Field &field, std::size_t steps = 1, double t = 0.) const override;

            /// A smooth random field of unit standard deviation, scaled by \p amp.
            Field RandomIC(double amp = 1.) const;
    };
}

#endif // KALMAN_SOLVERS_FINITEDIFFERENCE_H
