/**
 * *********************************************
 *
 * \file Spectral.h
 * \brief Header of the Kalman::Solvers::Spectral class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Solvers::Spectral
 * \brief Pseudo-spectral 2-D field solver: FFT discretization, composed time march.
 *
 * \details Owns the spatial part — the transform, the `ik` operators, the
 * 2/3-rule dealiasing mask — and delegates the time march to the composed
 * Kalman::Integrator applied to the **spectral** field. A subclass supplies the
 * physics through RHSHat(): the spectral `d(omega_hat)/dt`. This class is
 * domain-agnostic; a Kolmogorov flow or a barotropic model is an application.
 *
 * Spectral discretization requires a **regular periodic** mesh; bounded or
 * irregular grids are the job of the finite-difference kind. Wavenumbers scale
 * as `2 pi / L`, so `L = 2 pi` reproduces integer wavenumbers exactly.
 *
 * Mirrors `modules/framework/solvers/spectral.py` (the numpy path; the torch
 * backend is a Python concern).
 *
 * *********************************************
 */

#ifndef KALMAN_SOLVERS_SPECTRAL_H
#define KALMAN_SOLVERS_SPECTRAL_H

#include "Kalman/Solver.h"

#include <cstdint>
#include <random>

namespace Kalman::Solvers
{
    class Spectral : public Solver
    {
        protected:
            std::size_t fNy = 0, fNx = 0;
            double fLy = 0., fLx = 0.;
            double fDt = 0.01;
            Field fIx, fIy;            // integer mode indices
            Field fKx, fKy, fK2;       // physical wavenumbers and |k|^2
            Field fK2Inv;              // 1 / |k|^2, zero at k = 0
            Field fDealias;            // 2/3-rule mask (1 keep, 0 drop)
            mutable std::mt19937_64 fRandom;

        public:
            /**
             * \param ny, nx    grid cells along y and x.
             * \param dt        the solver's own time step.
             * \param Ly, Lx    physical domain lengths (default 2 pi each).
             * \param integrator the time march, or nullptr for RK4.
             * \param seed      seeds RandomIC().
             */
            Spectral(std::size_t ny, std::size_t nx, double dt = 0.01,
                     double Ly = 6.283185307179586, double Lx = 6.283185307179586,
                     std::shared_ptr<Integrator> integrator = nullptr, std::uint64_t seed = 0);

            SolverKind Kind() const override { return SolverKind::Spectral; }
            const char *Title() const override { return "Spectral Solver"; }
            const char *Description() const override
            { return "Pseudo-spectral (FFT) 2D field solver; explicit RK time march in spectral space."; }

            std::size_t Ny() const { return fNy; }
            std::size_t Nx() const { return fNx; }
            double Ly() const { return fLy; }
            double Lx() const { return fLx; }
            double Dt() const { return fDt; }
            void SetDt(double dt) { fDt = dt; }

            const Field &Kx() const { return fKx; }
            const Field &Ky() const { return fKy; }
            const Field &K2() const { return fK2; }
            const Field &K2Inv() const { return fK2Inv; }
            const Field &Dealias() const { return fDealias; }

            /// \throws std::runtime_error when \p w is not (ny, nx).
            void CheckShape(const Field &w) const;

            // --- the physics hook ----------------------------------------------------
            /// `d(w_hat)/dt` in spectral space. \throws std::runtime_error unless overridden.
            virtual SpectralField RHSHat(const SpectralField &wh, double t) const;

            // --- spectral operators ---------------------------------------------------
            /// `(dw/dx, dw/dy)` of a real field.
            void Gradient(const Field &w, Field &wx, Field &wy) const;
            /// Incompressible velocity `(u, v)` from the vorticity `w`.
            void Velocity(const Field &w, Field &u, Field &v) const;
            /// Convenience: elementwise `i kx wh` and `i ky wh`.
            SpectralField DerivativeX(const SpectralField &wh) const;
            SpectralField DerivativeY(const SpectralField &wh) const;

            // --- the time march -------------------------------------------------------
            /// One step of the spectral field with the composed integrator.
            SpectralField Step(const SpectralField &wh, double t = 0.) const;
            /// Advance a real field by \p nsteps.
            Field Advance(const Field &w, std::size_t nsteps, double t = 0.) const;
            Field Solve(const Field &field, std::size_t steps = 1, double t = 0.) const override;

            /// A random large-scale field of unit standard deviation, scaled by \p amp.
            Field RandomIC(double amp = 1.) const;
    };
}

#endif // KALMAN_SOLVERS_SPECTRAL_H
