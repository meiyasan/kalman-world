/**
 * *********************************************
 *
 * \file Vorticity.h
 * \brief Header of the Kalman::Solvers::Vorticity class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Solvers::Vorticity
 * \brief The 2-D vorticity equation on the spectral solver.
 *
 * \details Kalman::Solvers::Spectral ships no physics. This subclass is the
 * equation the framework's flow models share,
 *
 * \code{.unparsed}
 *   dw/dt = -u.grad(w) - beta v + nu lap(w) + F
 * \endcode
 *
 * on a periodic grid: derivatives and the velocity in Fourier space, the
 * products on the grid, 2/3-rule dealiasing. With `beta = 0` and a forcing it
 * is the Kolmogorov flow; with `beta` and no forcing, the barotropic
 * beta-plane.
 *
 * It steps one field at a time with the built-in FFT and needs nothing
 * outside libKalman. Kalman::Learn::Solvers::Vorticity is the same equation
 * on libTorch, batched, for CPU and GPU.
 *
 * *********************************************
 */

#ifndef KALMAN_SOLVERS_VORTICITY_H
#define KALMAN_SOLVERS_VORTICITY_H

#include "Kalman/Solvers/Spectral.h"

namespace Kalman::Solvers
{
    class Vorticity : public Spectral
    {
        protected:
            double fNu = 0.;
            double fBeta = 0.;
            SpectralField fForcing;      // transform of F; empty for none

        public:
            /**
             * \param nu       viscosity.
             * \param beta     planetary-vorticity gradient (0 for none).
             * \param forcing  the transform of the forcing F (ny x nx), or an empty field.
             */
            Vorticity(std::size_t ny, std::size_t nx, double dt, double Ly, double Lx,
                      double nu, double beta = 0., const SpectralField &forcing = SpectralField(),
                      std::shared_ptr<Integrator> integrator = nullptr, std::uint64_t seed = 0);

            const char *Title() const override { return "Vorticity Solver"; }
            const char *Description() const override
            { return "Pseudo-spectral 2D vorticity: advection, beta effect, viscosity and forcing."; }

            double Nu() const { return fNu; }
            double Beta() const { return fBeta; }

            SpectralField RHSHat(const SpectralField &wh, double t) const override;
    };
}

#endif // KALMAN_SOLVERS_VORTICITY_H
