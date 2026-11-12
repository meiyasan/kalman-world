/**
 * *********************************************
 *
 * \file Vorticity.h
 * \brief Header of the Kalman::Learn::Solvers::Vorticity class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Learn::Solvers::Vorticity
 * \brief Batched pseudo-spectral 2-D vorticity solver on libTorch (CPU, CUDA, MPS).
 *
 * \details Advances many vorticity fields at once, which is what an ensemble
 * filter on a flow needs: one call steps every member. The equation is
 *
 * \code{.unparsed}
 *   dw/dt = -u.grad(w) - beta v + nu lap(w) + F
 * \endcode
 *
 * on a periodic grid: velocities and gradients in Fourier space, the products
 * on the grid, 2/3-rule dealiasing, and any explicit Runge-Kutta scheme given
 * by its Butcher tableau. With `beta = 0` and a forcing this is the Kolmogorov
 * flow; with `beta` and no forcing, the barotropic beta-plane.
 *
 * It is the libTorch counterpart of Kalman::Solvers::Spectral, which steps one
 * field at a time with the built-in FFT, and it replaces the Python Torch
 * backend of `kalman.framework.solvers.SpectralSolver`.
 *
 * On MPS (Apple GPUs) there is no double precision: fields are stepped in
 * single precision there, in double everywhere else.
 *
 * libTorch stays behind a pImpl. HasRuntime() says whether this build has it;
 * without it the constructor throws.
 *
 * *********************************************
 */

#ifndef KALMAN_LEARN_SOLVERS_VORTICITY_H
#define KALMAN_LEARN_SOLVERS_VORTICITY_H

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace Kalman::Learn::Solvers
{
    class Vorticity
    {
        public:
            /// The grid's spectral operators, each `ny * nx` values, row-major.
            struct Grid
            {
                std::size_t ny = 0, nx = 0;
                std::vector<double> kx, ky;      ///< wavenumbers
                std::vector<double> k2, k2inv;   ///< |k|^2 and its inverse (0 at k = 0)
                std::vector<double> dealias;     ///< the 2/3-rule mask
            };

            /// An explicit Runge-Kutta scheme: `c` (s), `A` (s x s, row-major), `b` (s).
            struct Tableau
            {
                std::vector<double> c, A, b;
                std::size_t Stages() const { return b.size(); }
            };

        private:
            struct Impl;
            std::shared_ptr<Impl> fImpl;

        public:
            /**
             * \param grid      the spectral operators.
             * \param nu        viscosity.
             * \param beta      planetary-vorticity gradient (0 for none).
             * \param forcingRe real part of the forcing's transform (`ny * nx`), or empty.
             * \param forcingIm its imaginary part, or empty.
             * \param device    "cpu", "cuda" or "mps"; one that is not available falls back to "cpu".
             * \throws std::runtime_error when this build has no libTorch.
             */
            Vorticity(const Grid &grid, double nu, double beta,
                      const std::vector<double> &forcingRe, const std::vector<double> &forcingIm,
                      const std::string &device = "cpu");

            /// True when this build has libTorch at all.
            static bool HasRuntime();

            /// The device actually in use.
            std::string Device() const;

            /**
             * \brief Advance `batch` fields by `nsteps` steps of `dt`.
             * \param w    `batch * ny * nx` values, field after field.
             * \param out  the same layout, written on return (may alias \p w).
             */
            void Advance(const double *w, std::size_t batch, std::size_t nsteps, double dt,
                         const Tableau &tableau, double *out) const;
    };
}

#endif // KALMAN_LEARN_SOLVERS_VORTICITY_H
