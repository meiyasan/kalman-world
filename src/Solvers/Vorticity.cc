/**
 * *********************************************
 *
 * \file Vorticity.cc
 * \brief Implementation of the Kalman::Solvers::Vorticity class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Solvers/Vorticity.h"
#include "Kalman/FFT.h"

#include <complex>
#include <stdexcept>

namespace Kalman::Solvers
{
    namespace
    {
        using cd = std::complex<double>;
    }

    Vorticity::Vorticity(std::size_t ny, std::size_t nx, double dt, double Ly, double Lx,
                         double nu, double beta, const SpectralField &forcing,
                         std::shared_ptr<Integrator> integrator, std::uint64_t seed)
        : Spectral(ny, nx, dt, Ly, Lx, std::move(integrator), seed), fNu(nu), fBeta(beta), fForcing(forcing)
    {
        if (fForcing.Size() != 0 && (fForcing.Rows() != ny || fForcing.Cols() != nx))
            throw std::runtime_error("Solvers::Vorticity: the forcing must have the grid's shape.");
    }

    SpectralField Vorticity::RHSHat(const SpectralField &wh, double) const
    {
        const std::size_t size = wh.Size();

        // Stream function, velocity and gradient, all from wh by multiplications.
        SpectralField psih = wh;
        for (std::size_t i = 0; i < size; ++i) psih.Data()[i] *= fK2Inv.Data()[i];
        SpectralField vh = DerivativeX(psih);
        vh *= -1.;
        const Field u = Real(FFT::Inverse(DerivativeY(psih)));
        const Field v = Real(FFT::Inverse(vh));
        const Field wx = Real(FFT::Inverse(DerivativeX(wh)));
        const Field wy = Real(FFT::Inverse(DerivativeY(wh)));

        // The products are taken on the grid, then dealiased in Fourier space.
        Field product(fNy, fNx);
        for (std::size_t i = 0; i < size; ++i)
            product.Data()[i] = u.Data()[i] * wx.Data()[i] + v.Data()[i] * wy.Data()[i] + fBeta * v.Data()[i];
        SpectralField out = FFT::Forward(product);

        const bool forced = fForcing.Size() != 0;
        for (std::size_t i = 0; i < size; ++i)
        {
            cd value = -out.Data()[i] * fDealias.Data()[i] - fNu * fK2.Data()[i] * wh.Data()[i];
            if (forced) value += fForcing.Data()[i];
            out.Data()[i] = value;
        }
        return out;
    }
}
