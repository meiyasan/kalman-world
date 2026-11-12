/**
 * *********************************************
 *
 * \file FFT.h
 * \brief A dependency-free discrete Fourier transform for the spectral solver.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details libKalman has no required dependency, and the spectral
 * solver needs a 2-D FFT, so here is one: iterative radix-2 for power-of-two
 * lengths, Bluestein's chirp-z algorithm (which reduces any length to a
 * power-of-two convolution) otherwise. Correct for every grid size, fast
 * enough for the grids a Kalman ensemble marches (tens to hundreds of cells a
 * side). Conventions follow numpy: the forward transform is unnormalised, the
 * inverse divides by the length, and `Frequencies(n)` is `fftfreq(n) * n`.
 *
 * *********************************************
 */

#ifndef KALMAN_FFT_H
#define KALMAN_FFT_H

#include "Kalman/Grid.h"

#include <complex>
#include <vector>

namespace Kalman::FFT
{
    /// In-place 1-D transform of any length; \p inverse divides by the length.
    void Transform(std::vector<std::complex<double>> &x, bool inverse = false);

    /// Forward 2-D transform (rows then columns), unnormalised.
    SpectralField Forward(const SpectralField &f);
    /// Forward transform of a real field.
    SpectralField Forward(const Field &f);
    /// Inverse 2-D transform, normalised by `rows * cols`.
    SpectralField Inverse(const SpectralField &f);

    /// Integer mode indices `0, 1, ..., n/2-1, -n/2, ..., -1` (numpy's `fftfreq(n) * n`).
    std::vector<double> Frequencies(std::size_t n);
}

#endif // KALMAN_FFT_H
