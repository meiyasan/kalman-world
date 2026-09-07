/**
 * *********************************************
 *
 * \file FFT.cc
 * \brief Radix-2 and Bluestein transforms
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/FFT.h"

#include <cmath>

namespace Kalman::FFT
{
    namespace
    {
        using cd = std::complex<double>;
        constexpr double kPi = 3.14159265358979323846;

        bool IsPowerOfTwo(std::size_t n) { return n && !(n & (n - 1)); }

        // Iterative Cooley–Tukey, n a power of two.
        void Radix2(std::vector<cd> &a, bool inverse)
        {
            const std::size_t n = a.size();
            for (std::size_t i = 1, j = 0; i < n; ++i)
            {
                std::size_t bit = n >> 1;
                for (; j & bit; bit >>= 1) j ^= bit;
                j ^= bit;
                if (i < j) std::swap(a[i], a[j]);
            }
            for (std::size_t len = 2; len <= n; len <<= 1)
            {
                const double angle = 2. * kPi / static_cast<double>(len) * (inverse ? 1. : -1.);
                const cd wlen(std::cos(angle), std::sin(angle));
                for (std::size_t i = 0; i < n; i += len)
                {
                    cd w(1., 0.);
                    for (std::size_t j = 0; j < len / 2; ++j)
                    {
                        const cd u = a[i + j];
                        const cd v = a[i + j + len / 2] * w;
                        a[i + j] = u + v;
                        a[i + j + len / 2] = u - v;
                        w *= wlen;
                    }
                }
            }
        }

        // Bluestein: x_k e^{-i pi k^2 / n} convolved with the chirp, through a
        // power-of-two FFT of length >= 2n - 1.
        void Bluestein(std::vector<cd> &a, bool inverse)
        {
            const std::size_t n = a.size();
            std::size_t m = 1;
            while (m < 2 * n - 1) m <<= 1;

            const double sign = inverse ? 1. : -1.;
            std::vector<cd> chirp(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                // k^2 mod 2n keeps the angle small for long transforms.
                const double phase = sign * kPi * static_cast<double>((k * k) % (2 * n)) / static_cast<double>(n);
                chirp[k] = cd(std::cos(phase), std::sin(phase));
            }

            std::vector<cd> A(m, cd(0., 0.)), B(m, cd(0., 0.));
            for (std::size_t k = 0; k < n; ++k) A[k] = a[k] * chirp[k];
            B[0] = std::conj(chirp[0]);
            for (std::size_t k = 1; k < n; ++k) B[k] = B[m - k] = std::conj(chirp[k]);

            Radix2(A, false);
            Radix2(B, false);
            for (std::size_t k = 0; k < m; ++k) A[k] *= B[k];
            Radix2(A, true);
            const double norm = 1. / static_cast<double>(m);
            for (std::size_t k = 0; k < n; ++k) a[k] = A[k] * norm * chirp[k];
        }
    }

    void Transform(std::vector<cd> &x, bool inverse)
    {
        const std::size_t n = x.size();
        if (n <= 1) return;
        if (IsPowerOfTwo(n)) Radix2(x, inverse);
        else                 Bluestein(x, inverse);
        if (inverse)
        {
            const double norm = 1. / static_cast<double>(n);
            for (cd &v : x) v *= norm;
        }
    }

    namespace
    {
        SpectralField Transform2(const SpectralField &f, bool inverse)
        {
            const std::size_t ny = f.Rows(), nx = f.Cols();
            SpectralField out = f;

            std::vector<cd> line(nx);
            for (std::size_t i = 0; i < ny; ++i)
            {
                for (std::size_t j = 0; j < nx; ++j) line[j] = out(i, j);
                Transform(line, inverse);
                for (std::size_t j = 0; j < nx; ++j) out(i, j) = line[j];
            }
            line.assign(ny, cd());
            for (std::size_t j = 0; j < nx; ++j)
            {
                for (std::size_t i = 0; i < ny; ++i) line[i] = out(i, j);
                Transform(line, inverse);
                for (std::size_t i = 0; i < ny; ++i) out(i, j) = line[i];
            }
            return out;
        }
    }

    SpectralField Forward(const SpectralField &f) { return Transform2(f, false); }
    SpectralField Forward(const Field &f) { return Transform2(Complex(f), false); }
    SpectralField Inverse(const SpectralField &f) { return Transform2(f, true); }

    std::vector<double> Frequencies(std::size_t n)
    {
        std::vector<double> k(n);
        for (std::size_t i = 0; i < n; ++i)
            k[i] = (i < (n + 1) / 2) ? static_cast<double>(i) : static_cast<double>(i) - static_cast<double>(n);
        return k;
    }
}
