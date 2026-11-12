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

#include <algorithm>
#include <cmath>

#include <unordered_map>

namespace Kalman::FFT
{
    namespace
    {
        using cd = std::complex<double>;
        constexpr double kPi = 3.14159265358979323846;

        bool IsPowerOfTwo(std::size_t n) { return n && !(n & (n - 1)); }

        // What a power-of-two transform of length n needs and that does not depend
        // on the data: the bit-reversal permutation and the twiddle factors
        // e^{-2 pi i k / n}, k < n/2. Built once per length and per thread.
        struct Plan
        {
            std::vector<std::size_t> reversed;
            std::vector<double> cosine, sine;
        };

        const Plan &PlanFor(std::size_t n)
        {
            thread_local std::unordered_map<std::size_t, Plan> plans;
            const auto found = plans.find(n);
            if (found != plans.end()) return found->second;

            Plan plan;
            plan.reversed.resize(n);
            for (std::size_t i = 1, j = 0; i < n; ++i)
            {
                std::size_t bit = n >> 1;
                for (; j & bit; bit >>= 1) j ^= bit;
                j ^= bit;
                plan.reversed[i] = j;
            }
            plan.cosine.resize(n / 2);
            plan.sine.resize(n / 2);
            for (std::size_t k = 0; k < n / 2; ++k)
            {
                const double angle = -2. * kPi * static_cast<double>(k) / static_cast<double>(n);
                plan.cosine[k] = std::cos(angle);
                plan.sine[k] = std::sin(angle);
            }
            return plans.emplace(n, std::move(plan)).first->second;
        }

        // Iterative Cooley-Tukey, n a power of two. The butterflies are written on
        // the real and imaginary parts: std::complex's product checks for
        // infinities at every call, which costs more than the product.
        void Radix2(cd *a, std::size_t n, bool inverse)
        {
            const Plan &plan = PlanFor(n);
            for (std::size_t i = 1; i < n; ++i)
                if (i < plan.reversed[i]) std::swap(a[i], a[plan.reversed[i]]);

            double *x = reinterpret_cast<double *>(a);             // (re, im) pairs
            const double sign = inverse ? -1. : 1.;
            for (std::size_t len = 2; len <= n; len <<= 1)
            {
                const std::size_t half = len / 2, stride = n / len;
                for (std::size_t i = 0; i < n; i += len)
                    for (std::size_t j = 0; j < half; ++j)
                    {
                        const double wr = plan.cosine[j * stride], wi = sign * plan.sine[j * stride];
                        double *p = x + 2 * (i + j), *q = x + 2 * (i + j + half);
                        const double vr = q[0] * wr - q[1] * wi, vi = q[0] * wi + q[1] * wr;
                        q[0] = p[0] - vr;
                        q[1] = p[1] - vi;
                        p[0] += vr;
                        p[1] += vi;
                    }
            }
        }

        // Lengths with factors 3 and 5 (48, 96, 100, 120 ...): split by the odd
        // factor into shorter transforms and recombine, down to a power of two.
        // The n-th roots of unity are kept per length and per thread.
        const std::vector<cd> &RootsFor(std::size_t n)
        {
            thread_local std::unordered_map<std::size_t, std::vector<cd>> roots;
            const auto found = roots.find(n);
            if (found != roots.end()) return found->second;
            std::vector<cd> table(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                const double angle = -2. * kPi * static_cast<double>(k) / static_cast<double>(n);
                table[k] = cd(std::cos(angle), std::sin(angle));
            }
            return roots.emplace(n, std::move(table)).first->second;
        }

        std::size_t OddFactor(std::size_t n) { return n % 3 == 0 ? 3 : (n % 5 == 0 ? 5 : 0); }

        bool IsSmooth(std::size_t n)
        {
            for (std::size_t p : {std::size_t(2), std::size_t(3), std::size_t(5)})
                while (n % p == 0) n /= p;
            return n == 1;
        }

        // out[0..n) = transform of in[0], in[stride], ..., in[(n-1) stride].
        void MixedRadix(const cd *in, std::size_t stride, cd *out, std::size_t n, bool inverse)
        {
            if (IsPowerOfTwo(n))
            {
                for (std::size_t k = 0; k < n; ++k) out[k] = in[k * stride];
                if (n > 1) Radix2(out, n, inverse);
                return;
            }
            const std::size_t p = OddFactor(n), m = n / p;
            std::vector<cd> parts(n);
            for (std::size_t r = 0; r < p; ++r) MixedRadix(in + r * stride, stride * p, parts.data() + r * m, m, inverse);

            const std::vector<cd> &roots = RootsFor(n);
            const double *w = reinterpret_cast<const double *>(roots.data());
            const double *y = reinterpret_cast<const double *>(parts.data());
            const double sign = inverse ? -1. : 1.;
            for (std::size_t q = 0; q < p; ++q)
                for (std::size_t k = 0; k < m; ++k)
                {
                    const std::size_t index = k + q * m;
                    double re = y[2 * k], im = y[2 * k + 1];              // r = 0
                    for (std::size_t r = 1; r < p; ++r)
                    {
                        const std::size_t t = (r * index) % n;
                        const double wr = w[2 * t], wi = sign * w[2 * t + 1];
                        const double yr = y[2 * (r * m + k)], yi = y[2 * (r * m + k) + 1];
                        re += yr * wr - yi * wi;
                        im += yr * wi + yi * wr;
                    }
                    out[index] = cd(re, im);
                }
        }

        // Bluestein, for any other length: x_k e^{-i pi k^2 / n} convolved with the
        // chirp, through power-of-two transforms of length m >= 2n - 1. The chirp
        // and the transform of its mirror are kept per length and direction.
        struct Chirp
        {
            std::size_t m = 0;
            std::vector<cd> chirp;      // (n)
            std::vector<cd> kernel;     // (m) transform of the mirrored conjugate chirp
        };

        const Chirp &ChirpFor(std::size_t n, bool inverse)
        {
            thread_local std::unordered_map<std::size_t, Chirp> chirps[2];
            auto &table = chirps[inverse ? 1 : 0];
            const auto found = table.find(n);
            if (found != table.end()) return found->second;

            Chirp c;
            c.m = 1;
            while (c.m < 2 * n - 1) c.m <<= 1;
            const double sign = inverse ? 1. : -1.;
            c.chirp.resize(n);
            for (std::size_t k = 0; k < n; ++k)
            {
                // k^2 mod 2n keeps the angle small for long transforms.
                const double phase = sign * kPi * static_cast<double>((k * k) % (2 * n)) / static_cast<double>(n);
                c.chirp[k] = cd(std::cos(phase), std::sin(phase));
            }
            c.kernel.assign(c.m, cd(0., 0.));
            c.kernel[0] = std::conj(c.chirp[0]);
            for (std::size_t k = 1; k < n; ++k) c.kernel[k] = c.kernel[c.m - k] = std::conj(c.chirp[k]);
            Radix2(c.kernel.data(), c.m, false);
            return table.emplace(n, std::move(c)).first->second;
        }

        void Bluestein(std::vector<cd> &a, bool inverse)
        {
            const std::size_t n = a.size();
            const Chirp &c = ChirpFor(n, inverse);
            const std::size_t m = c.m;

            thread_local std::vector<cd> A;
            A.assign(m, cd(0., 0.));
            for (std::size_t k = 0; k < n; ++k) A[k] = a[k] * c.chirp[k];
            Radix2(A.data(), m, false);
            double *x = reinterpret_cast<double *>(A.data());
            const double *y = reinterpret_cast<const double *>(c.kernel.data());
            for (std::size_t k = 0; k < m; ++k)
            {
                const double re = x[2 * k] * y[2 * k] - x[2 * k + 1] * y[2 * k + 1];
                const double im = x[2 * k] * y[2 * k + 1] + x[2 * k + 1] * y[2 * k];
                x[2 * k] = re;
                x[2 * k + 1] = im;
            }
            Radix2(A.data(), m, true);
            const double norm = 1. / static_cast<double>(m);
            for (std::size_t k = 0; k < n; ++k) a[k] = A[k] * norm * c.chirp[k];
        }
    }

    void Transform(std::vector<cd> &x, bool inverse)
    {
        const std::size_t n = x.size();
        if (n <= 1) return;
        if (IsPowerOfTwo(n)) Radix2(x.data(), n, inverse);
        else if (IsSmooth(n))
        {
            thread_local std::vector<cd> scratch;
            scratch.resize(n);
            MixedRadix(x.data(), 1, scratch.data(), n, inverse);
            std::copy(scratch.begin(), scratch.end(), x.begin());
        }
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
