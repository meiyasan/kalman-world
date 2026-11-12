/**
 * *********************************************
 *
 * \file Cholesky.cc
 * \brief Implementation of the Cholesky factorisation and its solves
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Algebra/Cholesky.h"
#include "Kalman/Algebra/Symmetric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Kalman::Algebra
{
    Matrix Cholesky(const Matrix &S)
    {
        const std::size_t n = S.Rows();
        Matrix L(n, n, 0.);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j <= i; ++j)
            {
                double sum = S(i, j);
                for (std::size_t k = 0; k < j; ++k) sum -= L(i, k) * L(j, k);
                if (i == j)
                {
                    if (!(sum > 0.)) return Matrix();
                    L(i, j) = std::sqrt(sum);
                }
                else
                {
                    L(i, j) = sum / L(j, j);
                }
            }
        return L;
    }

    namespace
    {
        Matrix InverseFromCholesky(const Matrix &L)
        {
            const std::size_t n = L.Rows();
            Matrix inv(n, n, 0.);
            for (std::size_t c = 0; c < n; ++c)
            {
                Vector y(n, 0.), x(n, 0.);
                for (std::size_t i = 0; i < n; ++i)          // forward substitution
                {
                    double sum = (i == c) ? 1. : 0.;
                    for (std::size_t k = 0; k < i; ++k) sum -= L(i, k) * y(k);
                    y(i) = sum / L(i, i);
                }
                for (std::size_t ii = n; ii-- > 0;)          // back substitution
                {
                    double sum = y(ii);
                    for (std::size_t k = ii + 1; k < n; ++k) sum -= L(k, ii) * x(k);
                    x(ii) = sum / L(ii, ii);
                }
                for (std::size_t i = 0; i < n; ++i) inv(i, c) = x(i);
            }
            return inv;
        }
    }

    Matrix Inverse(const Matrix &S, bool *ok)
    {
        const std::size_t n = S.Rows();
        const Matrix sym = SymmetricPart(S);

        // Cholesky is both the cheapest route and the one that preserves the symmetry
        // we expect of a covariance, so it is tried first and unmodified.
        if (const Matrix L = Cholesky(sym); !L.Empty())
        {
            if (ok) *ok = true;
            return InverseFromCholesky(L);
        }

        // Not positive definite: build the Moore-Penrose pseudo-inverse from the
        // eigendecomposition, dropping the null directions instead of dividing by
        // (nearly) zero. This is what numpy.linalg.pinv does on the Python side.
        //
        // Note there is deliberately NO Tikhonov ladder here, unlike
        // CholeskyFactor(). Ridging a singular innovation covariance would let the
        // Cholesky succeed on `S + 1e-12 I` and hand back an inverse of order 1e12
        // -- a colossal Kalman gain conjured out of a direction the measurement
        // says nothing about. Dropping the direction is the honest answer.
        if (ok) *ok = false;

        Vector values;
        Matrix vectors;
        SymmetricEigen(sym, values, vectors);

        double vmax = 0.;
        for (std::size_t i = 0; i < n; ++i) vmax = std::max(vmax, std::abs(values(i)));
        const double tol = vmax * static_cast<double>(n) * 1e-15;

        Matrix inv(n, n, 0.);
        for (std::size_t k = 0; k < n; ++k)
        {
            if (std::abs(values(k)) <= tol) continue;
            const double w = 1. / values(k);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    inv(i, j) += w * vectors(i, k) * vectors(j, k);
        }
        return inv;
    }

    Vector Solve(const Matrix &S, const Vector &b)
    {
        // L L^T x = b by two substitutions; the inverse is never formed. A matrix
        // that is not positive-definite goes to Inverse(), which degrades gracefully.
        const std::size_t n = S.Rows();
        const Matrix L = Cholesky(SymmetricPart(S));
        if (L.Empty()) return Inverse(S) * b;

        Vector y(n, 0.), x(n, 0.);
        for (std::size_t i = 0; i < n; ++i)
        {
            double sum = b(i);
            for (std::size_t k = 0; k < i; ++k) sum -= L(i, k) * y(k);
            y(i) = sum / L(i, i);
        }
        for (std::size_t ii = n; ii-- > 0;)
        {
            double sum = y(ii);
            for (std::size_t k = ii + 1; k < n; ++k) sum -= L(k, ii) * x(k);
            x(ii) = sum / L(ii, ii);
        }
        return x;
    }

    Matrix CholeskyFactor(const Matrix &S, bool jitter)
    {
        const std::size_t n = S.Rows();
        const Matrix sym = SymmetricPart(S);

        const int attempts = jitter ? 12 : 1;
        for (int attempt = 0; attempt < attempts; ++attempt)
        {
            Matrix reg = sym;
            if (attempt > 0)
            {
                const double ridge = std::pow(10., -18. + attempt);
                for (std::size_t i = 0; i < n; ++i) reg(i, i) += ridge;
            }
            const Matrix L = Cholesky(reg);
            if (!L.Empty()) return L;
        }
        return Matrix();
    }
}
