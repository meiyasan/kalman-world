/**
 * *********************************************
 *
 * \file Symmetric.cc
 * \brief Implementation of the symmetric-matrix operations
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Algebra/Symmetric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Kalman::Algebra
{
    Matrix SymmetricPart(const Matrix &S)
    {
        const std::size_t n = S.Rows();
        Matrix out(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                out(i, j) = 0.5 * (S(i, j) + S(j, i));
        return out;
    }

    Matrix Similarity(const Matrix &A, const Matrix &P)
    {
        return A * P * A.Transpose();
    }

    void Symmetrize(Matrix &P)
    {
        const std::size_t n = P.Rows();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
            {
                const double avg = 0.5 * (P(i, j) + P(j, i));
                P(i, j) = avg;
                P(j, i) = avg;
            }
    }

    bool SymmetricEigen(const Matrix &S, Vector &values, Matrix &vectors)
    {
        const std::size_t n = S.Rows();
        Matrix A = SymmetricPart(S);
        vectors = Matrix::Identity(n);

        // Cyclic Jacobi: repeatedly zero the largest off-diagonal entry by a plane
        // rotation. Converges quadratically and needs no pivoting or workspace.
        for (int sweep = 0; sweep < 100; ++sweep)
        {
            double off = 0.;
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = i + 1; j < n; ++j) off += A(i, j) * A(i, j);
            if (off <= 1e-30)
            {
                values.Resize(n);
                for (std::size_t i = 0; i < n; ++i) values(i) = A(i, i);
                return true;
            }

            for (std::size_t p = 0; p < n; ++p)
                for (std::size_t q = p + 1; q < n; ++q)
                {
                    if (std::abs(A(p, q)) < 1e-300) continue;

                    const double theta = (A(q, q) - A(p, p)) / (2. * A(p, q));
                    const double sign = theta >= 0. ? 1. : -1.;
                    const double t = sign / (std::abs(theta) + std::sqrt(theta * theta + 1.));
                    const double c = 1. / std::sqrt(t * t + 1.);
                    const double s = t * c;

                    for (std::size_t k = 0; k < n; ++k)
                    {
                        const double akp = A(k, p), akq = A(k, q);
                        A(k, p) = c * akp - s * akq;
                        A(k, q) = s * akp + c * akq;
                    }
                    for (std::size_t k = 0; k < n; ++k)
                    {
                        const double apk = A(p, k), aqk = A(q, k);
                        A(p, k) = c * apk - s * aqk;
                        A(q, k) = s * apk + c * aqk;
                    }
                    for (std::size_t k = 0; k < n; ++k)
                    {
                        const double vkp = vectors(k, p), vkq = vectors(k, q);
                        vectors(k, p) = c * vkp - s * vkq;
                        vectors(k, q) = s * vkp + c * vkq;
                    }
                }
        }

        values.Resize(n);
        for (std::size_t i = 0; i < n; ++i) values(i) = A(i, i);
        return false;
    }

    double ConditionNumber(const Matrix &M)
    {
        Vector values;
        Matrix vectors;
        SymmetricEigen(M, values, vectors);

        double vmin = std::numeric_limits<double>::infinity(), vmax = 0.;
        for (std::size_t i = 0; i < values.Size(); ++i)
        {
            const double a = std::abs(values(i));
            vmin = std::min(vmin, a);
            vmax = std::max(vmax, a);
        }
        if (vmin <= 0.) return std::numeric_limits<double>::infinity();
        return vmax / vmin;
    }
}
