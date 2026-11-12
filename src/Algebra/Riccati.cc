/**
 * *********************************************
 *
 * \file Riccati.cc
 * \brief Implementation of the Riccati and Stein solvers
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Algebra/Riccati.h"
#include "Kalman/Algebra/Cholesky.h"
#include "Kalman/Algebra/Symmetric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace Kalman::Algebra
{
    namespace
    {
        // Both return NaN as soon as an entry is not a number (std::max would skip it),
        // so a diverging iteration cannot look converged.
        double MaxAbs(const Matrix &M)
        {
            double out = 0.;
            for (std::size_t i = 0; i < M.Rows() * M.Cols(); ++i)
            {
                const double v = std::abs(M.Data()[i]);
                if (std::isnan(v)) return v;
                out = std::max(out, v);
            }
            return out;
        }

        double MaxAbsDifference(const Matrix &a, const Matrix &b)
        {
            double out = 0.;
            for (std::size_t i = 0; i < a.Rows() * a.Cols(); ++i)
            {
                const double v = std::abs(a.Data()[i] - b.Data()[i]);
                if (std::isnan(v)) return v;
                out = std::max(out, v);
            }
            return out;
        }

        void CheckShapes(const Matrix &A, const Matrix &B, const Matrix &Q, const Matrix &R)
        {
            const std::size_t n = A.Rows(), l = B.Cols();
            if (A.Cols() != n || B.Rows() != n || l == 0 || Q.Rows() != n || Q.Cols() != n
                || R.Rows() != l || R.Cols() != l)
                throw std::invalid_argument("Algebra::SolveDARE: shapes must be A (n, n), B (n, l), Q (n, n), R (l, l).");
        }
    }

    Matrix SolveStein(const Matrix &A, const Matrix &W, std::size_t maxDoublings)
    {
        const std::size_t n = A.Rows();
        if (A.Cols() != n || W.Rows() != n || W.Cols() != n)
            throw std::invalid_argument("Algebra::SolveStein: A and W must be (n, n).");

        constexpr double eps = std::numeric_limits<double>::epsilon();
        Matrix X = W, Ak = A;
        for (std::size_t j = 0; j < maxDoublings; ++j)
        {
            const Matrix increment = Ak.Transpose() * X * Ak;
            X += increment;
            const double size = MaxAbs(X), step = MaxAbs(increment);
            if (!std::isfinite(size)) break;
            // The terms left are at most the last increment again, shrinking
            // geometrically; below rounding the sum is done.
            if (step <= eps * size || size == 0.) { Symmetrize(X); return X; }
            Ak = Ak * Ak;
        }
        throw std::runtime_error("Algebra::SolveStein: the sum does not converge (is A Schur stable?).");
    }

    Matrix RiccatiGain(const Matrix &A, const Matrix &B, const Matrix &R, const Matrix &P)
    {
        const Matrix Bt = B.Transpose();
        Matrix S = Bt * P * B;
        S += R;
        return Inverse(S) * (Bt * P * A);
    }

    namespace
    {
        /// Newton (Hewer, 1971) from \p P: the cost of the current gain, a Stein
        /// equation, is the next P. From any stabilising gain it decreases
        /// monotonically to the stabilising solution, quadratically at the end.
        /// False, with \p P untouched, when the gain of \p P does not stabilise
        /// the pair or the steps do not settle.
        bool Newton(const Matrix &A, const Matrix &B, const Matrix &Q, const Matrix &R, Matrix &P)
        {
            constexpr double eps = std::numeric_limits<double>::epsilon();
            Matrix current = P;
            double previous = std::numeric_limits<double>::infinity();
            for (std::size_t it = 0; it < 100; ++it)
            {
                const Matrix K = RiccatiGain(A, B, R, current);
                Matrix closed = A;
                closed -= B * K;
                Matrix W = K.Transpose() * R * K;
                W += Q;

                Matrix next;
                try { next = SolveStein(closed, W); }
                catch (const std::runtime_error &) { return false; }

                const double delta = MaxAbsDifference(next, current), scale = std::max(1., MaxAbs(next));
                if (!std::isfinite(delta)) return false;
                current = next;
                // Rounding reached, or, close to it, no more progress.
                if (delta <= 8. * eps * scale || (delta >= previous && delta <= 1e-10 * scale))
                {
                    P = current;
                    return true;
                }
                previous = delta;
            }
            return false;
        }
    }

    Matrix SolveDARE(const Matrix &A, const Matrix &B, const Matrix &Q, const Matrix &R,
                     double tolerance, std::size_t maxIterations)
    {
        CheckShapes(A, B, Q, R);

        // The Riccati recursion from P = Q converges, monotonically, to the stabilising
        // solution of a stabilisable pair, but linearly, at the square of the slowest
        // closed-loop pole: near the unit circle it is slow and stops short of
        // rounding (measured: 6e-8 from scipy for a pole at 0.999). Its gain, though,
        // stabilises the pair long before P has converged. So Newton is tried at
        // iterations 0, 1, 2, 4, 8 ... and takes over from the first gain that
        // stabilises; the recursion alone answers only if Newton never settles.
        const Matrix At = A.Transpose();
        Matrix P = Q;
        std::size_t nextTry = 0;
        for (std::size_t it = 0; it < maxIterations; ++it)
        {
            if (it == nextTry)
            {
                Matrix polished = P;
                if (Newton(A, B, Q, R, polished)) return polished;
                nextTry = nextTry ? 2 * nextTry : 1;
            }

            const Matrix G = RiccatiGain(A, B, R, P);
            Matrix next = At * P * A;
            next -= (At * P * B) * G;
            next += Q;
            Symmetrize(next);

            const double delta = MaxAbsDifference(next, P), scale = std::max(1., MaxAbs(next));
            P = next;
            if (!std::isfinite(delta)) break;
            if (delta <= tolerance * scale)
            {
                Newton(A, B, Q, R, P);                       // polish if it can; else the recursion's P
                return P;
            }
        }
        throw std::runtime_error("Algebra::SolveDARE: the Riccati iteration did not converge "
                                 "(is the pair (A, B) stabilisable?).");
    }
}
