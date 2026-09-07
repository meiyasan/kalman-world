/**
 * *********************************************
 *
 * \file Utils.cc
 * \brief Implementation of the shared linear-algebra helpers
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Utils.h"
#include "Kalman/Constants.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Kalman
{
    const char *ToString(EngineKind kind)
    {
        switch (kind)
        {
            case EngineKind::Linear:    return "Linear";
            case EngineKind::Extended:  return "Extended";
            case EngineKind::Unscented: return "Unscented";
            case EngineKind::Particle:  return "Particle";
            case EngineKind::Ensemble:  return "Ensemble";
            case EngineKind::Combinatorial: return "Combinatorial";
        }
        return "Unknown";
    }

    const char *ToString(StateFlavor flavor)
    {
        switch (flavor)
        {
            case StateFlavor::Prior:     return "Prior";
            case StateFlavor::Posterior: return "Posterior";
            case StateFlavor::Smoothed:  return "Smoothed";
        }
        return "Unknown";
    }
}

namespace Kalman::Utils
{
    namespace
    {
        /// Symmetric part of \p S, so callers never depend on the input being exact.
        Matrix SymmetricPart(const Matrix &S)
        {
            const std::size_t n = S.Rows();
            Matrix out(n, n);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    out(i, j) = 0.5 * (S(i, j) + S(j, i));
            return out;
        }

        /// Lower-triangular Cholesky factor, or an empty matrix if not PD.
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

        /// Inverse from a Cholesky factor, by solving against each unit vector.
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
        return Inverse(S) * b;
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

    std::vector<std::size_t> ValidChannels(const Vector &z)
    {
        std::vector<std::size_t> keep;
        keep.reserve(z.Size());
        for (std::size_t i = 0; i < z.Size(); ++i)
            if (!std::isnan(z(i))) keep.push_back(i);
        return keep;
    }

    bool IsComplete(const Vector &z)
    {
        return !z.AnyNaN();
    }

    Vector Compress(const Vector &v, const std::vector<std::size_t> &keep)
    {
        Vector out(keep.size());
        for (std::size_t a = 0; a < keep.size(); ++a) out(a) = v(keep[a]);
        return out;
    }

    Matrix CompressRows(const Matrix &M, const std::vector<std::size_t> &keep)
    {
        Matrix out(keep.size(), M.Cols());
        for (std::size_t a = 0; a < keep.size(); ++a)
            for (std::size_t j = 0; j < M.Cols(); ++j)
                out(a, j) = M(keep[a], j);
        return out;
    }

    Matrix CompressBoth(const Matrix &M, const std::vector<std::size_t> &keep)
    {
        Matrix out(keep.size(), keep.size());
        for (std::size_t a = 0; a < keep.size(); ++a)
            for (std::size_t b = 0; b < keep.size(); ++b)
                out(a, b) = M(keep[a], keep[b]);
        return out;
    }

    Matrix NumericalJacobian(const std::function<Vector(const Vector &)> &fn,
                             const Vector &x, double eps)
    {
        const std::size_t n = x.Size();
        const Vector f0 = fn(x);
        Matrix J(f0.Size(), n, 0.);

        for (std::size_t j = 0; j < n; ++j)
        {
            const double step = eps * std::max(1., std::abs(x(j)));

            Vector xp(x); xp(j) += step;
            Vector xm(x); xm(j) -= step;

            const Vector fp = fn(xp);
            const Vector fm = fn(xm);
            for (std::size_t i = 0; i < J.Rows(); ++i)
                J(i, j) = (fp(i) - fm(i)) / (2. * step);
        }
        return J;
    }

    double Mahalanobis(const Vector &y, const Matrix &Sinv)
    {
        const Vector Sy = Sinv * y;
        double sum = 0.;
        for (std::size_t i = 0; i < y.Size(); ++i) sum += y(i) * Sy(i);
        return sum;
    }

    Matrix JosephForm(const Matrix &K, const Matrix &H, const Matrix &P, const Matrix &R)
    {
        Matrix IKH = Matrix::Identity(P.Rows());
        IKH -= K * H;

        Matrix out = Similarity(IKH, P);
        out += Similarity(K, R);
        Symmetrize(out);
        return out;
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
