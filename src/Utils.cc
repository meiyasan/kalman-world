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
#include <stdexcept>
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

    const char *ToString(DynamicsKind kind)
    {
        return kind == DynamicsKind::Continuous ? "continuous" : "discrete";
    }

    const char *ToString(DifferentiatorKind kind)
    {
        switch (kind)
        {
            case DifferentiatorKind::Analytic:  return "analytic";
            case DifferentiatorKind::Forward:   return "forward";
            case DifferentiatorKind::Central:   return "central";
            case DifferentiatorKind::Automatic: return "automatic";
        }
        return "unknown";
    }

    const char *ToString(IntegratorKind kind)
    {
        switch (kind)
        {
            case IntegratorKind::Euler:         return "Euler";
            case IntegratorKind::RK4:           return "RK4";
            case IntegratorKind::DormandPrince: return "DormandPrince";
            case IntegratorKind::ImplicitEuler: return "ImplicitEuler";
        }
        return "Unknown";
    }

    const char *ToString(SolverKind kind)
    {
        switch (kind)
        {
            case SolverKind::Spectral:         return "Spectral";
            case SolverKind::FiniteDifference: return "FiniteDifference";
            case SolverKind::FiniteVolume:     return "FiniteVolume";
            case SolverKind::FiniteElement:    return "FiniteElement";
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

    void Correct(const Matrix &Hc, const Matrix &Rc, const Vector &yc,
                 const Vector &x, const Matrix &P, Vector &xOut, Matrix &POut)
    {
        Matrix S = Similarity(Hc, P);                  // H P H^T
        S += Rc;
        const Matrix K = P * Hc.Transpose() * Inverse(S);
        xOut = x;
        xOut += K * yc;
        POut = JosephForm(K, Hc, P, Rc);
    }

    void WeightedOuterSum(std::size_t nsigma, std::size_t na, std::size_t nb,
                          const double *w, const double *A, const double *a,
                          const double *B, const double *b, double *out)
    {
        for (std::size_t i = 0; i < na * nb; ++i) out[i] = 0.;
        for (std::size_t s = 0; s < nsigma; ++s)
            for (std::size_t i = 0; i < na; ++i)
            {
                const double da = w[s] * (A[s * na + i] - a[i]);
                if (da == 0.) continue;
                for (std::size_t j = 0; j < nb; ++j) out[i * nb + j] += da * (B[s * nb + j] - b[j]);
            }
    }

    void SystematicResample(std::size_t n, const double *weights, double u0, int *indices)
    {
        // Walk the cumulative weight once; the last entry is pinned to 1 so
        // round-off in the sum cannot leave the final positions unmatched.
        double cumulative = weights[0];
        std::size_t j = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double position = (static_cast<double>(i) + u0) / static_cast<double>(n);
            while (j + 1 < n && cumulative < position) cumulative += weights[++j];
            indices[i] = static_cast<int>(j);
        }
    }

}
