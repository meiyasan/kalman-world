/**
 * *********************************************
 *
 * \file Unscented.cc
 * \brief Implementation of the Kalman::Engines::Unscented class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Unscented.h"
#include "Kalman/Utils.h"

#include <cmath>
#include <stdexcept>

namespace Kalman::Engines
{
    namespace
    {
        // Rows of sigma points, flat, for the shared accumulation in Utils.
        std::vector<double> Pack(const std::vector<Vector> &pts)
        {
            std::vector<double> flat;
            flat.reserve(pts.size() * pts.front().Size());
            for (const Vector &p : pts) flat.insert(flat.end(), p.Data(), p.Data() + p.Size());
            return flat;
        }

        // sum_i w_i (A_i - a)(B_i - b)^T — the one accumulation every moment here
        // needs, and the same code the Python engine reaches through the C kernels.
        Matrix WeightedOuter(const std::vector<double> &w,
                             const std::vector<Vector> &A, const Vector &a,
                             const std::vector<Vector> &B, const Vector &b)
        {
            Matrix C(a.Size(), b.Size(), 0.);
            Utils::WeightedOuterSum(A.size(), a.Size(), b.Size(), w.data(),
                                    Pack(A).data(), a.Data(), Pack(B).data(), b.Data(), C.Data());
            return C;
        }

        Vector WeightedMean(const std::vector<double> &w, const std::vector<Vector> &Y)
        {
            Vector mean(Y.front().Size(), 0.);
            for (std::size_t s = 0; s < Y.size(); ++s)
                for (std::size_t i = 0; i < mean.Size(); ++i)
                    mean(i) += w[s] * Y[s](i);
            return mean;
        }

        // Symmetrise and add the same 1e-12 ridge the Python engine adds, so a
        // covariance that collapsed to exactly singular still factors next step.
        void Regularise(Matrix &S)
        {
            Utils::Symmetrize(S);
            for (std::size_t i = 0; i < S.Rows(); ++i) S(i, i) += 1e-12;
        }

        Vector ControlOf(const Model &model, const Vector *u)
        {
            Vector uu(model.NControl(), 0.);
            if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;
            return uu;
        }
    }

    Unscented::Unscented(double alpha, double beta, double kappa)
        : fAlpha(alpha), fBeta(beta), fKappa(kappa)
    {
        if (!(alpha > 0.)) throw std::invalid_argument("Engines::Unscented: alpha must be positive.");
    }

    void Unscented::Weights(std::size_t n, double &gamma, std::vector<double> &wm,
                            std::vector<double> &wc) const
    {
        const double nn = static_cast<double>(n);
        const double lambda = fAlpha * fAlpha * (nn + fKappa) - nn;
        gamma = std::sqrt(nn + lambda);

        wm.assign(2 * n + 1, 0.5 / (nn + lambda));
        wc = wm;
        wm[0] = lambda / (nn + lambda);
        wc[0] = lambda / (nn + lambda) + (1. - fAlpha * fAlpha + fBeta);
    }

    std::vector<Vector> Unscented::SigmaPoints(const Vector &mean, const Matrix &P, double gamma)
    {
        const std::size_t n = mean.Size();
        const Matrix L = Utils::CholeskyFactor(P, true);
        if (L.Empty())
            throw std::runtime_error("Engines::Unscented: the covariance has no Cholesky factor "
                                     "(not positive semi-definite even after jitter).");

        std::vector<Vector> pts(2 * n + 1, mean);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t r = 0; r < n; ++r)
            {
                const double d = gamma * L(r, i);
                pts[i + 1](r) += d;
                pts[n + i + 1](r) -= d;
            }
        return pts;
    }

    void Unscented::Propagate(const Model &model, const Vector &x, const Vector *u,
                              double t, Vector &xk, Matrix &F) const
    {
        xk = model.f(x, ControlOf(model, u), t);
        F = Matrix::Identity(model.NState());
    }

    void Unscented::Measure(const Model &model, const Vector &x, double t,
                            Vector &zpred, Matrix &H) const
    {
        zpred = model.h(x, t);
        H = model.JacobianH(x, t);
    }

    void Unscented::Predict(const Model &model, const Vector &x, const Matrix &P,
                            const Vector *u, double t,
                            Vector &xk, Matrix &Pk, Matrix &F) const
    {
        const std::size_t n = model.NState();
        const Vector uu = ControlOf(model, u);

        double gamma;
        std::vector<double> wm, wc;
        Weights(n, gamma, wm, wc);

        const std::vector<Vector> sigma = SigmaPoints(x, P, gamma);
        std::vector<Vector> Y;
        Y.reserve(sigma.size());
        for (const Vector &s : sigma) Y.push_back(model.f(s, uu, t));

        xk = WeightedMean(wm, Y);
        Pk = WeightedOuter(wc, Y, xk, Y, xk);

        Matrix Q = model.Q(t);
        const Matrix L = model.L(t);
        if (!L.Empty()) Q = Utils::Similarity(L, Q);
        Pk += Q;
        Regularise(Pk);

        F = Matrix::Identity(n);
    }

    std::size_t Unscented::Update(const Model &model, const Vector &x, const Matrix &P,
                                  const Vector &z, double t,
                                  Vector &xOut, Matrix &POut) const
    {
        xOut = x;
        POut = P;

        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        if (keep.empty()) return 0;

        double gamma;
        std::vector<double> wm, wc;
        Weights(model.NState(), gamma, wm, wc);

        const std::vector<Vector> sigma = SigmaPoints(x, P, gamma);
        std::vector<Vector> Z;
        Z.reserve(sigma.size());
        for (const Vector &s : sigma) Z.push_back(model.h(s, t));

        const Vector zpred = WeightedMean(wm, Z);
        Matrix S = WeightedOuter(wc, Z, zpred, Z, zpred);
        S += model.R(t);
        const Matrix Pxz = WeightedOuter(wc, sigma, x, Z, zpred);

        // Restrict to the channels that arrived.
        Matrix Sc = Utils::CompressBoth(S, keep);
        Regularise(Sc);
        Matrix Pxzc(model.NState(), keep.size());
        for (std::size_t i = 0; i < model.NState(); ++i)
            for (std::size_t j = 0; j < keep.size(); ++j)
                Pxzc(i, j) = Pxz(i, keep[j]);

        Vector y = Utils::Compress(z, keep);
        y -= Utils::Compress(zpred, keep);

        const Matrix K = Pxzc * Utils::Inverse(Sc);

        xOut += K * y;
        POut -= Utils::Similarity(K, Sc);              // P - K S K^T
        Regularise(POut);

        return keep.size();
    }
}
