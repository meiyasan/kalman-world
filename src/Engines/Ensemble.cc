/**
 * *********************************************
 *
 * \file Ensemble.cc
 * \brief Implementation of the Kalman::Engines::Ensemble class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Engines
{
    namespace
    {
        bool AnyNonZero(const Matrix &M)
        {
            for (std::size_t i = 0; i < M.Rows() * M.Cols(); ++i)
                if (M.Data()[i] != 0.) return true;
            return false;
        }
    }

    Ensemble::Ensemble(std::size_t nMembers, double inflation, std::uint64_t seed)
        : fNMembers(nMembers), fInflation(inflation), fRandom(seed)
    {
        if (nMembers < 2) throw std::invalid_argument("Engines::Ensemble: at least two members.");
        if (!(inflation > 0.)) throw std::invalid_argument("Engines::Ensemble: inflation must be positive.");
    }

    void Ensemble::SetLocalization(const Matrix &rhoXY, const Matrix &rhoYY)
    {
        fRhoXY = rhoXY;
        fRhoYY = rhoYY;
    }

    Vector Ensemble::Draw(const Matrix &factor) const
    {
        std::normal_distribution<double> gauss(0., 1.);
        Vector w(factor.Cols());
        for (std::size_t i = 0; i < w.Size(); ++i) w(i) = gauss(fRandom);
        return factor * w;
    }

    void Ensemble::Reset(const Model &model)
    {
        const Matrix factor = Utils::CholeskyFactor(model.P0(), true);
        if (factor.Empty())
            throw std::runtime_error("Engines::Ensemble: P0 has no Cholesky factor; cannot seed the members.");

        fMembers.assign(fNMembers, model.X0());
        for (Vector &x : fMembers) x += Draw(factor);
    }

    Vector Ensemble::Mean() const
    {
        if (fMembers.empty()) throw std::runtime_error("Engines::Ensemble: no members; call Reset() first.");
        Vector mean(fMembers.front().Size(), 0.);
        for (const Vector &x : fMembers) mean += x;
        mean *= 1. / static_cast<double>(fMembers.size());
        return mean;
    }

    void Ensemble::Estimate(Vector &mean, Matrix &cov) const
    {
        mean = Mean();
        const std::size_t n = mean.Size();
        const double norm = 1. / static_cast<double>(std::max<std::size_t>(fMembers.size() - 1, 1));

        cov = Matrix(n, n, 0.);
        for (const Vector &x : fMembers)
            for (std::size_t i = 0; i < n; ++i)
            {
                const double di = (x(i) - mean(i)) * norm;
                for (std::size_t j = 0; j < n; ++j) cov(i, j) += di * (x(j) - mean(j));
            }
        Utils::Symmetrize(cov);
        for (std::size_t i = 0; i < n; ++i) cov(i, i) += 1e-12;
    }

    double Ensemble::Spread() const
    {
        const Vector mean = Mean();
        const std::size_t n = mean.Size();
        double total = 0.;
        for (std::size_t i = 0; i < n; ++i)
        {
            double v = 0.;
            for (const Vector &x : fMembers) v += (x(i) - mean(i)) * (x(i) - mean(i));
            total += std::sqrt(v / static_cast<double>(fMembers.size()));
        }
        return total / static_cast<double>(n);
    }

    Vector Ensemble::Forecast(const Model &model, const Vector *u, double t) const
    {
        if (fMembers.empty()) const_cast<Ensemble *>(this)->Reset(model);

        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;

        Matrix Q = model.Q(t);
        const Matrix L = model.L(t);
        if (!L.Empty()) Q = Utils::Similarity(L, Q);
        const Matrix factor = AnyNonZero(Q) ? Utils::CholeskyFactor(Q, true) : Matrix();

        for (Vector &x : fMembers)
        {
            x = model.f(x, uu, t);
            if (!factor.Empty()) x += Draw(factor);
        }
        return Mean();
    }

    Vector Ensemble::Analyze(const Model &model, const Vector &z, double t) const
    {
        if (fMembers.empty()) const_cast<Ensemble *>(this)->Reset(model);

        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        if (keep.empty()) return Mean();

        const std::size_t n = model.NState();
        const std::size_t p = keep.size();
        const std::size_t N = fMembers.size();
        const double norm = 1. / static_cast<double>(N - 1);

        // Inflate the anomalies before the analysis.
        Vector mean = Mean();
        if (fInflation != 1.)
            for (Vector &x : fMembers)
            {
                Vector a = x; a -= mean;
                x = mean + fInflation * a;
            }

        // Observation-space ensemble, restricted to the channels that arrived.
        std::vector<Vector> Y;
        Y.reserve(N);
        for (const Vector &x : fMembers) Y.push_back(Utils::Compress(model.h(x, t), keep));
        Vector ymean(p, 0.);
        for (const Vector &y : Y) ymean += y;
        ymean *= 1. / static_cast<double>(N);

        Matrix PHt(n, p, 0.), HPHt(p, p, 0.);
        for (std::size_t s = 0; s < N; ++s)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                const double xa = (fMembers[s](i) - mean(i)) * norm;
                for (std::size_t j = 0; j < p; ++j) PHt(i, j) += xa * (Y[s](j) - ymean(j));
            }
            for (std::size_t i = 0; i < p; ++i)
            {
                const double ya = (Y[s](i) - ymean(i)) * norm;
                for (std::size_t j = 0; j < p; ++j) HPHt(i, j) += ya * (Y[s](j) - ymean(j));
            }
        }

        if (!fRhoXY.Empty())
        {
            if (fRhoXY.Rows() != n || fRhoXY.Cols() != model.NMeasure())
                throw std::runtime_error("Engines::Ensemble: rho_xy must be (n, m).");
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < p; ++j) PHt(i, j) *= fRhoXY(i, keep[j]);
        }
        if (!fRhoYY.Empty())
        {
            if (fRhoYY.Rows() != model.NMeasure() || fRhoYY.Cols() != model.NMeasure())
                throw std::runtime_error("Engines::Ensemble: rho_yy must be (m, m).");
            for (std::size_t i = 0; i < p; ++i)
                for (std::size_t j = 0; j < p; ++j) HPHt(i, j) *= fRhoYY(keep[i], keep[j]);
        }

        const Matrix R = Utils::CompressBoth(model.R(t), keep);
        Matrix S = HPHt; S += R;
        const Matrix K = PHt * Utils::Inverse(S);

        const Matrix Rfac = Utils::CholeskyFactor(R, true);
        const Vector zc = Utils::Compress(z, keep);
        for (std::size_t s = 0; s < N; ++s)
        {
            Vector innovation = zc;                      // z + eps - h(x_s)
            if (!Rfac.Empty()) innovation += Draw(Rfac);
            innovation -= Y[s];
            fMembers[s] += K * innovation;
        }
        return Mean();
    }

    void Ensemble::Propagate(const Model &model, const Vector &x, const Vector *u,
                             double t, Vector &xk, Matrix &F) const
    {
        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;
        xk = model.f(x, uu, t);
        F = Matrix::Identity(model.NState());
    }

    void Ensemble::Measure(const Model &model, const Vector &x, double t,
                           Vector &zpred, Matrix &H) const
    {
        zpred = model.h(x, t);
        H = model.JacobianH(x, t);
    }

    void Ensemble::Predict(const Model &model, const Vector &, const Matrix &,
                           const Vector *u, double t,
                           Vector &xk, Matrix &Pk, Matrix &F) const
    {
        Forecast(model, u, t);
        Estimate(xk, Pk);
        F = Matrix::Identity(model.NState());
    }

    std::size_t Ensemble::Update(const Model &model, const Vector &, const Matrix &,
                                 const Vector &z, double t,
                                 Vector &xOut, Matrix &POut) const
    {
        Analyze(model, z, t);
        Estimate(xOut, POut);
        return Utils::ValidChannels(z).size();
    }

    void Ensemble::Smooth(const Vector &, const Matrix &, const Vector &, const Matrix &,
                          const Vector &, const Matrix &, const Matrix &,
                          Vector &, Matrix &) const
    {
        throw std::runtime_error("Engines::Ensemble: an ensemble has no RTS smoother; use Forward() only.");
    }
}
