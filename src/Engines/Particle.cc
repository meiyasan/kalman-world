/**
 * *********************************************
 *
 * \file Particle.cc
 * \brief Implementation of the Kalman::Engines::Particle class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Particle.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Engines
{
    Particle::Particle(std::size_t nParticles, double threshold, std::uint64_t seed)
        : fNParticles(nParticles), fThreshold(threshold), fRandom(seed)
    {
        if (nParticles < 2) throw std::invalid_argument("Engines::Particle: at least two particles.");
        if (threshold < 0. || threshold > 1.)
            throw std::invalid_argument("Engines::Particle: the resample threshold is a fraction in [0, 1].");
    }

    Vector Particle::Draw(const Matrix &factor) const
    {
        std::normal_distribution<double> gauss(0., 1.);
        Vector w(factor.Cols());
        for (std::size_t i = 0; i < w.Size(); ++i) w(i) = gauss(fRandom);
        return factor * w;
    }

    void Particle::SetCloud(std::vector<Vector> cloud)
    {
        if (!cloud.empty() && cloud.size() != fNParticles)
            throw std::invalid_argument("Engines::Particle: the cloud must hold one state per particle.");
        fGiven = std::move(cloud);
    }

    void Particle::Reset(const Model &model)
    {
        if (!fGiven.empty())
        {
            for (const Vector &p : fGiven)
                if (p.Size() != model.NState())
                    throw std::invalid_argument("Engines::Particle: the given cloud has states of the wrong size.");
            fParticles = fGiven;
            fWeights.assign(fNParticles, 1. / static_cast<double>(fNParticles));
            fResamples = 0;
            return;
        }
        const Matrix factor = Utils::CholeskyFactor(model.P0(), true);
        if (factor.Empty())
            throw std::runtime_error("Engines::Particle: P0 has no Cholesky factor; cannot seed the cloud.");

        fParticles.assign(fNParticles, model.X0());
        for (Vector &p : fParticles) p += Draw(factor);
        fWeights.assign(fNParticles, 1. / static_cast<double>(fNParticles));
        fResamples = 0;
    }

    void Particle::Estimate(Vector &mean, Matrix &cov) const
    {
        if (fParticles.empty()) throw std::runtime_error("Engines::Particle: no cloud; call Reset() first.");
        const std::size_t n = fParticles.front().Size();

        mean = Vector(n, 0.);
        for (std::size_t s = 0; s < fParticles.size(); ++s)
            for (std::size_t i = 0; i < n; ++i) mean(i) += fWeights[s] * fParticles[s](i);

        cov = Matrix(n, n, 0.);
        for (std::size_t s = 0; s < fParticles.size(); ++s)
            for (std::size_t i = 0; i < n; ++i)
            {
                const double di = fWeights[s] * (fParticles[s](i) - mean(i));
                for (std::size_t j = 0; j < n; ++j) cov(i, j) += di * (fParticles[s](j) - mean(j));
            }
        Utils::Symmetrize(cov);
        for (std::size_t i = 0; i < n; ++i) cov(i, i) += 1e-9;   // the Python engine's ridge
    }

    double Particle::EffectiveSampleSize() const
    {
        double s = 0.;
        for (double w : fWeights) s += w * w;
        return s > 0. ? 1. / s : 0.;
    }

    void Particle::Resample() const
    {
        // Systematic resampling: one uniform draw, N evenly spaced positions.
        const std::size_t N = fParticles.size();
        std::uniform_real_distribution<double> uniform(0., 1.);
        std::vector<int> indices(N);
        const double u = fDraws.uniform ? fDraws.uniform() : uniform(fRandom);
        Utils::SystematicResample(N, fWeights.data(), u, indices.data());

        std::vector<Vector> next;
        next.reserve(N);
        for (std::size_t i = 0; i < N; ++i) next.push_back(fParticles[static_cast<std::size_t>(indices[i])]);
        fParticles.swap(next);
        fWeights.assign(N, 1. / static_cast<double>(N));
        ++fResamples;
    }

    void Particle::Propagate(const Model &model, const Vector &x, const Vector *u,
                             double t, Vector &xk, Matrix &F) const
    {
        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;
        xk = model.f(x, uu, t);
        F = Matrix::Identity(model.NState());
    }

    void Particle::Measure(const Model &model, const Vector &x, double t,
                           Vector &zpred, Matrix &H) const
    {
        zpred = model.h(x, t);
        H = model.JacobianH(x, t);
    }

    void Particle::Predict(const Model &model, const Vector &, const Matrix &,
                           const Vector *u, double t,
                           Vector &xk, Matrix &Pk, Matrix &F) const
    {
        if (fParticles.empty()) const_cast<Particle *>(this)->Reset(model);

        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;

        Matrix Q = model.Q(t);
        const bool through = model.HasProcessNoiseInput();   // w goes into f, not after it
        const Matrix L = model.L(t);
        if (!through && !L.Empty()) Q = Utils::Similarity(L, Q);

        if (fDraws.gaussian)
        {
            // The draws come from outside, all at once, in the order of the particles.
            std::vector<Vector> noise;
            fDraws.gaussian(Q, fParticles.size(), noise);
            if (noise.size() != fParticles.size())
                throw std::runtime_error("Engines::Particle: the noise source returned the wrong number of draws.");
            for (std::size_t s = 0; s < fParticles.size(); ++s)
            {
                if (noise[s].Size() != Q.Rows())
                    throw std::runtime_error("Engines::Particle: a noise draw has the wrong size.");
                if (through) fParticles[s] = model.f(fParticles[s], uu, noise[s], t);
                else
                {
                    fParticles[s] = model.f(fParticles[s], uu, t);
                    fParticles[s] += noise[s];
                }
            }
        }
        else
        {
            bool noisy = false;
            for (std::size_t i = 0; i < Q.Rows() * Q.Cols() && !noisy; ++i) noisy = Q.Data()[i] != 0.;
            const Matrix factor = noisy ? Utils::CholeskyFactor(Q, true) : Matrix();

            for (Vector &p : fParticles)
            {
                if (through)
                    p = model.f(p, uu, factor.Empty() ? Vector(Q.Rows(), 0.) : Draw(factor), t);
                else
                {
                    p = model.f(p, uu, t);
                    if (!factor.Empty()) p += Draw(factor);
                }
            }
        }

        Estimate(xk, Pk);
        F = Matrix::Identity(model.NState());
    }

    std::size_t Particle::Update(const Model &model, const Vector &, const Matrix &,
                                 const Vector &z, double t,
                                 Vector &xOut, Matrix &POut) const
    {
        if (fParticles.empty()) const_cast<Particle *>(this)->Reset(model);

        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        if (keep.empty())
        {
            Estimate(xOut, POut);
            return 0;
        }

        const Vector zc = Utils::Compress(z, keep);
        const Matrix Rinv = Utils::Inverse(Utils::CompressBoth(model.R(t), keep));

        std::vector<double> logw(fParticles.size());
        double top = -INFINITY;
        for (std::size_t s = 0; s < fParticles.size(); ++s)
        {
            Vector dz = zc;
            dz -= Utils::Compress(model.h(fParticles[s], t), keep);
            logw[s] = std::log(fWeights[s] + 1e-300) - 0.5 * Utils::Mahalanobis(dz, Rinv);
            top = std::max(top, logw[s]);
        }

        double total = 0.;
        for (std::size_t s = 0; s < logw.size(); ++s)
        {
            fWeights[s] = std::exp(logw[s] - top);
            total += fWeights[s];
        }
        for (double &w : fWeights) w /= total;

        if (EffectiveSampleSize() < fThreshold * static_cast<double>(fParticles.size())) Resample();

        Estimate(xOut, POut);
        return keep.size();
    }

    void Particle::Smooth(const Vector &, const Matrix &, const Vector &, const Matrix &,
                          const Vector &, const Matrix &, const Matrix &,
                          Vector &, Matrix &) const
    {
        throw std::runtime_error("Engines::Particle: a particle cloud has no RTS smoother; use Forward() only.");
    }
}
