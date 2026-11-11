/**
 * *********************************************
 *
 * \file IKF.cc
 * \brief Implementation of the Kalman::Filters::IKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Filters/IKF.h"
#include "Kalman/Engines/Extended.h"
#include "Kalman/Engines/Unscented.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Engines/Linear.h"
#include "Kalman/Routers/Bayesian.h"
#include "Kalman/Utils.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace Kalman::Filters
{
    namespace
    {
        std::unique_ptr<Engine> MakeEngine(EngineKind kind)
        {
            switch (kind)
            {
                case EngineKind::Linear:    return std::make_unique<Engines::Linear>();
                case EngineKind::Extended:  return std::make_unique<Engines::Extended>();
                case EngineKind::Unscented: return std::make_unique<Engines::Unscented>();
                case EngineKind::Combinatorial: return std::make_unique<Engines::Combinatorial>();
                // Particle and Ensemble are deliberately absent: the IKF mixing step
                // hands each mode a Gaussian (x, P) to predict from, and a sample-set
                // engine ignores that argument -- the modes would silently decouple.
                default: break;
            }
            std::ostringstream os;
            os << "Kalman::Filters::IKF: no C++ engine implements the '" << ToString(kind)
               << "' kind requested by one of the modes.";
            throw std::runtime_error(os.str());
        }
    }

    IKF::IKF(const std::vector<Model *> &models, const Matrix &Pi, const Routers::Router *router)
        : fModels(models), fPi(Pi)
    {
        const std::size_t M = fModels.size();
        if (M < 2) throw std::runtime_error("Kalman::Filters::IKF: at least two models are required.");
        if (Pi.Rows() != M || Pi.Cols() != M)
        {
            std::ostringstream os;
            os << "Kalman::Filters::IKF: Pi has shape (" << Pi.Rows() << ", " << Pi.Cols()
               << "), expected (" << M << ", " << M << ").";
            throw std::runtime_error(os.str());
        }

        for (std::size_t i = 0; i < M; ++i)
        {
            if (!fModels[i]) throw std::runtime_error("Kalman::Filters::IKF: a model is null.");
            fModels[i]->CheckShapes();

            // Mixing averages states across modes, so it is only meaningful when every
            // mode means the same thing by "the state".
            if (fModels[i]->NState() != fModels[0]->NState() ||
                fModels[i]->NMeasure() != fModels[0]->NMeasure())
            {
                std::ostringstream os;
                os << "Kalman::Filters::IKF: model '" << fModels[i]->Name() << "' has (n, m) = ("
                   << fModels[i]->NState() << ", " << fModels[i]->NMeasure() << ") but '"
                   << fModels[0]->Name() << "' has (" << fModels[0]->NState() << ", "
                   << fModels[0]->NMeasure() << "). Every mode must share both.";
                throw std::runtime_error(os.str());
            }

            // Each mode brings its own engine, so a linear and a nonlinear mode can
            // coexist in one bank.
            fEngines.push_back(MakeEngine(fModels[i]->DefaultEngine()));
        }

        for (std::size_t i = 0; i < M; ++i)
        {
            double row = 0.;
            for (std::size_t j = 0; j < M; ++j)
            {
                if (Pi(i, j) < 0.)
                    throw std::runtime_error("Kalman::Filters::IKF: Pi has a negative entry.");
                row += Pi(i, j);
            }
            if (std::abs(row - 1.) > 1e-9)
            {
                std::ostringstream os;
                os << "Kalman::Filters::IKF: row " << i << " of Pi sums to " << row
                   << ", expected 1.";
                throw std::runtime_error(os.str());
            }
        }

        if (router) fRouter = router;
        else
        {
            fOwnRouter = std::make_unique<Routers::Bayesian>();
            fRouter = fOwnRouter.get();
        }

        Reset();
    }

    void IKF::SetRouter(const Routers::Router *router)
    {
        if (!router) throw std::runtime_error("Kalman::Filters::IKF: the router must not be null.");
        fRouter = router;
    }

    void IKF::Reset()
    {
        const std::size_t M = fModels.size();

        fMu = Vector(M, 1. / static_cast<double>(M));
        fX.clear();
        fP.clear();
        for (std::size_t j = 0; j < M; ++j)
        {
            fX.push_back(fModels[j]->X0());
            fP.push_back(fModels[j]->P0());
        }

        fNis = Matrix();
        fModeHistory = Matrix();
        fCombined.clear();
        fCombinedP.clear();
        fTime.clear();
    }

    void IKF::SetModeProbabilities(const Vector &mu)
    {
        if (mu.Size() != fModels.size())
            throw std::runtime_error("Kalman::Filters::IKF: mu must have one entry per model.");
        fMu = Routers::Router::Normalise(mu);
    }

    double IKF::Likelihood(const Vector &nu, const Matrix &S)
    {
        const std::size_t m = nu.Size();
        if (m == 0) return 1.;                       // nothing measured: no evidence either way

        const Matrix Sinv = Utils::Inverse(S);
        const double quad = Utils::Mahalanobis(nu, Sinv);

        // |S| from the Cholesky factor when it exists; the eigenvalues otherwise.
        double logdet = 0.;
        const Matrix L = Utils::CholeskyFactor(S, /* jitter */ true);
        if (!L.Empty())
        {
            for (std::size_t i = 0; i < m; ++i) logdet += 2. * std::log(std::abs(L(i, i)));
        }
        else
        {
            Vector values;
            Matrix vectors;
            Utils::SymmetricEigen(S, values, vectors);
            for (std::size_t i = 0; i < m; ++i)
                if (std::abs(values(i)) > 1e-300) logdet += std::log(std::abs(values(i)));
        }

        const double logL = -0.5 * (quad + logdet + static_cast<double>(m) * std::log(2. * M_PI));
        // Floored, not clamped to zero: a mode whose likelihood underflows must stay
        // recoverable, and exp() of a very negative number is exactly what underflows.
        return std::max(std::exp(logL), 1e-300);
    }

    Record IKF::Forward(const std::vector<Vector> &zk)
    {
        if (zk.empty()) throw std::runtime_error("Kalman::Filters::IKF::Forward: no measurements given.");

        const std::size_t M = fModels.size();
        const std::size_t m = fModels[0]->NMeasure();
        const std::size_t n = fModels[0]->NState();
        const std::size_t N = zk.size();

        Reset();

        fNis = Matrix(N, M, 0.);
        fModeHistory = Matrix(N, M, 0.);
        fCombined.reserve(N);
        fCombinedP.reserve(N);
        fTime.reserve(N);

        const double dt = fModels[0]->DeltaT();
        double t = fModels[0]->T0();

        for (std::size_t k = 0; k < N; ++k)
        {
            if (zk[k].Size() != m)
            {
                std::ostringstream os;
                os << "Kalman::Filters::IKF::Forward: measurement " << k << " has "
                   << zk[k].Size() << " channels, expected " << m << ".";
                throw std::runtime_error(os.str());
            }

            // --- 1. Markov prediction ------------------------------------
            Vector c(M, 0.);
            for (std::size_t j = 0; j < M; ++j)
                for (std::size_t i = 0; i < M; ++i) c(j) += fPi(i, j) * fMu(i);

            // --- 2. Mixing: every mode starts from a blend of all modes ----
            // This is what separates an IKF from M independent filters: a mode that
            // becomes relevant inherits the state the others built up.
            std::vector<Vector> xMix(M);
            std::vector<Matrix> pMix(M);
            for (std::size_t j = 0; j < M; ++j)
            {
                Vector x(n, 0.);
                for (std::size_t i = 0; i < M; ++i)
                {
                    const double w = (c(j) > 1e-300) ? fPi(i, j) * fMu(i) / c(j) : 1. / M;
                    x += w * fX[i];
                }

                Matrix P(n, n, 0.);
                for (std::size_t i = 0; i < M; ++i)
                {
                    const double w = (c(j) > 1e-300) ? fPi(i, j) * fMu(i) / c(j) : 1. / M;
                    Vector d(fX[i]);
                    d -= x;

                    Matrix spread(n, n, 0.);
                    for (std::size_t a = 0; a < n; ++a)
                        for (std::size_t b = 0; b < n; ++b) spread(a, b) = d(a) * d(b);

                    Matrix contrib(fP[i]);
                    contrib += spread;
                    contrib *= w;
                    P += contrib;
                }
                Utils::Symmetrize(P);

                xMix[j] = x;
                pMix[j] = P;
            }

            // --- 3. Predict + update each mode from its mixed start --------
            Vector likelihoods(M, 1.);
            for (std::size_t j = 0; j < M; ++j)
            {
                Vector xPrior;
                Matrix PPrior, F;
                fEngines[j]->Predict(*fModels[j], xMix[j], pMix[j], nullptr, t, xPrior, PPrior, F);

                Vector nu;
                Matrix S;
                fEngines[j]->Innovation(*fModels[j], xPrior, PPrior, zk[k], t, nu, S);

                // Missing channels drop out of both the likelihood and the NIS, so a
                // partly-blind step simply carries less evidence rather than none.
                const std::vector<std::size_t> keep = Utils::ValidChannels(nu);
                if (keep.empty())
                {
                    likelihoods(j) = 1.;
                    fNis(k, j) = static_cast<double>(m);   // neutral: the expected value
                }
                else
                {
                    const Vector nuc = Utils::Compress(nu, keep);
                    const Matrix Sc = Utils::CompressBoth(S, keep);
                    likelihoods(j) = Likelihood(nuc, Sc);
                    fNis(k, j) = Utils::Mahalanobis(nuc, Utils::Inverse(Sc));
                }

                Vector xPost;
                Matrix PPost;
                fEngines[j]->Update(*fModels[j], xPrior, PPrior, zk[k], t, xPost, PPost);
                fX[j] = xPost;
                fP[j] = PPost;
            }

            // --- 4. Route: the only decision, and the only pluggable part --
            Matrix nisSoFar(k + 1, M, 0.);
            for (std::size_t r = 0; r <= k; ++r)
                for (std::size_t j = 0; j < M; ++j) nisSoFar(r, j) = fNis(r, j);

            fMu = Routers::Router::Normalise(fRouter->Route(nisSoFar, c, likelihoods));
            for (std::size_t j = 0; j < M; ++j) fModeHistory(k, j) = fMu(j);

            // --- 5. Combination -------------------------------------------
            Vector x(n, 0.);
            for (std::size_t j = 0; j < M; ++j) x += fMu(j) * fX[j];

            Matrix P(n, n, 0.);
            for (std::size_t j = 0; j < M; ++j)
            {
                Vector d(fX[j]);
                d -= x;

                Matrix spread(n, n, 0.);
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = 0; b < n; ++b) spread(a, b) = d(a) * d(b);

                Matrix contrib(fP[j]);
                contrib += spread;
                contrib *= fMu(j);
                P += contrib;
            }
            Utils::Symmetrize(P);

            fCombined.push_back(x);
            fCombinedP.push_back(P);
            fTime.push_back(t);
            t += dt;
        }

        Record record("ikf", StateFlavor::Posterior);
        record.Reserve(N);
        for (std::size_t k = 0; k < N; ++k)
            record.Add(fTime[k], fCombined[k], fCombinedP[k], zk[k]);
        return record;
    }

    std::size_t IKF::DominantMode(std::size_t k) const
    {
        if (k >= fModeHistory.Rows())
            throw std::out_of_range("Kalman::Filters::IKF::DominantMode: step out of range.");

        std::size_t best = 0;
        for (std::size_t j = 1; j < fModeHistory.Cols(); ++j)
            if (fModeHistory(k, j) > fModeHistory(k, best)) best = j;
        return best;
    }
}
