/**
 * *********************************************
 *
 * \file Interacting.cc
 * \brief Source of the Kalman::Engines::Interacting class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Interacting.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Engines/Extended.h"
#include "Kalman/Engines/Linear.h"
#include "Kalman/Engines/Unscented.h"
#include "Kalman/Utils.h"

#include <cmath>
#include <stdexcept>
#include <string>

namespace Kalman::Engines
{
    namespace
    {
        std::unique_ptr<Engine> EngineOf(const Model &mode)
        {
            switch (mode.DefaultEngine())
            {
                case EngineKind::Linear:        return std::make_unique<Linear>();
                case EngineKind::Extended:      return std::make_unique<Extended>();
                case EngineKind::Unscented:     return std::make_unique<Unscented>();
                case EngineKind::Combinatorial: return std::make_unique<Combinatorial>();
                // Particle and Ensemble are deliberately absent: mixing hands each
                // mode a Gaussian (x, P) to predict from, and a sample-set engine
                // ignores that argument: the modes would silently decouple.
                default: break;
            }
            throw std::invalid_argument(std::string("Engines::Interacting: the mode '") + mode.Name() + "' asks for the "
                                        + ToString(mode.DefaultEngine()) + " engine, which cannot step a mode.");
        }

        Matrix Table(const std::vector<std::vector<double>> &rows, std::size_t M)
        {
            Matrix out(rows.size(), M, 0.);
            for (std::size_t k = 0; k < rows.size(); ++k)
                for (std::size_t j = 0; j < M; ++j) out(k, j) = rows[k][j];
            return out;
        }
    }

    const Models::Router &Interacting::Bind(const Model &model) const
    {
        const auto *router = dynamic_cast<const Models::Router *>(&model);
        if (!router)
            throw std::invalid_argument("Engines::Interacting: the model must be a Models::Router (several modes "
                                        "of one system); '" + model.Name() + "' is not.");
        if (router != fModel || fX.size() != router->NModes()) const_cast<Interacting *>(this)->Reset(model);
        return *router;
    }

    void Interacting::Reset(const Model &model)
    {
        const auto *router = dynamic_cast<const Models::Router *>(&model);
        if (!router)
            throw std::invalid_argument("Engines::Interacting: the model must be a Models::Router (several modes "
                                        "of one system); '" + model.Name() + "' is not.");
        fModel = router;
        const std::size_t M = router->NModes();
        fEngines.clear();
        fX.clear();
        fP.clear();
        for (std::size_t j = 0; j < M; ++j)
        {
            fEngines.push_back(EngineOf(router->Mode(j)));
            fEngines.back()->Reset(router->Mode(j));
            fX.push_back(router->Mode(j).X0());
            fP.push_back(router->Mode(j).P0());
        }
        fMu = router->InitialProbabilities();
        fC = fMu;
        fNis.clear();
        fModeHistory.clear();
        std::size_t best = 0;
        for (std::size_t j = 1; j < M; ++j)
            if (fMu(j) > fMu(best)) best = j;
        router->SetLeading(best);
    }

    void Interacting::Propagate(const Model &model, const Vector &x, const Vector *u, double t, Vector &xk, Matrix &F) const
    {
        const Models::Router &router = Bind(model);
        fEngines[router.Leading()]->Propagate(router.Mode(router.Leading()), x, u, t, xk, F);
    }

    void Interacting::Measure(const Model &model, const Vector &x, double t, Vector &zpred, Matrix &H) const
    {
        const Models::Router &router = Bind(model);
        fEngines[router.Leading()]->Measure(router.Mode(router.Leading()), x, t, zpred, H);
    }

    void Interacting::Combine(const Vector &weights, Vector &x, Matrix &P) const
    {
        const std::size_t M = fX.size(), n = fX.front().Size();
        x = Vector(n, 0.);
        for (std::size_t j = 0; j < M; ++j) x += weights(j) * fX[j];
        P = Matrix(n, n, 0.);
        for (std::size_t j = 0; j < M; ++j)
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t b = 0; b < n; ++b)
                    P(a, b) += weights(j) * (fP[j](a, b) + (fX[j](a) - x(a)) * (fX[j](b) - x(b)));
        Utils::Symmetrize(P);
    }

    void Interacting::Predict(const Model &model, const Vector &, const Matrix &, const Vector *u, double t,
                              Vector &xk, Matrix &Pk, Matrix &F) const
    {
        const Models::Router &router = Bind(model);
        const std::size_t M = fX.size(), n = router.NState();
        const Matrix &Pi = router.Transition();

        // --- the probabilities, one step on ---------------------------------------
        Vector c(M, 0.);
        for (std::size_t j = 0; j < M; ++j)
        {
            for (std::size_t i = 0; i < M; ++i) c(j) += Pi(i, j) * fMu(i);
            if (!(c(j) > 1e-12)) c(j) = 1e-12;
        }

        // --- mixing: every mode starts from a blend of all of them ------------------
        std::vector<Vector> xMix(M);
        std::vector<Matrix> pMix(M);
        for (std::size_t j = 0; j < M; ++j)
        {
            Vector x(n, 0.);
            for (std::size_t i = 0; i < M; ++i) x += (Pi(i, j) * fMu(i) / c(j)) * fX[i];
            Matrix P(n, n, 0.);
            for (std::size_t i = 0; i < M; ++i)
            {
                const double w = Pi(i, j) * fMu(i) / c(j);
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = 0; b < n; ++b)
                        P(a, b) += w * (fP[i](a, b) + (fX[i](a) - x(a)) * (fX[i](b) - x(b)));
            }
            xMix[j] = x;
            pMix[j] = P;
        }

        // --- each mode predicts from its mixed start ---------------------------------
        std::vector<Matrix> Fs(M);
        for (std::size_t j = 0; j < M; ++j)
        {
            Vector xPrior;
            Matrix PPrior;
            fEngines[j]->Predict(router.Mode(j), xMix[j], pMix[j], u, t, xPrior, PPrior, Fs[j]);
            fX[j] = xPrior;
            fP[j] = PPrior;
        }
        fC = c;

        Vector weights = Routers::Router::Normalise(c);
        Combine(weights, xk, Pk);
        std::size_t best = 0;
        for (std::size_t j = 1; j < M; ++j)
            if (c(j) > c(best)) best = j;
        F = Fs[best];
    }

    std::size_t Interacting::Update(const Model &model, const Vector &, const Matrix &, const Vector &z, double t,
                                    Vector &xOut, Matrix &POut) const
    {
        const Models::Router &router = Bind(model);
        const std::size_t M = fX.size(), m = router.NMeasure();

        Vector likelihoods(M, 1.);
        std::vector<double> nis(M, 0.);
        std::size_t used = 0;
        for (std::size_t j = 0; j < M; ++j)
        {
            Vector nu;
            Matrix S;
            fEngines[j]->Innovation(router.Mode(j), fX[j], fP[j], z, t, nu, S);

            // Missing channels drop out of both the likelihood and the NIS, so a
            // partly blind step carries less evidence, not none.
            const std::vector<std::size_t> keep = Utils::ValidChannels(nu);
            if (keep.empty())
            {
                likelihoods(j) = 1.;
                nis[j] = static_cast<double>(m);       // neutral: the expected value
            }
            else
            {
                const Vector nuc = Utils::Compress(nu, keep);
                const Matrix Sc = Utils::CompressBoth(S, keep);
                likelihoods(j) = Likelihood(nuc, Sc);
                nis[j] = Utils::Mahalanobis(nuc, Utils::Inverse(Sc));
            }

            Vector xPost;
            Matrix PPost;
            used = fEngines[j]->Update(router.Mode(j), fX[j], fP[j], z, t, xPost, PPost);
            fX[j] = xPost;
            fP[j] = PPost;
        }
        fNis.push_back(nis);

        // --- route: the only decision, and the only pluggable part ---------------------
        fMu = Routers::Router::Normalise(router.Decision().Route(Table(fNis, M), fC, likelihoods));
        std::vector<double> row(M);
        std::size_t best = 0;
        for (std::size_t j = 0; j < M; ++j)
        {
            row[j] = fMu(j);
            if (fMu(j) > fMu(best)) best = j;
        }
        fModeHistory.push_back(row);
        router.SetLeading(best);

        Combine(fMu, xOut, POut);
        return used;
    }

    double Interacting::Likelihood(const Vector &nu, const Matrix &S)
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

    Matrix Interacting::ModeHistory() const { return Table(fModeHistory, fX.size()); }
    Matrix Interacting::NisHistory() const { return Table(fNis, fX.size()); }

    std::size_t Interacting::DominantMode(std::size_t k) const
    {
        if (k >= fModeHistory.size()) throw std::out_of_range("Engines::Interacting::DominantMode: step out of range.");
        std::size_t best = 0;
        for (std::size_t j = 1; j < fModeHistory[k].size(); ++j)
            if (fModeHistory[k][j] > fModeHistory[k][best]) best = j;
        return best;
    }
}
