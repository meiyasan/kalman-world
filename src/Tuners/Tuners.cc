/**
 * *********************************************
 *
 * \file Tuners.cc
 * \brief Source of the tuners: Kalman::Tuners::Boost and AdaptiveQR
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Tuners/Adaptive.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Tuners
{
    namespace
    {
        double Trace(const Matrix &M)
        {
            double t = 0.;
            for (std::size_t i = 0; i < std::min(M.Rows(), M.Cols()); ++i) t += M(i, i);
            return t;
        }
    }

    // --- Boost ----------------------------------------------------------------------

    Boost::Boost(Model *model, double boostFactor, double threshold, std::size_t window,
                 const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Tuner(model, xRef, Qcost, Rcost), fBoostFactor(boostFactor), fThreshold(threshold), fWindow(window)
    {
        if (!(boostFactor > 0.) || window == 0) throw std::invalid_argument("Tuners::Boost: positive factor and window.");
    }

    void Boost::Update(const Vector &xHat, const Matrix &P, double t, const Vector *nu)
    {
        if (!nu) return;
        if (fRNominal.Empty()) fRNominal = fModel->R(t);

        const std::vector<std::size_t> keep = Utils::ValidChannels(*nu);
        if (keep.empty()) return;

        const Matrix H = Utils::CompressRows(fModel->JacobianH(xHat, t), keep);
        Matrix S = Utils::Similarity(H, P);
        S += Utils::CompressBoth(fModel->R(t), keep);
        fNis.push_back(Utils::Mahalanobis(Utils::Compress(*nu, keep), Utils::Inverse(S)));

        const std::size_t first = fNis.size() > fWindow ? fNis.size() - fWindow : 0;
        double mean = 0.;
        for (std::size_t i = first; i < fNis.size(); ++i) mean += fNis[i];
        mean /= static_cast<double>(fNis.size() - first);

        const double scale = mean > fThreshold * static_cast<double>(keep.size()) ? 1. / fBoostFactor : 1.;
        Matrix R = fRNominal;
        R *= scale;
        fModel->SetR(R);
        fScaleLog.emplace_back(t, scale);
    }

    void Boost::Reset()
    {
        Controller::Reset();
        if (!fRNominal.Empty()) fModel->SetR(fRNominal);
        fNis.clear();
        fScaleLog.clear();
    }

    // --- AdaptiveQR -----------------------------------------------------------------

    AdaptiveQR::AdaptiveQR(Model *model, std::size_t window, double qMin, double qMax,
                           const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Tuner(model, xRef, Qcost, Rcost), fWindow(window), fQMin(qMin), fQMax(qMax)
    {
        if (window == 0 || !(qMin > 0.) || qMax < qMin) throw std::invalid_argument("Tuners::AdaptiveQR: bad window or bounds.");
    }

    void AdaptiveQR::Update(const Vector &xHat, const Matrix &P, double t, const Vector *nu)
    {
        if (!nu) return;
        if (fQNominal.Empty()) fQNominal = fModel->Q(t);
        if (!Utils::IsComplete(*nu)) return;              // a partial innovation has no full outer product

        fInnovations.push_back(*nu);
        if (fInnovations.size() < fWindow) return;

        const std::size_t m = nu->Size();
        Matrix empirical(m, m, 0.);
        for (std::size_t s = fInnovations.size() - fWindow; s < fInnovations.size(); ++s)
            for (std::size_t i = 0; i < m; ++i)
                for (std::size_t j = 0; j < m; ++j) empirical(i, j) += fInnovations[s](i) * fInnovations[s](j);
        empirical *= 1. / static_cast<double>(fWindow);

        Matrix S = Utils::Similarity(fModel->JacobianH(xHat, t), P);
        S += fModel->R(t);

        const double lambda = std::min(fQMax, std::max(fQMin, Trace(empirical) / std::max(Trace(S), 1e-12)));
        Matrix Q = fQNominal;
        Q *= lambda;
        fModel->SetQ(Q);
        fScaleLog.emplace_back(t, lambda);
    }

    void AdaptiveQR::Reset()
    {
        Controller::Reset();
        if (!fQNominal.Empty()) fModel->SetQ(fQNominal);
        fInnovations.clear();
        fScaleLog.clear();
    }
}
