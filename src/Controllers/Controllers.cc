/**
 * *********************************************
 *
 * \file Controllers.cc
 * \brief Implementation of the shipped controllers
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Controllers/Adaptive.h"
#include "Kalman/Controllers/LQR.h"
#include "Kalman/Controllers/MPPI.h"
#include "Kalman/Controllers/PID.h"
#include "Kalman/Controllers/Proportional.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Controllers
{
    namespace
    {
        void CheckGain(const Matrix &K, std::size_t l, std::size_t n, const char *what)
        {
            if (K.Rows() != l || K.Cols() != n)
                throw std::runtime_error(std::string("Kalman::Controllers: ") + what + " must be (l, n) = ("
                                         + std::to_string(l) + ", " + std::to_string(n) + ").");
        }

        double Trace(const Matrix &M)
        {
            double t = 0.;
            for (std::size_t i = 0; i < std::min(M.Rows(), M.Cols()); ++i) t += M(i, i);
            return t;
        }
    }

    // --- Proportional ---------------------------------------------------------------

    Proportional::Proportional(Model *model, const Matrix &Kp, const Vector &xRef,
                               const Matrix &Qcost, const Matrix &Rcost)
        : Controller(model, xRef, Qcost, Rcost), fKp(Kp)
    {
        CheckGain(fKp, NControl(), fModel->NState(), "Kp");
    }

    Vector Proportional::Policy(const Vector &xHat, const Matrix &, double, const Vector *)
    {
        Vector e = xHat;
        e -= fXRef;
        Vector u = fKp * e;
        u *= -1.;
        return u;
    }

    // --- PID ------------------------------------------------------------------------

    PID::PID(Model *model, const Matrix &Kp, const Matrix &Ki, const Matrix &Kd,
             const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Controller(model, xRef, Qcost, Rcost), fKp(Kp),
          fKi(Ki.Empty() ? Matrix(Kp.Rows(), Kp.Cols(), 0.) : Ki),
          fKd(Kd.Empty() ? Matrix(Kp.Rows(), Kp.Cols(), 0.) : Kd),
          fIntegral(model->NState(), 0.), fPrevError(model->NState(), 0.), fDt(model->DeltaT())
    {
        CheckGain(fKp, NControl(), fModel->NState(), "Kp");
        CheckGain(fKi, NControl(), fModel->NState(), "Ki");
        CheckGain(fKd, NControl(), fModel->NState(), "Kd");
    }

    Vector PID::Policy(const Vector &xHat, const Matrix &, double, const Vector *)
    {
        Vector e = xHat;
        e -= fXRef;
        fIntegral += fDt * e;
        Vector de = e;
        de -= fPrevError;
        de *= 1. / fDt;
        fPrevError = e;

        Vector u = fKp * e;
        u += fKi * fIntegral;
        u += fKd * de;
        u *= -1.;
        return u;
    }

    void PID::Reset()
    {
        Controller::Reset();
        fIntegral = Vector(fModel->NState(), 0.);
        fPrevError = Vector(fModel->NState(), 0.);
    }

    // --- LQR ------------------------------------------------------------------------

    LQR::LQR(Model *model, const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Controller(model, xRef, Qcost, Rcost)
    {
    }

    Matrix LQR::SolveDARE(const Matrix &A, const Matrix &B, const Matrix &Q, const Matrix &R,
                          double tolerance, std::size_t maxIterations)
    {
        // The value iteration P <- Q + A^T P A - A^T P B (R + B^T P B)^-1 B^T P A converges
        // to the stabilising solution for a stabilisable pair; it is slower than a
        // Schur-based solver but needs nothing beyond a small SPD inverse.
        const Matrix At = A.Transpose(), Bt = B.Transpose();
        Matrix P = Q;
        for (std::size_t it = 0; it < maxIterations; ++it)
        {
            Matrix S = Bt * P * B;
            S += R;
            const Matrix G = Utils::Inverse(S) * (Bt * P * A);   // (R + B^T P B)^-1 B^T P A
            Matrix next = At * P * A;
            next -= (At * P * B) * G;
            next += Q;
            Utils::Symmetrize(next);

            double delta = 0., scale = 1.;
            for (std::size_t i = 0; i < next.Rows() * next.Cols(); ++i)
            {
                delta = std::max(delta, std::abs(next.Data()[i] - P.Data()[i]));
                scale = std::max(scale, std::abs(next.Data()[i]));
            }
            P = next;
            if (delta <= tolerance * scale) return P;
        }
        throw std::runtime_error("Controllers::LQR: the Riccati iteration did not converge "
                                 "(is the pair (F, B) stabilisable?).");
    }

    LQR &LQR::Fit()
    {
        if (fModel->NControl() == 0)
            throw std::runtime_error("Controllers::LQR: the model has no control input (l = 0).");
        const Matrix A = fModel->F(fModel->T0());
        Matrix B = fModel->B(fModel->T0());
        if (B.Empty()) B = Matrix::Identity(fModel->NState());   // the control enters directly

        const Matrix P = SolveDARE(A, B, fQCost, fRCost, fTolerance, fMaxIterations);
        Matrix S = B.Transpose() * P * B;
        S += fRCost;
        fK = Utils::Inverse(S) * (B.Transpose() * P * A);
        return *this;
    }

    Vector LQR::Policy(const Vector &xHat, const Matrix &, double, const Vector *)
    {
        if (fK.Empty()) throw std::runtime_error("Controllers::LQR: call Fit() before use.");
        Vector e = xHat;
        e -= fXRef;
        Vector u = fK * e;
        u *= -1.;
        return u;
    }

    // --- MPPI -----------------------------------------------------------------------

    MPPI::MPPI(Model *model, std::size_t nRollouts, std::size_t horizon, double noiseSigma, double lambda,
               std::size_t nIters, std::uint64_t seed, const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Controller(model, xRef, Qcost, Rcost), fNRollouts(nRollouts), fHorizon(horizon),
          fNoiseSigma(noiseSigma), fLambda(lambda), fNIters(nIters), fSeed(seed), fRandom(seed)
    {
        if (nRollouts == 0 || horizon == 0) throw std::invalid_argument("Controllers::MPPI: rollouts and horizon must be positive.");
        if (!(lambda > 0.)) throw std::invalid_argument("Controllers::MPPI: lambda must be positive.");
        fNominal.assign(fHorizon, Vector(NControl(), 0.));
    }

    Vector MPPI::Policy(const Vector &xHat, const Matrix &, double t, const Vector *)
    {
        const std::size_t l = NControl(), n = fModel->NState();
        const double dt = fModel->DeltaT();
        std::normal_distribution<double> gauss(0., fNoiseSigma);

        std::vector<Vector> nominal = fNominal;
        std::vector<std::vector<Vector>> trial(fNRollouts, std::vector<Vector>(fHorizon, Vector(l, 0.)));
        std::vector<double> costs(fNRollouts);

        for (std::size_t iter = 0; iter < fNIters; ++iter)
        {
            for (std::size_t i = 0; i < fNRollouts; ++i)
            {
                Vector xi = xHat;
                double cost = 0.;
                for (std::size_t k = 0; k < fHorizon; ++k)
                {
                    Vector uk = nominal[k];
                    for (std::size_t j = 0; j < l; ++j) uk(j) += gauss(fRandom);
                    // Clip the candidate before rolling it out or costing it, so the
                    // averaged nominal stays inside what the actuator can deliver.
                    uk = fModel->ClipControl(uk);
                    trial[i][k] = uk;

                    xi = fModel->f(xi, fModel->NControl() > 0 ? uk : Vector(), t + static_cast<double>(k) * dt);
                    Vector e = xi;
                    e -= fXRef;
                    for (std::size_t a = 0; a < n; ++a)
                        for (std::size_t b = 0; b < n; ++b) cost += e(a) * fQCost(a, b) * e(b);
                    for (std::size_t a = 0; a < l; ++a)
                        for (std::size_t b = 0; b < l; ++b) cost += uk(a) * fRCost(a, b) * uk(b);
                }
                costs[i] = cost;
            }

            const double best = *std::min_element(costs.begin(), costs.end());
            std::vector<double> weights(fNRollouts);
            double total = 0.;
            for (std::size_t i = 0; i < fNRollouts; ++i)
            {
                weights[i] = std::exp(-(costs[i] - best) / fLambda);
                total += weights[i];
            }
            total += 1e-12;

            for (std::size_t k = 0; k < fHorizon; ++k)
            {
                Vector acc(l, 0.);
                for (std::size_t i = 0; i < fNRollouts; ++i) acc += (weights[i] / total) * trial[i][k];
                nominal[k] = acc;
            }
        }

        const Vector u0 = nominal[0];
        // Shift the warm-started sequence for the next call's receding horizon.
        for (std::size_t k = 0; k + 1 < fHorizon; ++k) fNominal[k] = nominal[k + 1];
        fNominal[fHorizon - 1] = Vector(l, 0.);
        return u0;
    }

    void MPPI::Reset()
    {
        Controller::Reset();
        fNominal.assign(fHorizon, Vector(NControl(), 0.));
        fRandom.seed(fSeed);
    }

    // --- Boost ----------------------------------------------------------------------

    Boost::Boost(Model *model, double boostFactor, double threshold, std::size_t window,
                 const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Controller(model, xRef, Qcost, Rcost), fBoostFactor(boostFactor), fThreshold(threshold), fWindow(window)
    {
        if (!(boostFactor > 0.) || window == 0) throw std::invalid_argument("Controllers::Boost: positive factor and window.");
    }

    Vector Boost::Policy(const Vector &, const Matrix &, double, const Vector *)
    {
        return Vector(NControl(), 0.);
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
        : Controller(model, xRef, Qcost, Rcost), fWindow(window), fQMin(qMin), fQMax(qMax)
    {
        if (window == 0 || !(qMin > 0.) || qMax < qMin) throw std::invalid_argument("Controllers::AdaptiveQR: bad window or bounds.");
    }

    Vector AdaptiveQR::Policy(const Vector &, const Matrix &, double, const Vector *)
    {
        return Vector(NControl(), 0.);
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
