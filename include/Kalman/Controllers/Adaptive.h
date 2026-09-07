/**
 * *********************************************
 *
 * \file Adaptive.h
 * \brief The two innovation-driven adapters: a gain boost on R, a Q scaling.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Neither commands anything — Policy() is zero. They live in the
 * controller slot because that is the hook a Kalman::Filter offers once per step
 * with the innovation in hand, and adapting the model *is* a control decision
 * about the filter. Both restore the nominal matrix on Reset(). They act through
 * Model::SetR / SetQ, so a model whose R(t) or Q(t) override ignores the stored
 * matrix is unaffected — the same caveat as the Python originals.
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLERS_ADAPTIVE_H
#define KALMAN_CONTROLLERS_ADAPTIVE_H

#include "Kalman/Controller.h"

#include <utility>
#include <vector>

namespace Kalman::Controllers
{
    /**
     * \brief When the filter is surprised (NIS >> m over a window), shrink R so the gain grows.
     * Mirrors `controllers/boost.py`.
     */
    class Boost : public Controller
    {
        protected:
            double fBoostFactor = 3.;
            double fThreshold = 2.;
            std::size_t fWindow = 10;
            Matrix fRNominal;
            std::vector<double> fNis;
            std::vector<std::pair<double, double>> fScaleLog;   // (t, scale)

        public:
            Boost(Model *model, double boostFactor = 3., double threshold = 2., std::size_t window = 10,
                  const Vector &xRef = Vector(), const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "Boost Controller"; }
            const char *Description() const override { return "Innovation-based Kalman gain boost by R scaling."; }

            Vector Policy(const Vector &, const Matrix &, double, const Vector *) override;
            void Update(const Vector &xHat, const Matrix &P, double t, const Vector *nu) override;
            void Reset() override;

            const std::vector<std::pair<double, double>> &ScaleLog() const { return fScaleLog; }
    };

    /**
     * \brief Compare the empirical innovation covariance with the model's S; scale Q by the ratio.
     * Clipped to `[qMin, qMax]`. Mirrors `controllers/adaptive_qr.py`.
     */
    class AdaptiveQR : public Controller
    {
        protected:
            std::size_t fWindow = 20;
            double fQMin = 0.1, fQMax = 10.;
            Matrix fQNominal;
            std::vector<Vector> fInnovations;
            std::vector<std::pair<double, double>> fScaleLog;

        public:
            AdaptiveQR(Model *model, std::size_t window = 20, double qMin = 0.1, double qMax = 10.,
                       const Vector &xRef = Vector(), const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "Adaptive Q/R Controller"; }
            const char *Description() const override { return "Innovation-based Q scaling."; }

            Vector Policy(const Vector &, const Matrix &, double, const Vector *) override;
            void Update(const Vector &xHat, const Matrix &P, double t, const Vector *nu) override;
            void Reset() override;

            const std::vector<std::pair<double, double>> &ScaleLog() const { return fScaleLog; }
    };
}

#endif // KALMAN_CONTROLLERS_ADAPTIVE_H
