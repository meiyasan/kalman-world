/**
 * *********************************************
 *
 * \file Threshold.h
 * \brief Header of the Kalman::Routers::Threshold class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::Threshold
 * \brief Routes to whichever model's recent NIS sits closest to `m`.
 *
 * \details Under the correct model `NIS ~ chi2(m)`, so `E[NIS] = m`; a model
 * that does not fit produces a much larger NIS. Averaging over a short window
 * and picking the closest to `m` is therefore a defensible rule that uses no
 * likelihood at all — which makes it a useful cross-check on
 * Kalman::Routers::Bayesian, and a guard rail when blended into
 * Kalman::Routers::Hybrid.
 *
 * Two modes:
 * - **hard** (default) — all the weight on the single best match. Decisive, and
 *   prone to chattering when two models fit about equally well.
 * - **soft** — inverse-distance weighting across all models. Smoother, and it
 *   keeps a mode alive that the hard rule would zero out.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_THRESHOLD_H
#define KALMAN_ROUTERS_THRESHOLD_H

#include "Kalman/Routers/Base.h"

namespace Kalman::Routers
{
    class Threshold : public Router
    {
        private:
            std::size_t fM = 1;        // measurement dimension = E[NIS] when correct
            std::size_t fWindow = 5;   // NIS values averaged before comparing
            bool fSoft = false;        // inverse-distance weighting vs hard argmin

        public:
            /**
             * \param m      measurement dimension; the NIS a correct model should show.
             * \param window how many past NIS values to average (all of them while fewer
             *               have been seen; 0 = the whole history).
             * \param soft   inverse-distance weighting (true) or hard argmin (false).
             */
            explicit Threshold(std::size_t m = 1, std::size_t window = 5, bool soft = false)
                : fM(m), fWindow(window), fSoft(soft) {}

            const char *Title() const override { return "Threshold Router"; }
            const char *Description() const override
            { return "Routes by how close each model's windowed NIS is to its expected value m."; }

            std::size_t MeasurementDim() const { return fM; }
            std::size_t Window() const { return fWindow; }
            bool IsSoft() const { return fSoft; }

            Vector Route(const Matrix &nis, const Vector &c,
                         const Vector &likelihoods) const override;
    };
}

#endif // KALMAN_ROUTERS_THRESHOLD_H
