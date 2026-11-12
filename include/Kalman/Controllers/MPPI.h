/**
 * *********************************************
 *
 * \file MPPI.h
 * \brief Online receding-horizon model predictive path-integral control.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLERS_MPPI_H
#define KALMAN_CONTROLLERS_MPPI_H

#include "Kalman/Controller.h"

#include <cstdint>
#include <random>

namespace Kalman::Controllers
{
    /**
     * \brief Replans a sampled control sequence at every step by Monte-Carlo rollout of the true `f`.
     *
     * Unlike LQR — one gain from one linearisation — MPPI resolves
     * `argmin E[sum e^T Qcost e + u^T Rcost u]` from scratch each tick by rolling
     * `nRollouts` perturbed sequences through the model's own nonlinear `f()`, so any
     * actuator limit the model enforces (ClipControl) is respected *during* planning.
     * Warm-started: the nominal found at step k, shifted by one, seeds step k+1.
     * The random stream is controller-owned and seeded. Mirrors `controllers/mppi.py`.
     */
    class MPPI : public Controller
    {
        protected:
            std::size_t fNRollouts = 200;
            std::size_t fHorizon = 30;
            double fNoiseSigma = 0.2;
            double fLambda = 1.;
            std::size_t fNIters = 3;
            std::uint64_t fSeed = 0;
            std::mt19937_64 fRandom;
            std::vector<Vector> fNominal;   // (horizon) commands (l)
            std::vector<double> fDraws;     // the caller's perturbations for the next Policy(), or empty

        public:
            MPPI(Model *model, std::size_t nRollouts = 200, std::size_t horizon = 30,
                 double noiseSigma = 0.2, double lambda = 1., std::size_t nIters = 3, std::uint64_t seed = 0,
                 const Vector &xRef = Vector(), const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "MPPI Controller"; }
            const char *Description() const override
            { return "Receding-horizon path-integral control by Monte-Carlo rollout of the model's f."; }

            std::size_t NRollouts() const { return fNRollouts; }
            std::size_t Horizon() const { return fHorizon; }
            const std::vector<Vector> &Nominal() const { return fNominal; }
            /// The number of perturbations one Policy() call draws, `nIters x nRollouts x horizon x l`.
            std::size_t NDraws() const;

            /**
             * \brief Hand the next Policy() call its perturbations instead of drawing them.
             *
             * \p draws holds NDraws() values, already scaled by the noise sigma, in the
             * order iteration, rollout, horizon step, command component — the order
             * `controllers/mppi.py` draws them from its numpy generator, so a seeded
             * Python controller makes the same decisions whoever runs the rollouts
             * (as kalman_propagate takes its noise from the caller). Used once.
             */
            void SetDraws(std::vector<double> draws);

            Vector Policy(const Vector &xHat, const Matrix &, double t, const Vector *) override;
            void Reset() override;
    };
}

#endif // KALMAN_CONTROLLERS_MPPI_H
