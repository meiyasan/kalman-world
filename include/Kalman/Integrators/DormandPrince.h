/**
 * *********************************************
 *
 * \file DormandPrince.h
 * \brief Dormand–Prince 5(4): an embedded pair with error-controlled substeps.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_INTEGRATORS_DORMANDPRINCE_H
#define KALMAN_INTEGRATORS_DORMANDPRINCE_H

#include "Kalman/Integrator.h"

namespace Kalman::Integrators
{
    /**
     * \brief Adaptive RK45.
     *
     * Step() advances one *outer* step `dt` by as many substeps as the local
     * error tolerance requires, so a fast-varying `f` is resolved without the
     * caller shrinking `dt`. The fifth-order solution is returned; its fourth-order
     * companion drives the step-size control. GetTableau() hands out the
     * fifth-order weights, which is what a field solver uses as a fixed-step RK5
     * (per-member adaptivity makes no sense for a batched ensemble). Mirrors
     * `integrators/dormand_prince.py`.
     */
    class DormandPrince : public Integrator
    {
        private:
            double fRtol = 1e-6;
            double fAtol = 1e-9;
            std::size_t fMaxSubsteps = 10000;

        public:
            explicit DormandPrince(double rtol = 1e-6, double atol = 1e-9, std::size_t maxSubsteps = 10000)
                : fRtol(rtol), fAtol(atol), fMaxSubsteps(maxSubsteps) {}

            IntegratorKind Kind() const override { return IntegratorKind::DormandPrince; }
            const char *Title() const override { return "Dormand-Prince Integrator"; }
            const char *Description() const override
            { return "Adaptive RK45 (Dormand-Prince 5(4)) with error-controlled substeps."; }

            double Rtol() const { return fRtol; }
            double Atol() const { return fAtol; }
            std::size_t MaxSubsteps() const { return fMaxSubsteps; }

            Tableau GetTableau() const override;

            Vector Step(const Derivative &f, const Vector &x, const Vector &u,
                        double t, double dt) const override;
    };
}

#endif // KALMAN_INTEGRATORS_DORMANDPRINCE_H
