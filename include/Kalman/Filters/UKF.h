/**
 * *********************************************
 *
 * \file UKF.h
 * \brief Header of the Kalman::Filters::UKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::UKF
 * \brief The unscented Kalman filter — Kalman::Filter driven by Engines::Unscented.
 *
 * \details A named leaf, like KF and EKF: it names the engine and hands it the
 * spread parameters, and owns no algorithm of its own. Mirrors
 * `modules/framework/filters/ukf.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_UKF_H
#define KALMAN_FILTERS_UKF_H

#include "Kalman/Engines/Unscented.h"
#include "Kalman/Filter.h"

namespace Kalman::Filters
{
    class UKF : public Filter
    {
        public:
            explicit UKF(Model *model, double alpha = 1e-3, double beta = 2., double kappa = 0.)
                : Filter(model, new Engines::Unscented(alpha, beta, kappa), true) {}

            const char *Title() const override { return "Unscented Kalman Filter"; }
            const char *Description() const override
            { return "Forward/backward workflow over a model driven by the sigma-point engine."; }
    };
}

#endif // KALMAN_FILTERS_UKF_H
