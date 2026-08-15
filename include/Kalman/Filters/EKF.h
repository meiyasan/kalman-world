/**
 * *********************************************
 *
 * \file EKF.h
 * \brief Header of the Kalman::Filters::EKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::EKF
 * \brief The extended Kalman filter — a model driven by Kalman::Engines::Extended.
 *
 * \details Runs on nonlinear models and reduces to the linear computation on
 * matrix models (the Jacobian of `F x` is `F`), so it is the safe choice whenever
 * a model's linearity is not obvious from its declaration.
 *
 * \code{.cpp}
 * MyNonlinearModel model;
 * Kalman::Filters::EKF filter(&model);
 * auto record = filter.Forward(measurements);
 * \endcode
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_EKF_H
#define KALMAN_FILTERS_EKF_H

#include "Kalman/Engines/Extended.h"
#include "Kalman/Filter.h"

namespace Kalman::Filters
{
    class EKF : public Filter
    {
        public:
            explicit EKF(Model *model) : Filter(model, new Engines::Extended(), true) {}

            const char *Title() const override { return "Extended Kalman Filter"; }
            const char *Description() const override
            { return "Forward/backward workflow over a model driven by the Jacobian engine."; }
    };
}

#endif // KALMAN_FILTERS_EKF_H
