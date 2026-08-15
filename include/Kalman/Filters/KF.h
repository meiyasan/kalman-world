/**
 * *********************************************
 *
 * \file KF.h
 * \brief Header of the Kalman::Filters::KF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filters::KF
 * \brief The standard Kalman filter — a model driven by Kalman::Engines::Linear.
 *
 * \details The recognisable name for the composition
 * `Filter(model, new Engines::Linear)`. It exists so that code reads the way the
 * textbook does and so the choice of algorithm is visible at the construction
 * site — and it is deliberately trivial. A leaf that grew a `Forward()` of its
 * own would be putting algorithm in the workflow layer.
 *
 * \code{.cpp}
 * MyLinearModel model;
 * Kalman::Filters::KF filter(&model);
 * auto record = filter.Forward(measurements);
 * \endcode
 *
 * *********************************************
 */

#ifndef KALMAN_FILTERS_KF_H
#define KALMAN_FILTERS_KF_H

#include "Kalman/Engines/Linear.h"
#include "Kalman/Filter.h"

namespace Kalman::Filters
{
    class KF : public Filter
    {
        public:
            explicit KF(Model *model) : Filter(model, new Engines::Linear(), true) {}

            const char *Title() const override { return "Linear Kalman Filter"; }
            const char *Description() const override
            { return "Forward/backward workflow over a model driven by the linear engine."; }
    };
}

#endif // KALMAN_FILTERS_KF_H
