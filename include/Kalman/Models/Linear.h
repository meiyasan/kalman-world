/**
 * *********************************************
 *
 * \file Linear.h
 * \brief Header of the Kalman::Models::Linear class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Linear
 * \brief A model that is linear by declaration — matrices only, no f()/h().
 *
 * \details The typed base for systems whose dynamics really are `F x + B u`.
 * Declaring that is not decoration: it fixes the default engine to
 * Kalman::EngineKind::Linear and keeps the analytic Jacobian, so nothing pays for
 * central differences it does not need.
 *
 * Subclass it, set the matrices, and stop:
 *
 * \code{.cpp}
 * class MySystem : public Kalman::Models::Linear
 * {
 *     public:
 *         MySystem() : Linear("mine", 2, 1) { SetF(...); SetH(...); }
 * };
 * \endcode
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_LINEAR_H
#define KALMAN_MODELS_LINEAR_H

#include "Kalman/Model.h"

namespace Kalman::Models
{
    class Linear : public Model
    {
        public:
            Linear(std::string name, std::size_t n, std::size_t m, std::size_t l = 0)
                : Model(std::move(name), n, m, l)
            {
                SetDefaultEngine(EngineKind::Linear);
                SetAnalyticJacobian(true);
                SetStructure(StructureKind::Matrices);
            }

            const char *Title() const override { return "Linear Kalman Model"; }
            const char *Description() const override
            { return "System defined by constant (or time-varying) F/H/B/L/Q/R matrices."; }
    };
}

#endif // KALMAN_MODELS_LINEAR_H
