/**
 * *********************************************
 *
 * \file Extended.h
 * \brief Header of the Kalman::Models::Extended class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Extended
 * \brief A model that is nonlinear by declaration — override f() and/or h().
 *
 * \details The typed base for systems whose dynamics are a genuine map rather
 * than a matrix. It fixes the default engine to Kalman::EngineKind::Extended and
 * — the part that is easy to forget and expensive to get wrong — turns *off* the
 * analytic-Jacobian assumption, so `Jacobian()` differentiates your `f()` instead
 * of returning `F(t)`, which for a nonlinear model is not its Jacobian at all.
 *
 * \code{.cpp}
 * class MySystem : public Kalman::Models::Extended
 * {
 *     public:
 *         MySystem() : Extended("my-system", 2, 1) { ... }
 *
 *         // The nonlinear map. Whatever your domain's equations are, they go
 *         // here — in your application, never in the framework.
 *         Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &u,
 *                          double t) const override;
 * };
 * \endcode
 *
 * Override Jacobian() / JacobianH() as well if you have the analytic forms; they
 * are both faster and more accurate than central differences.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_EXTENDED_H
#define KALMAN_MODELS_EXTENDED_H

#include "Kalman/Model.h"

namespace Kalman::Models
{
    class Extended : public Model
    {
        public:
            Extended(std::string name, std::size_t n, std::size_t m, std::size_t l = 0)
                : Model(std::move(name), n, m, l)
            {
                SetDefaultEngine(EngineKind::Extended);
                SetAnalyticJacobian(false);
            }

            const char *Title() const override { return "Extended Kalman Model"; }
            const char *Description() const override
            { return "Nonlinear system defined by f()/h(), linearised by Jacobian."; }
    };
}

#endif // KALMAN_MODELS_EXTENDED_H
