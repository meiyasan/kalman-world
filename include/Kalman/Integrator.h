/**
 * *********************************************
 *
 * \file Integrator.h
 * \brief Header of the Kalman::Integrator class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Integrator
 * \brief The **time-stepping scheme** — WHEN, within one dt.
 *
 * \details An integrator advances a state one step, `x(t) → x(t + dt)`, given a
 * *continuous* derivative `xdot = f(x, u, t)`. It is the counterpart of
 * Kalman::Engine for the deterministic mean: an ODE model declares which one it
 * wants and overrides Kalman::Model::Derivative(), and Model::f() then steps it;
 * a field solver composes one for its time march.
 *
 * Explicit schemes expose their Butcher tableau, and RungeKuttaStep() runs *any*
 * such tableau on *any* state type with `+=` and scalar `*` — a Kalman::Vector or
 * a Kalman::Grid — so the same integrator drives both an ODE and a spectral field
 * without a scheme being written twice.
 *
 * Integrators own no run state. Mirrors `modules/framework/integrator.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_INTEGRATOR_H
#define KALMAN_INTEGRATOR_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Constants.h"
#include "Kalman/Matrix.h"

#include <functional>
#include <memory>
#include <vector>

namespace Kalman
{
    /// `xdot = f(x, u, t)` — the continuous dynamics an integrator steps.
    using Derivative = std::function<Vector(const Vector &, const Vector &, double)>;

    /// Butcher tableau `(c, A, b)` of an explicit Runge–Kutta method.
    struct Tableau
    {
        std::vector<double> c;
        std::vector<std::vector<double>> a;
        std::vector<double> b;

        bool Empty() const { return b.empty(); }
        std::size_t Stages() const { return b.size(); }
    };

    /**
     * \brief One explicit Runge–Kutta step of \p tableau on any additive state.
     * \param tableau the Butcher tableau of the scheme.
     * \param rhs   `rhs(state, t)` — the derivative of the state at time t.
     * \param x     the state at \p t.
     * \param t     the time of \p x.
     * \param dt    the step.
     */
    template <typename State, typename RHS>
    State RungeKuttaStep(const Tableau &tableau, const RHS &rhs, const State &x, double t, double dt)
    {
        std::vector<State> k;
        k.reserve(tableau.Stages());
        for (std::size_t i = 0; i < tableau.Stages(); ++i)
        {
            State xi = x;
            for (std::size_t j = 0; j < i; ++j)
                if (tableau.a[i][j] != 0.) xi += (dt * tableau.a[i][j]) * k[j];
            k.push_back(rhs(xi, t + tableau.c[i] * dt));
        }
        State out = x;
        for (std::size_t i = 0; i < tableau.Stages(); ++i)
            if (tableau.b[i] != 0.) out += (dt * tableau.b[i]) * k[i];
        return out;
    }

    class Integrator : public Abstract::Metadata
    {
        public:
            Integrator() = default;
            virtual ~Integrator() = default;

            virtual IntegratorKind Kind() const = 0;

            /// Advance one step of \p dt; \p u is held constant across it (zero-order hold).
            virtual Vector Step(const Derivative &f, const Vector &x, const Vector &u,
                                double t, double dt) const = 0;

            /// Advance \p n steps of \p dt.
            virtual Vector Integrate(const Derivative &f, const Vector &x, const Vector &u,
                                     double t, double dt, std::size_t n) const;

            /// The explicit tableau, or an empty one for adaptive / implicit schemes.
            virtual Tableau GetTableau() const { return Tableau(); }
    };

    namespace Integrators
    {
        /**
         * \brief The shipped integrator for \p kind.
         * \throws std::runtime_error for ImplicitEuler, which is reserved (it needs a
         *         per-step Newton solve), rather than silently handing back another scheme.
         */
        std::unique_ptr<Integrator> Make(IntegratorKind kind);
    }
}

#endif // KALMAN_INTEGRATOR_H
