/**
 * *********************************************
 *
 * \file Solver.h
 * \brief Header of the Kalman::Solver class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Solver
 * \brief The **spatial discretization** — HOW, in space.
 *
 * \details A solver turns a continuous field problem into something an
 * integrator can march in time. Its own job is the spatial part — the
 * pseudo-spectral right-hand side, the finite-difference stencils; the time
 * march is **delegated to a composed** Kalman::Integrator, one home for the
 * stepping arithmetic. For an ODE the discretization is the identity and a
 * solver *is* an integrator, which is why an ODE model binds an integrator
 * directly and a PDE model binds a solver.
 *
 * Composition, not inheritance: a solver *has* an integrator, and different
 * solvers may pair with different integrators. Mirrors
 * `modules/framework/solver.py`.
 *
 * *********************************************
 */

#ifndef KALMAN_SOLVER_H
#define KALMAN_SOLVER_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Constants.h"
#include "Kalman/Grid.h"
#include "Kalman/Integrator.h"

#include <memory>

namespace Kalman
{
    class Solver : public Abstract::Metadata
    {
        private:
            std::shared_ptr<Integrator> fIntegrator;   // the composed time march

        public:
            /// \param integrator the time march, or nullptr for the RK4 default.
            explicit Solver(std::shared_ptr<Integrator> integrator = nullptr);
            virtual ~Solver() = default;

            virtual SolverKind Kind() const = 0;

            Integrator &GetIntegrator() const { return *fIntegrator; }
            void SetIntegrator(std::shared_ptr<Integrator> integrator);
            void SetIntegrator(IntegratorKind kind);

            /**
             * \brief The composed integrator's explicit tableau.
             * \throws std::runtime_error when it has none: a field is marched with a
             *         fixed-step explicit scheme, so an adaptive or implicit integrator
             *         cannot drive a solver.
             */
            Tableau ExplicitTableau() const;

            /// Advance a real field by \p steps of the solver's own dt from time \p t.
            virtual Field Solve(const Field &field, std::size_t steps = 1, double t = 0.) const = 0;
    };

    namespace Solvers
    {
        /**
         * \brief The shipped solver for \p kind on a square grid of \p n cells a side.
         * \throws std::runtime_error for the reserved FiniteVolume / FiniteElement kinds.
         */
        std::unique_ptr<Solver> Make(SolverKind kind, std::size_t n = 64, double dt = 0.01);
    }
}

#endif // KALMAN_SOLVER_H
