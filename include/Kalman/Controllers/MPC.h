/**
 * *********************************************
 *
 * \file MPC.h
 * \brief Model predictive control: a plan over a horizon, under limits, remade at every step.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLERS_MPC_H
#define KALMAN_CONTROLLERS_MPC_H

#include "Kalman/Controller.h"

#include <cstddef>
#include <vector>

namespace Kalman::Controllers
{
    /**
     * \brief Plans `horizon` commands ahead at every step, honouring limits on the
     * command while it plans, applies the first one, and plans again.
     *
     * \details The plan minimises, over the commands `u_0 ... u_{N-1}`,
     *
     * \code{.unparsed}
     *     sum_{k=1}^{N-1} e_k^T Qcost e_k  +  e_N^T Pf e_N  +  sum_{k=0}^{N-1} u_k^T Rcost u_k
     *
     *     e_k = x_k - x_ref,     x_{k+1} = f(x_k, u_k, t_k),     x_0 = the estimate
     * \endcode
     *
     * subject to `uMin <= u_k <= uMax` and, when a rate limit is given,
     * `|u_k - u_{k-1}| <= duMax` (with `u_{-1}` the command applied last).
     *
     * Where it sits beside the other controllers: Controllers::LQR is one gain
     * from one linearisation, with no notion of a limit; Controllers::MPPI
     * replans by rolling random command sequences through `f`. This one replans
     * deterministically, and the limits are constraints of the plan, not a clip
     * applied after it.
     *
     * - **The prediction model is the model's own `f`.** A model of matrices is
     *   predicted exactly (`F`, `B`). A nonlinear one is linearised along the
     *   previous plan: its state Jacobian from Model::Jacobian(), its command
     *   Jacobian differenced towards the inside of the limits (an `f` that clips
     *   its command has no slope outside them). `nIterations` such passes are
     *   made per step, each around the plan the last one found; one is enough
     *   when the plan is warm, which it is: the plan of the previous step,
     *   shifted by one, starts the next.
     * - **The terminal cost `Pf`** is by default the solution of the discrete
     *   Riccati equation at the set point (Algebra::SolveDARE), so that with no
     *   limit in the way the first command is the LQR command, whatever the
     *   horizon. Where that equation has no solution it falls back to `Qcost`.
     * - **The quadratic programme** is solved in its dual by Hildreth's method
     *   (coordinate ascent on the multipliers of the limits): no dependency, no
     *   general factorisation (the one inverse is of the plan's Hessian, which
     *   is symmetric positive definite), and the same answer for the same input.
     *   With no limit active it is one linear solve.
     *
     * The plan it made and the states it predicts for it are kept: Plan(),
     * Predicted().
     */
    class MPC : public Controller
    {
        protected:
            std::size_t fHorizon = 20;
            Vector fUMin, fUMax;            // (l); +-infinity for no limit
            Vector fDUMax;                  // (l); infinity for no rate limit
            std::size_t fNIterations = 1;
            Matrix fTerminal;               // (n,n) terminal cost; empty until first needed
            bool fTerminalGiven = false;
            std::size_t fMaxSweeps = 5000;  // Hildreth sweeps
            double fTolerance = 1e-10;      // on the multipliers, relative

            std::vector<Vector> fPlan;      // (horizon) commands (l)
            std::vector<Vector> fPredicted; // (horizon + 1) states the plan leads through
            Vector fLast;                   // the command applied last; empty before the first
            std::size_t fSweeps = 0;        // sweeps of the last solve
            std::size_t fActive = 0;        // limits active in the last plan

            void Linearise(const Vector &x, const Vector &u, double t, Matrix &A, Matrix &B) const;
            const Matrix &Terminal(double t);
            void Solve(const Vector &x0, double t);

        public:
            /**
             * \param model        the plant; must outlive the controller.
             * \param horizon      the number of steps planned ahead (> 0).
             * \param uMin, uMax   limits on the command, (l); empty for none.
             * \param duMax        limit on its change per step, (l); empty for none.
             * \param nIterations  linearisation passes per step (>= 1).
             * \throws std::invalid_argument for a model with no control, a zero horizon,
             *         limits of the wrong size, or `uMin > uMax`.
             */
            MPC(Model *model, std::size_t horizon = 20, const Vector &uMin = Vector(), const Vector &uMax = Vector(),
                const Vector &duMax = Vector(), std::size_t nIterations = 1,
                const Vector &xRef = Vector(), const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "MPC Controller"; }
            const char *Description() const override
            { return "Model predictive control: a plan over a horizon under limits, remade at every step."; }

            std::size_t Horizon() const { return fHorizon; }
            const Vector &UMin() const { return fUMin; }
            const Vector &UMax() const { return fUMax; }
            const Vector &DUMax() const { return fDUMax; }

            /// Give the terminal cost, (n, n); the Riccati solution is used otherwise.
            void SetTerminalCost(const Matrix &Pf);
            /// The terminal cost in use (solved on first use when none was given).
            const Matrix &TerminalCost() { return Terminal(fModel->T0()); }
            /// Bound the solver: sweeps over the limits, and the relative change at which it stops.
            void SetSolver(std::size_t maxSweeps, double tolerance);

            /// The commands planned at the last step, first one first.
            const std::vector<Vector> &Plan() const { return fPlan; }
            /// The states the last plan leads through: the estimate, then one per command.
            const std::vector<Vector> &Predicted() const { return fPredicted; }
            /// Sweeps the last solve took (0 when no limit was in the way).
            std::size_t Sweeps() const { return fSweeps; }
            /// Limits active in the last plan.
            std::size_t ActiveLimits() const { return fActive; }

            Vector Policy(const Vector &xHat, const Matrix &, double t, const Vector *) override;
            void Reset() override;
    };
}

#endif // KALMAN_CONTROLLERS_MPC_H
