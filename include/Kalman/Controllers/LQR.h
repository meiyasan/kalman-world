/**
 * *********************************************
 *
 * \file LQR.h
 * \brief The linear-quadratic regulator: one gain from the discrete Riccati equation.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLERS_LQR_H
#define KALMAN_CONTROLLERS_LQR_H

#include "Kalman/Controller.h"

namespace Kalman::Controllers
{
    /**
     * \brief `u = -K (x_hat - x_ref)`, `K` from the DARE of the model's `F`, `B`, `Qcost`, `Rcost`.
     *
     * The Riccati equation is solved by Algebra::SolveDARE (the recursion, then
     * Newton steps; it agrees with scipy's `solve_discrete_are`, the Python
     * reference, to rounding), and Fit() must be called before use. Mirrors
     * `controllers/lqr.py`, which forwards to this class.
     */
    class LQR : public Controller
    {
        protected:
            Matrix fK;              // (l, n), empty until Fit()
            double fTolerance = 1e-10;
            std::size_t fMaxIterations = 100000;

        public:
            explicit LQR(Model *model, const Vector &xRef = Vector(),
                         const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());

            const char *Title() const override { return "LQR Controller"; }
            const char *Description() const override { return "u = -K (x_hat - x_ref), K from the discrete Riccati equation."; }

            /**
             * \brief Solve the DARE for the model's F and B and store the gain.
             * \throws std::runtime_error when the model has no control (l = 0) or the
             *         iteration does not converge (an uncontrollable pair, typically).
             */
            LQR &Fit();
            /**
             * \brief Fit against a given linearisation `x_next = A x + B u` instead of the model's
             * F() and B() at T0 — what the Python class does, with its own `model.F()`, `model.B()`.
             */
            LQR &Fit(const Matrix &A, const Matrix &B);
            /// Use a gain computed elsewhere, (l, n).
            void SetGain(const Matrix &K);
            bool IsFitted() const { return !fK.Empty(); }
            const Matrix &Gain() const { return fK; }

            /// The DARE solution `P = Q + A^T P A - A^T P B (R + B^T P B)^-1 B^T P A` (Algebra::SolveDARE).
            static Matrix SolveDARE(const Matrix &A, const Matrix &B, const Matrix &Q, const Matrix &R,
                                    double tolerance = 1e-10, std::size_t maxIterations = 100000);

            Vector Policy(const Vector &xHat, const Matrix &, double, const Vector *) override;
    };
}

#endif // KALMAN_CONTROLLERS_LQR_H
