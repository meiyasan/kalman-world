/**
 * *********************************************
 *
 * \file Controller.h
 * \brief Header of the Kalman::Controller class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Controller
 * \brief Closed-loop control over a Kalman filter — `u = ctrl(x_hat, P, t, nu)`.
 *
 * \details A controller is what a Kalman::Filter calls once per step, in place of
 * a recorded control sequence, to decide the command from the current belief:
 * the posterior mean and covariance, the time, and the last innovation. It may
 * also **adapt the model** as a side effect — a gain boost scales `R`, an
 * adaptive scheme scales `Q` — through Update(), which runs after Policy().
 *
 * Every call logs the quadratic cost `e^T Qcost e + u^T Rcost u` with `e = x_hat -
 * x_ref`, charged against the command the actuator can actually deliver
 * (Kalman::Model::ClipControl), not the controller's raw ask: a fixed-gain
 * controller with no notion of an actuator limit can ask for far more than the
 * plant ever receives, and costing that fictitious effort would inflate it
 * against a controller whose output already respects the limit.
 *
 * Mirrors `modules/framework/control.py`; the concrete policies live in
 * `Kalman/Controllers/`.
 *
 * *********************************************
 */

#ifndef KALMAN_CONTROLLER_H
#define KALMAN_CONTROLLER_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Matrix.h"
#include "Kalman/Model.h"

#include <utility>
#include <vector>

namespace Kalman
{
    class Controller : public Abstract::Metadata
    {
        protected:
            Model *fModel = nullptr;                 // the plant (not owned); may be adapted
            Vector fXRef;                            // (n) the set point
            Matrix fQCost;                           // (n,n) state cost
            Matrix fRCost;                           // (l,l) control cost
            std::vector<std::pair<double, double>> fCostLog;   // (t, cost)
            std::vector<std::pair<double, Vector>> fControlLog; // (t, raw command)

        public:
            /**
             * \param model the plant; must outlive the controller.
             * \param xRef  the set point, or empty for the origin.
             * \param Qcost state cost, or empty for the identity.
             * \param Rcost control cost, or empty for the identity.
             */
            explicit Controller(Model *model, const Vector &xRef = Vector(),
                                const Matrix &Qcost = Matrix(), const Matrix &Rcost = Matrix());
            virtual ~Controller() = default;

            const char *Title() const override { return "Kalman Controller"; }
            const char *Description() const override
            { return "Closed-loop control policy over a filter's belief."; }

            Model *GetModel() const { return fModel; }
            const Vector &XRef() const { return fXRef; }
            void SetXRef(const Vector &xRef);
            const Matrix &QCost() const { return fQCost; }
            const Matrix &RCost() const { return fRCost; }
            /// The command dimension, at least 1 so an uncontrolled plant still gets a vector.
            std::size_t NControl() const;

            /**
             * \brief The command for this step; logs the cost and the command.
             * \param xHat the posterior mean.
             * \param P    the posterior covariance.
             * \param t    the time.
             * \param nu   the last innovation, or nullptr on the first step.
             */
            Vector Command(const Vector &xHat, const Matrix &P, double t, const Vector *nu = nullptr);
            Vector operator()(const Vector &xHat, const Matrix &P, double t, const Vector *nu = nullptr)
            { return Command(xHat, P, t, nu); }

            /// The policy itself. Required.
            virtual Vector Policy(const Vector &xHat, const Matrix &P, double t, const Vector *nu) = 0;
            /// Adapt the model after the policy (scale R, scale Q, ...). Optional.
            virtual void Update(const Vector &, const Matrix &, double, const Vector *) {}

            /// Clear the logs and any internal state; restore what Update() changed.
            virtual void Reset();

            const std::vector<std::pair<double, double>> &CostLog() const { return fCostLog; }
            const std::vector<std::pair<double, Vector>> &ControlLog() const { return fControlLog; }
            /// Sum of the logged costs.
            double TotalCost() const;
    };
}

#endif // KALMAN_CONTROLLER_H
