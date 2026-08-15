/**
 * *********************************************
 *
 * \file Model.h
 * \brief Header of the Kalman::Model class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Model
 * \brief The **system** — WHAT is being estimated.
 *
 * \details A model owns the dynamics and the observation map, and nothing else:
 * no run state, no algorithm, no workflow. It answers, for a given time `t`:
 *
 * \code{.unparsed}
 *     x_{k+1} = f(x_k, u_k, t)        the dynamics
 *     z_k     = h(x_k, t)             the observation
 * \endcode
 *
 * together with the matrices the estimator needs — the transition `F`, the
 * observation `H`, the control map `B`, the process-noise coupling `L`, and the
 * covariances `Q` (process) and `R` (measurement).
 *
 * The class is **concrete and linear out of the box**: construct it with
 * dimensions, set F/H/Q/R, and the default f()/h() are `F x + B u` and `H x`.
 *
 * For a **nonlinear** system, subclass and override f() and/or h(). The Jacobians
 * are then obtained by central differences unless you also override Jacobian() /
 * JacobianH() with the analytic ones — tell the base which by calling
 * SetAnalyticJacobian(false) in your constructor.
 *
 * \note This class is the *contract*, not a catalogue. libKalman ships no
 * physics: a pendulum, a drift chamber or a power grid is an **application**, and
 * applications are built on top of this header, not inside it. The equations live
 * where the domain knowledge lives.
 *
 * *********************************************
 */

#ifndef KALMAN_MODEL_H
#define KALMAN_MODEL_H

#include "Kalman/Matrix.h"
#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Constants.h"

#include <string>

namespace Kalman
{
    class Model : public Abstract::Metadata
    {
        protected:
            std::string fName = "model";

            std::size_t fNState = 0;    // n — state dimension
            std::size_t fNMeasure = 0;  // m — measurement dimension
            std::size_t fNControl = 0;  // l — control dimension (0 = uncontrolled)

            double fSampleRate = 1.;    // fs [Hz] — sets the default dt
            double fT0 = 0.;            // time tag of the first step [s]

            Matrix fF;                  // (n,n) state transition
            Matrix fH;                  // (m,n) observation
            Matrix fB;                  // (n,l) control map (empty => u added directly)
            Matrix fL;                  // (n,n) process-noise coupling (empty => identity)
            Matrix fQ;                  // (n,n) process-noise covariance
            Matrix fR;                  // (m,m) measurement-noise covariance

            Vector fX0;                 // (n) initial state mean
            Matrix fP0;                 // (n,n) initial state covariance

            EngineKind fDefaultEngine = EngineKind::Linear;
            bool fAnalyticJacobian = true;  // false => differentiate f()/h() numerically
            double fEpsilon = 1e-5;         // step for the numerical Jacobian

        public:
            Model() = default;
            Model(std::string name, std::size_t n, std::size_t m, std::size_t l = 0);
            virtual ~Model() = default;

            const char *Title() const override { return "Kalman Model"; }
            const char *Description() const override
            { return "System model: dynamics f/F, observation h/H, noise Q/R."; }

            /// The *instance* name your application chose, distinct from Title().
            const std::string &Name() const { return fName; }
            void SetName(std::string name) { fName = std::move(name); }

            // --- dimensions ------------------------------------------------
            std::size_t NState()   const { return fNState; }
            std::size_t NMeasure() const { return fNMeasure; }
            std::size_t NControl() const { return fNControl; }

            double SampleRate() const { return fSampleRate; }
            void SetSampleRate(double fs);
            /// Step length implied by the sample rate (1 when fs is unset).
            double DeltaT() const { return fSampleRate > 0. ? 1. / fSampleRate : 1.; }

            double T0() const { return fT0; }
            void SetT0(double t0) { fT0 = t0; }

            // --- coefficients ----------------------------------------------
            // Override any of these for a time-varying system; the defaults return
            // the stored matrix and ignore t.
            virtual Matrix F(double t = 0.) const;
            virtual Matrix H(double t = 0.) const;
            virtual Matrix B(double t = 0.) const;
            virtual Matrix L(double t = 0.) const;
            virtual Matrix Q(double t = 0.) const;
            virtual Matrix R(double t = 0.) const;

            // --- maps --------------------------------------------------------
            /// One step of the dynamics. Default: `F x + B u`.
            virtual Vector f(const Vector &x, const Vector &u, double t = 0.) const;
            /// The observation map. Default: `H x`.
            virtual Vector h(const Vector &x, double t = 0.) const;

            /// `df/dx` — F(t) when analytic, central differences otherwise.
            virtual Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const;
            /// `dh/dx` — H(t) when analytic, central differences otherwise.
            virtual Matrix JacobianH(const Vector &x, double t = 0.) const;

            // --- setters ------------------------------------------------------
            void SetF(const Matrix &F);
            void SetH(const Matrix &H);
            void SetB(const Matrix &B);
            void SetL(const Matrix &L);
            void SetQ(const Matrix &Q);
            void SetR(const Matrix &R);

            const Vector &X0() const { return fX0; }
            const Matrix &P0() const { return fP0; }
            void SetInitialState(const Vector &x0);
            void SetInitialCovariance(const Matrix &P0);

            EngineKind DefaultEngine() const { return fDefaultEngine; }
            void SetDefaultEngine(EngineKind kind) { fDefaultEngine = kind; }

            bool HasAnalyticJacobian() const { return fAnalyticJacobian; }
            void SetAnalyticJacobian(bool on) { fAnalyticJacobian = on; }

            double Epsilon() const { return fEpsilon; }
            void SetEpsilon(double eps) { fEpsilon = eps; }

            /**
             * \brief Validate every coefficient against (n, m, l).
             * \throws std::runtime_error on the first mismatch, naming the matrix.
             *
             * Called by Kalman::Filter before a run, so a shape bug surfaces once with
             * a readable message rather than deep inside the loop.
             */
            virtual void CheckShapes() const;

            /// One-line description, for logs.
            virtual std::string Describe() const;

        protected:
            /// Size every coefficient to (n, m, l) and seed sane defaults.
            void Allocate(std::size_t n, std::size_t m, std::size_t l);
    };
}

#endif // KALMAN_MODEL_H
