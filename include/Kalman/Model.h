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
 * For a system given as an **ODE**, override Derivative() with `xdot = f(x, u, t)`
 * and call UseContinuousDynamics() in your constructor: the default f() then
 * advances the derivative one sample with the model's Kalman::Integrator
 * (DefaultIntegrator(), RK4 unless declared otherwise, in Substeps() substeps),
 * which is the ODE counterpart of a PDE model binding a Kalman::Solver. The
 * integrator is *declared* through its kind and resolved on first use, so a
 * model never constructs one — and a custom scheme can still be handed in
 * through SetIntegrator(std::shared_ptr<Integrator>).
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
#include "Kalman/Integrator.h"
#include "Kalman/Differentiator.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace Kalman
{
    class Model : public Abstract::Metadata
    {
        public:
            /// One component of the state or of the measurement: what it is and its unit.
            struct Component
            {
                std::string name;
                std::string unit;   ///< "" when dimensionless or not stated
            };

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
            DifferentiatorKind fNumericalScheme = DifferentiatorKind::Central;   // used when not analytic
            double fEpsilon = 1e-5;         // relative step of the numerical scheme

            // Time stepping, for a model that overrides Derivative(). Declared by
            // kind; the instance is built lazily so a model never imports a scheme.
            bool fContinuous = false;
            IntegratorKind fDefaultIntegrator = IntegratorKind::RK4;
            SolverKind fDefaultSolver = SolverKind::Spectral;
            mutable std::shared_ptr<Integrator> fIntegrator;
            std::size_t fSubsteps = 1;

            // A continuous model's own statement, kept so it can be decoded again
            // when the sample rate changes: the generator A of a linear model,
            // and the noise intensity (per unit time) with its coupling.
            Matrix fGenerator;
            Matrix fNoiseIntensity;
            Matrix fNoiseCoupling;
            void Decode();

            // Named scalar parameters (see DeclareParameter()).
            std::map<std::string, double> fParameters;

            // What each component of the state and of the measurement is
            // (see DeclareState() and DeclareMeasurement()).
            std::vector<Component> fStateComponents;
            std::vector<Component> fMeasurementComponents;

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
            /**
             * \brief One step of the dynamics.
             *
             * Default: `F x + B u`, or — when UseContinuousDynamics() is on — the
             * integral of Derivative() over DeltaT() with the model's integrator.
             */
            virtual Vector f(const Vector &x, const Vector &u, double t = 0.) const;

            /**
             * \brief The continuous dynamics `xdot = f(x, u, t)`, for an ODE model.
             * \throws std::runtime_error unless overridden.
             */
            virtual Vector Derivative(const Vector &x, const Vector &u, double t = 0.) const;
            /// The observation map. Default: `H x`.
            virtual Vector h(const Vector &x, double t = 0.) const;

            // --- noise that enters the maps ------------------------------------
            // By default the process noise is added after f and the measurement
            // noise after h. A model in which the noise goes *through* the map
            // (a noisy thrust, a multiplicative sensor error) says so here and
            // gives the map with its noise argument. The sigma-point engine then
            // draws its points over the state and the noise together.
            /// True when f takes the process noise as an argument: see f(x, u, w, t).
            virtual bool HasProcessNoiseInput() const { return false; }
            /// The step with the process noise \p w (same size as Q) inside it.
            /// Default: `f(x, u, t) + w`.
            virtual Vector f(const Vector &x, const Vector &u, const Vector &w, double t) const;
            /// True when h takes the measurement noise as an argument: see h(x, v, t).
            virtual bool HasMeasurementNoiseInput() const { return false; }
            /// The observation with the measurement noise \p v inside it.
            /// Default: `h(x, t) + v`.
            virtual Vector MeasureWithNoise(const Vector &x, const Vector &v, double t) const;

            /**
             * \brief Actuator-limit hook: the command the plant can actually deliver.
             *
             * Identity by default. A model with real limits (a steering clip, a
             * thrust ceiling) overrides it so a controller's logged cost and an MPPI
             * rollout see the same constraint `f()` enforces internally.
             */
            virtual Vector ClipControl(const Vector &u) const { return u; }

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

            // --- time stepping -----------------------------------------------
            /// Route the default f() through Derivative() and the integrator.
            /// Also switches the Jacobian to numerical: F(t) is not df/dx of an ODE step.
            void UseContinuousDynamics(bool on = true);
            bool HasContinuousDynamics() const { return fContinuous; }

            /// Discrete (f() is the step) or Continuous (decoded from a derivative or a generator).
            DynamicsKind Dynamics() const
            { return (fContinuous || !fGenerator.Empty()) ? DynamicsKind::Continuous : DynamicsKind::Discrete; }

            /**
             * \brief State a linear model in continuous time: `dx/dt = A x`.
             * The transition `F = exp(A dt)` is decoded for the sample rate, and
             * again whenever the sample rate changes.
             */
            void SetGenerator(const Matrix &A);
            const Matrix &Generator() const { return fGenerator; }

            /**
             * \brief State the process noise in continuous time: white noise of
             * intensity \p Qc (per unit time), entering through \p Lc (identity
             * when empty). The per-step covariance Q is decoded for the sample
             * rate by Utils::VanLoan(), with the generator A of a linear model, or
             * for a model with a Derivative() its Jacobian at the initial state.
             */
            void SetNoiseIntensity(const Matrix &Qc, const Matrix &Lc = Matrix());
            const Matrix &NoiseIntensity() const { return fNoiseIntensity; }

            IntegratorKind DefaultIntegrator() const { return fDefaultIntegrator; }
            /// Declare the scheme; the instance is built on first use.
            void SetIntegrator(IntegratorKind kind);
            /// Hand in a scheme of your own (a custom Integrator subclass, or a configured one).
            void SetIntegrator(std::shared_ptr<Integrator> integrator);
            /// The resolved integrator (built from DefaultIntegrator() if none was given).
            Integrator &GetIntegrator() const;

            /// Integrator substeps per sample (each of DeltaT() / Substeps()).
            std::size_t Substeps() const { return fSubsteps; }
            void SetSubsteps(std::size_t n);

            /// The spatial discretization a field model declares (informational here:
            /// a PDE model owns and drives its Kalman::Solver itself).
            SolverKind DefaultSolver() const { return fDefaultSolver; }
            void SetDefaultSolver(SolverKind kind) { fDefaultSolver = kind; }

            bool HasAnalyticJacobian() const { return fAnalyticJacobian; }
            void SetAnalyticJacobian(bool on) { fAnalyticJacobian = on; }

            /// Where Jacobian() and JacobianH() come from: Analytic, or a numerical scheme.
            DifferentiatorKind Differentiator() const
            { return fAnalyticJacobian ? DifferentiatorKind::Analytic : fNumericalScheme; }

            /**
             * \brief Declare how this model's Jacobians are obtained.
             * \param kind  Analytic (F() and H() are the Jacobians), Forward or Central.
             * \param step  relative step of a numerical scheme; <= 0 keeps the scheme's default.
             * \throws std::runtime_error for Automatic, which is reserved for libKalmanLearn.
             */
            void SetDifferentiator(DifferentiatorKind kind, double step = 0.);

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

            // --- named parameters ---------------------------------------------
            // How a caller that did not compile the model configures it: the
            // Python class a C++ model sits behind, a spec read from JSON, a ROOT
            // macro. A model declares its scalars once, in its constructor, and
            // reads them back in ParametersChanged().
            /// The declared names, sorted.
            std::vector<std::string> ParameterNames() const;
            bool HasParameter(const std::string &name) const { return fParameters.count(name) > 0; }
            /// \throws std::out_of_range naming the unknown parameter and the known ones.
            double Parameter(const std::string &name) const;
            /// Set a declared parameter, then call ParametersChanged().
            /// \throws std::out_of_range as Parameter().
            void SetParameter(const std::string &name, double value);

            // --- named components ----------------------------------------------
            // What a reader of a result needs to label it: which entry of the
            // state is the position, in which unit. A model that states nothing
            // gets x0, x1, ... and z0, z1, ... with no unit.
            /// The n components of the state, in the order of the state vector.
            std::vector<Component> StateComponents() const;
            /// The m components of the measurement, in the order of z.
            std::vector<Component> MeasurementComponents() const;

        protected:
            /// Name the components of the state (constructor only).
            /// \throws std::invalid_argument unless exactly n are given.
            void DeclareState(std::vector<Component> components);
            /// Name the components of the measurement (constructor only).
            /// \throws std::invalid_argument unless exactly m are given.
            void DeclareMeasurement(std::vector<Component> components);

            /// Size every coefficient to (n, m, l) and seed sane defaults.
            void Allocate(std::size_t n, std::size_t m, std::size_t l);

            /// Declare a named parameter with its default (constructor only).
            void DeclareParameter(const std::string &name, double value);
            /// Called after every SetParameter(): refresh whatever is derived from them.
            virtual void ParametersChanged() {}
    };
}

#endif // KALMAN_MODEL_H
