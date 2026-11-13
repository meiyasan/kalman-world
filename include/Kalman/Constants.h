/**
 * *********************************************
 *
 * \file Constants.h
 * \brief Enumerations shared across the Kalman framework
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#ifndef KALMAN_TYPES_H
#define KALMAN_TYPES_H

namespace Kalman
{
    /**
     * \brief Which inference algorithm a model wants, and which one an engine provides.
     *
     * Every kind has a C++ engine. Combinatorial is an Extended step plus a
     * track-finding search over an event (Kalman::Engines::Combinatorial::Find);
     * Kalman::Filters::CKF is its leaf, since its input is not a time series.
     *
     * The integer values are the contract with `modules/framework/models/native.py`
     * (`ENGINE_BY_INT`): never renumber.
     */
    enum class EngineKind
    {
        Linear = 0,        ///< constant-matrix predict/update (KF)
        Extended = 1,      ///< Jacobian linearisation (EKF)
        Unscented = 2,     ///< sigma points (UKF)
        Particle = 3,      ///< sequential importance resampling (PF)
        Ensemble = 4,      ///< perturbed-observation ensemble (EnKF)
        Combinatorial = 5  ///< branch-on-ambiguous-hit track finding (CKF)
    };

    /**
     * \brief What a model states about time: its step, or its rate of change.
     *
     * Discrete: f() is the step, the state one sample later, used as written.
     * Continuous: the model states the derivative (Derivative(), or a generator
     * matrix A) and a noise intensity; the framework decodes them into the step
     * and the per-step covariance, so engines always see a discrete model.
     */
    enum class DynamicsKind
    {
        Discrete = 0,
        Continuous = 1
    };

    /**
     * \brief What a model is made of, which decides what a file can hold of it.
     *
     * Maps: f() and h() are code. A file can name the model and carry its
     * settings, not the model itself. This is the default: a model that says
     * nothing is never written out as if its matrices were all of it.
     * Matrices: the model is its constant F, H, B, L, Q, R. A file holds all of it.
     * TimeVarying: the model is its matrices, and they change with time. A
     * file holds all of it over a stated range of samples, as tables.
     */
    enum class StructureKind
    {
        Maps = 0,
        Matrices = 1,
        TimeVarying = 2
    };

    /**
     * \brief Where a model's Jacobians come from (see Kalman::Differentiator).
     */
    enum class DifferentiatorKind
    {
        Analytic = 0,   ///< the model gives the formula
        Forward = 1,    ///< one-sided differences (first order)
        Central = 2,    ///< two-sided differences (second order)
        Automatic = 3   ///< automatic differentiation: reserved, libKalmanLearn
    };

    /**
     * \brief Time-stepping scheme of a Kalman::Integrator — advances `x(t) -> x(t + dt)`
     * for a continuous `xdot = f(x, u, t)`.
     *
     * A model declares its DefaultIntegrator(); Kalman::Integrators::Make resolves it.
     * ImplicitEuler is reserved (needs a per-step Newton solve) and Make() throws for it.
     */
    enum class IntegratorKind
    {
        Euler = 0,          ///< explicit forward Euler (first order)
        RK4 = 1,            ///< classic 4-stage Runge-Kutta (fourth order)
        DormandPrince = 2,  ///< RK45 embedded pair, adaptive substeps
        ImplicitEuler = 3   ///< backward Euler (stiff) — reserved
    };

    /**
     * \brief Spatial-discretization strategy of a Kalman::Solver.
     *
     * A solver discretizes a field and marches it in time by *composing* an
     * Integrator; for an ODE the two coincide. FiniteVolume and FiniteElement are
     * reserved and Kalman::Solvers::Make throws for them.
     */
    enum class SolverKind
    {
        Spectral = 0,          ///< pseudo-spectral (FFT) discretization
        FiniteDifference = 1,  ///< central stencils on a rectangular grid
        FiniteVolume = 2,      ///< reserved
        FiniteElement = 3      ///< reserved
    };

    /**
     * \brief Which pass of the filter a stored estimate came from.
     */
    enum class StateFlavor
    {
        Prior = 0,      ///< predicted, before the measurement was folded in
        Posterior = 1,  ///< corrected, after the forward update
        Smoothed = 2    ///< corrected using the whole record (RTS backward pass)
    };

    /// Human-readable name of an engine kind (never null).
    const char *ToString(EngineKind kind);

    /// Human-readable name of a state flavour (never null).
    const char *ToString(StateFlavor flavor);
    /// Human-readable name of an integrator kind (never null).
    const char *ToString(IntegratorKind kind);
    const char *ToString(DynamicsKind kind);
    const char *ToString(StructureKind kind);
    const char *ToString(DifferentiatorKind kind);
    /// Human-readable name of a solver kind (never null).
    const char *ToString(SolverKind kind);
}

#endif // KALMAN_TYPES_H
