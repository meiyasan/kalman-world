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
     * The C++ core carries the Linear, Extended, Unscented, Particle and Ensemble
     * engines. Combinatorial is declared so that a model's declared default
     * survives a round-trip with the Python framework, where the track-finding
     * pass lives; asking a C++ filter for it throws rather than substituting.
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
        Combinatorial = 5  ///< branch-on-ambiguous-hit track finding (CKF) — Python only
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
}

#endif // KALMAN_TYPES_H
