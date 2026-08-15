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
     * The C++ core carries the two Jacobian-based engines; the sigma-point,
     * particle and ensemble kinds are declared so that a model's declared default
     * survives a round-trip with the Python framework, but no engine is registered
     * for them here.
     */
    enum class EngineKind
    {
        Linear = 0,     ///< constant-matrix predict/update (KF)
        Extended = 1,   ///< Jacobian linearisation (EKF)
        Unscented = 2,  ///< sigma points (UKF) — not implemented in C++
        Particle = 3,   ///< sequential importance resampling (PF) — not implemented in C++
        Ensemble = 4    ///< perturbed-observation ensemble (EnKF) — not implemented in C++
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
