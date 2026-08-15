/**
 * *********************************************
 *
 * \file Base.h
 * \brief Header of the Kalman::Routers::Router contract
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::Router
 * \brief Decides *which model is right now* — the IMM decision layer.
 *
 * \details When one model cannot describe a system across all of its regimes —
 * a manoeuvring target, a storm that recurves, a suspension that goes nonlinear
 * — the answer is several models running in parallel with a probability on each.
 * A router owns that probability update, and only that:
 *
 * \code{.unparsed}
 *     mu = router.Route(nis, c, likelihoods)
 * \endcode
 *
 * \param nis          `(k+1, M)` NIS history up to the current step. Under the
 *                     correct model `NIS_j ~ chi2(m)`, so `E[NIS_j] = m`; under a
 *                     wrong one it is much larger. NIS is the feature of choice
 *                     because it is dimensionless and scale-free, so it does not
 *                     shift between training and inference.
 * \param c            `(M)` Markov-predicted mode probabilities, `c_j = sum_i Pi(i,j) mu_i`.
 * \param likelihoods  `(M)` Gaussian innovation likelihood `L_j = N(nu_j; 0, S_j)`.
 * \return             `(M)` updated mode probabilities: non-negative, summing to 1.
 *
 * One IMM step, with the router as the only pluggable part:
 *
 * \code{.unparsed}
 *   1. Markov prediction   c_j = sum_i Pi(i,j) mu_i
 *   2. State mixing        x0_j, P0_j from every mode
 *   3. Per model           predict + update  ->  nu_j, S_j, NIS_j, L_j
 *   4. Router              mu = Route(nis, c, likelihoods)      <- pluggable
 *   5. Combination         x = sum_j mu_j x_j
 * \endcode
 *
 * Kalman::Filters::IMM passes those three quantities and never looks inside.
 * Implementations here mirror `modules/framework/routers/` one for one:
 * Kalman::Routers::Bayesian (classical), Kalman::Routers::Threshold (rule-based),
 * Kalman::Routers::Hybrid (a blend) and Kalman::Routers::Neural (a policy
 * trained in Python and executed here through ONNX Runtime).
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_BASE_H
#define KALMAN_ROUTERS_BASE_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Matrix.h"

namespace Kalman::Routers
{
    class Router : public Abstract::Metadata
    {
        public:
            ~Router() override = default;

            /// Updated mode probabilities — see the class documentation.
            virtual Vector Route(const Matrix &nis, const Vector &c,
                                 const Vector &likelihoods) const = 0;

            /// Any router is callable.
            Vector operator()(const Matrix &nis, const Vector &c,
                              const Vector &likelihoods) const
            {
                return Route(nis, c, likelihoods);
            }

            /**
             * \brief Clip to a positive floor and normalise to a probability vector.
             *
             * The floor matters: a mode that reaches exactly zero can never recover,
             * because every subsequent update multiplies it by something finite. An
             * IMM that has written off a regime cannot switch back to it.
             */
            static Vector Normalise(Vector mu);

        protected:
            /// Last \p window rows of \p nis, front-padded with 1.0 (the chi2(1) mean).
            ///
            /// Padding at the *front* keeps the most recent step in the last slot, so
            /// a feature vector means the same thing at step 2 as at step 2000.
            static Matrix NisWindow(const Matrix &nis, std::size_t window);
    };
}

#endif // KALMAN_ROUTERS_BASE_H
