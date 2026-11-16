/**
 * *********************************************
 *
 * \file Observability.h
 * \brief Header of the Kalman::Observability class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Observability
 * \brief Can the measurements tell the components of the state apart?
 *
 * \details A question asked *about a model*, before any data: along the path
 * the model follows from its initial state, how much do `N` measurements say
 * about each direction of the state? The answer is the observability Gramian,
 *
 * \code{.unparsed}
 *     G = sum_k  Phi_k^T H_k^T R_k^-1 H_k Phi_k,      Phi_k = F_{k-1} ... F_0
 * \endcode
 *
 * with `F_k` and `H_k` the Jacobians along the noise-free path. It is the
 * information the measurements carry about the initial state; a direction `d`
 * with `d^T G d = 0` changes no measurement, so no filter can estimate it.
 *
 * The Gramian is scaled by the initial standard deviations, `S = D G D` with
 * `D = diag(sqrt(P0_ii))`, which makes it dimensionless: an eigenvalue of `S`
 * is the information gained along its direction in units of what was known
 * to begin with. A direction is *weak* when its eigenvalue is below
 * `tolerance` times the largest one. Each weak direction is reported with the
 * share of every state component in it, so the answer reads "the gain and the
 * amplitude cannot be told apart", not "rank 2 of 3".
 *
 * Where this matters: a model whose state carries parameters (Models::Joint).
 * A parameter that enters only as a product with another unknown passes every
 * shape check and still sits on its prior for ever.
 *
 * \note Local, like every linearisation: it answers for the path from the
 * given initial state. A system at rest may hide a parameter that motion would
 * reveal; give a representative initial state.
 *
 * *********************************************
 */

#ifndef KALMAN_OBSERVABILITY_H
#define KALMAN_OBSERVABILITY_H

#include "Kalman/Matrix.h"
#include "Kalman/Model.h"

#include <cstddef>
#include <vector>

namespace Kalman
{
    class Observability
    {
        private:
            Matrix fGramian;       // (n,n) scaled Gramian S
            Vector fValues;        // its eigenvalues, ascending
            Matrix fVectors;       // its eigenvectors, one per column, in the same order
            double fTolerance;
            std::size_t fRank = 0;

        public:
            /**
             * \param model      the model; its noise-free path from \p x0 is followed.
             * \param steps      the number of measurements considered (> 0).
             * \param x0         where the path starts; the model's initial state when null.
             * \param tolerance  a direction is weak below this fraction of the largest eigenvalue.
             * \throws std::invalid_argument for steps = 0 or an x0 of the wrong size.
             */
            explicit Observability(const Model &model, std::size_t steps = 50, const Vector *x0 = nullptr,
                                   double tolerance = 1e-9);

            /// The scaled Gramian `D G D`.
            const Matrix &Gramian() const { return fGramian; }
            /// Its eigenvalues, smallest first.
            const Vector &Eigenvalues() const { return fValues; }
            /// Its eigenvectors, one per column, in the order of Eigenvalues(). Unit length,
            /// in units of the initial standard deviation of each component.
            const Matrix &Directions() const { return fVectors; }
            double Tolerance() const { return fTolerance; }

            /// The number of directions the measurements do determine.
            std::size_t Rank() const { return fRank; }
            /// True when every direction is determined.
            bool IsObservable() const { return fRank == fValues.Size(); }
            /// The number of weak directions: the first NWeak() columns of Directions().
            std::size_t NWeak() const { return fValues.Size() - fRank; }

            /**
             * \brief The share of state component \p i in the weak directions, in [0, 1].
             *
             * 0: the measurements determine it. 1: they say nothing about it. In
             * between: it is determined only in combination with the other
             * components that share those directions.
             */
            double Weakness(std::size_t i) const;
            /// The components whose Weakness() exceeds \p threshold.
            std::vector<std::size_t> WeakComponents(double threshold = 1e-3) const;
    };
}

#endif // KALMAN_OBSERVABILITY_H
