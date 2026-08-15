/**
 * *********************************************
 *
 * \file Hybrid.h
 * \brief Header of the Kalman::Routers::Hybrid class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::Hybrid
 * \brief Geometric blend (log-linear opinion pool) of two routers.
 *
 * \code{.unparsed}
 *     mu = mu_a^(1-alpha) * mu_b^alpha / Z
 * \endcode
 *
 * At `alpha = 0` it is purely the first router, at `alpha = 1` purely the
 * second. A geometric blend rather than an arithmetic one because it is
 * *conjunctive*: a mode both routers dislike stays suppressed, where an
 * arithmetic mean would let one confident router resurrect it.
 *
 * The usual reason to reach for it is to keep a learned policy on a leash:
 *
 * \code{.cpp}
 * Kalman::Routers::Bayesian bayes;
 * Kalman::Routers::Neural   learned("router.onnx");
 * Kalman::Routers::Hybrid   router(&bayes, &learned, 0.3);  // mostly Bayes
 * \endcode
 *
 * \note Neither child is owned; both must outlive the blend.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_HYBRID_H
#define KALMAN_ROUTERS_HYBRID_H

#include "Kalman/Routers/Base.h"

namespace Kalman::Routers
{
    class Hybrid : public Router
    {
        private:
            const Router *fA = nullptr;
            const Router *fB = nullptr;
            double fAlpha = 0.5;

        public:
            /**
             * \param a     the router at `alpha = 0`; not owned.
             * \param b     the router at `alpha = 1`; not owned.
             * \param alpha blend weight toward \p b, in [0, 1].
             */
            Hybrid(const Router *a, const Router *b, double alpha = 0.5);

            const char *Title() const override { return "Hybrid Router"; }
            const char *Description() const override
            { return "Geometric (log-linear) blend of two routers."; }

            double Alpha() const { return fAlpha; }
            void SetAlpha(double alpha);

            Vector Route(const Matrix &nis, const Vector &c,
                         const Vector &likelihoods) const override;
    };
}

#endif // KALMAN_ROUTERS_HYBRID_H
