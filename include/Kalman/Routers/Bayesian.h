/**
 * *********************************************
 *
 * \file Bayesian.h
 * \brief Header of the Kalman::Routers::Bayesian class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Routers::Bayesian
 * \brief The classical IMM update: `mu_j ∝ c_j · L_j`.
 *
 * \details Combines the Markov transition prior with the Gaussian innovation
 * likelihood — Bayes' theorem, nothing more. Optimal when every model is a
 * correctly specified Gaussian, and the right default in almost every case:
 * it needs no training data, no export pipeline and no runtime dependency, and
 * each update can be explained.
 *
 * Reach for Kalman::Routers::Neural only when the regime structure is genuinely
 * not Markov, or the switching signature is one a likelihood ratio cannot see.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_BAYESIAN_H
#define KALMAN_ROUTERS_BAYESIAN_H

#include "Kalman/Routers/Base.h"

namespace Kalman::Routers
{
    class Bayesian : public Router
    {
        public:
            const char *Title() const override { return "Bayesian Router"; }
            const char *Description() const override
            { return "Classical IMM mode update: mu proportional to c * likelihood."; }

            Vector Route(const Matrix &nis, const Vector &c,
                         const Vector &likelihoods) const override;
    };
}

#endif // KALMAN_ROUTERS_BAYESIAN_H
