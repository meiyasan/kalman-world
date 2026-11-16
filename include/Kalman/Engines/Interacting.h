/**
 * *********************************************
 *
 * \file Interacting.h
 * \brief Header of the Kalman::Engines::Interacting class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Engines::Interacting
 * \brief The predict/update step for a model with several modes (Models::Router):
 * the Interacting Multiple Model filter of the literature.
 *
 * \details The engine keeps one Gaussian belief per mode and the probability
 * `mu` of each. One step, in the words of the contract:
 *
 * \code{.unparsed}
 *   predict
 *     c_j        = sum_i Pi_ij mu_i                    the probabilities, one step on (Markov)
 *     x_j, P_j  <- blend of every mode's belief        "mixing": weights Pi_ij mu_i / c_j
 *     x_j, P_j  <- predict of mode j, with its own engine
 *     prior      = sum_j c_j (x_j, P_j)                what the filter records
 *   update
 *     L_j, NIS_j = how well mode j explains z          the likelihood of its innovation
 *     x_j, P_j  <- update of mode j, with its own engine
 *     mu         = decision(NIS so far, c, L)          the router (Routers::Bayesian: mu_j ~ c_j L_j)
 *     posterior  = sum_j mu_j (x_j, P_j)               the combined estimate
 * \endcode
 *
 * Mixing is what separates this from several filters run side by side: a mode
 * that becomes relevant starts from the state the others built up. A combined
 * covariance adds the spread of the modes' means to their covariances.
 *
 * Each mode is stepped by the engine its model asks for (linear, extended,
 * unscented, combinatorial), so a linear and a nonlinear mode can sit in one
 * model. A mode on a sample-set engine (particle, ensemble) is refused: mixing
 * hands each mode a Gaussian to start from, which those engines would ignore.
 *
 * The belief the filter passes in is not what the engine steps from: the
 * modes' own beliefs are, which start from each mode's initial state at
 * Reset(). There is no smoother.
 *
 * Words of this field, kept here: *mode* (one of the models), *mixing* (the
 * blend before the prediction), *mode probability* (`mu`).
 *
 * *********************************************
 */

#ifndef KALMAN_ENGINES_INTERACTING_H
#define KALMAN_ENGINES_INTERACTING_H

#include "Kalman/Engine.h"
#include "Kalman/Models/Router.h"

#include <memory>
#include <vector>

namespace Kalman::Engines
{
    class Interacting : public Engine
    {
        private:
            // The state of a pass: set at Reset(), advanced by Predict() and Update().
            mutable const Models::Router *fModel = nullptr;
            mutable std::vector<std::unique_ptr<Engine>> fEngines;   // one per mode
            mutable std::vector<Vector> fX;          // (M) per-mode mean
            mutable std::vector<Matrix> fP;          // (M) per-mode covariance
            mutable Vector fMu;                      // (M) mode probabilities
            mutable Vector fC;                       // (M) the same, one step on, before the measurement
            mutable std::vector<std::vector<double>> fNis;           // (T)(M)
            mutable std::vector<std::vector<double>> fModeHistory;   // (T)(M)

            const Models::Router &Bind(const Model &model) const;
            void Combine(const Vector &weights, Vector &x, Matrix &P) const;

        public:
            Interacting() = default;
            ~Interacting() override = default;

            EngineKind Kind() const override { return EngineKind::Interacting; }
            const char *Title() const override { return "Interacting Engine"; }
            const char *Description() const override
            { return "Steps several modes of one system, mixing their beliefs and routing between them."; }

            /// Every mode back to its own initial state, the probabilities to the model's.
            /// \throws std::invalid_argument unless \p model is a Models::Router.
            void Reset(const Model &model) override;
            bool HasSmoother() const override { return false; }

            /// The leading mode's own propagation and measurement.
            void Propagate(const Model &model, const Vector &x, const Vector *u,
                           double t, Vector &xk, Matrix &F) const override;
            void Measure(const Model &model, const Vector &x, double t,
                         Vector &zpred, Matrix &H) const override;

            void Predict(const Model &model, const Vector &x, const Matrix &P,
                         const Vector *u, double t,
                         Vector &xk, Matrix &Pk, Matrix &F) const override;
            std::size_t Update(const Model &model, const Vector &x, const Matrix &P,
                               const Vector &z, double t,
                               Vector &xOut, Matrix &POut) const override;

            std::size_t NModes() const { return fX.size(); }
            /// The current mode probabilities.
            const Vector &ModeProbabilities() const { return fMu; }
            /// The belief of mode \p j.
            const Vector &ModeState(std::size_t j) const { return fX.at(j); }
            const Matrix &ModeCovariance(std::size_t j) const { return fP.at(j); }
            /// The number of updates since Reset().
            std::size_t NSteps() const { return fModeHistory.size(); }
            /// `(T, M)` mode probability per step: which regime, when.
            Matrix ModeHistory() const;
            /// `(T, M)` NIS per mode per step, what the decision layers read.
            Matrix NisHistory() const;
            /// The most probable mode at step \p k. \throws std::out_of_range.
            std::size_t DominantMode(std::size_t k) const;

            /// The Gaussian likelihood `N(nu; 0, S)` of an innovation, floored at 1e-300.
            static double Likelihood(const Vector &nu, const Matrix &S);
    };
}

#endif // KALMAN_ENGINES_INTERACTING_H
