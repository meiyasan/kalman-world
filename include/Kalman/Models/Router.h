/**
 * *********************************************
 *
 * \file Router.h
 * \brief Header of the Kalman::Models::Router class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Router
 * \brief Several models of one system, with a probability on each.
 *
 * \details A system that changes regime (an aircraft that cruises, then
 * turns; a storm that runs straight, then recurves) is described by one model
 * per regime, the *modes*. They share the state and the measurement: every
 * mode means the same thing by `x` and by `z`. What changes between them is
 * the motion or the noise. A Markov matrix `Pi` says how likely the system is
 * to go from mode `i` to mode `j` in one step, and the model carries a
 * probability `mu` on each mode.
 *
 * The model holds the modes, `Pi` and the initial probabilities. The
 * estimation is done by the interacting engine (Kalman::Engines::Interacting,
 * this model's default engine; the Interacting Multiple Model filter of the
 * literature), run by a plain Kalman::Filter:
 *
 * \code{.cpp}
 * auto modes = Kalman::Route({cruise, turn}, Pi);
 * Kalman::Filter filter(modes.get());
 * Kalman::Record record = filter.Forward(zk);          // the combined estimate
 * \endcode
 *
 * Seen as one model, it answers `f`, `h` and its coefficients with those of
 * its *leading* mode: the most probable one, which the engine keeps up to
 * date (the first of the highest initial probabilities before any data). Its
 * initial state is the modes', combined with the initial probabilities.
 *
 * Who decides the probabilities at each step is a Kalman::Routers::Router,
 * the decision layer (Bayesian by default; a rule, a network, a blend).
 *
 * A parameter of a mode is `<mode>.<name>`, the mode being called by its
 * model's name (numbered from the second that shares one) or by the names
 * given. The modes are shared, not copied.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_ROUTER_H
#define KALMAN_MODELS_ROUTER_H

#include "Kalman/Model.h"
#include "Kalman/Routers/Base.h"

#include <memory>
#include <string>
#include <vector>

namespace Kalman::Models
{
    class Router : public Model
    {
        private:
            std::vector<std::shared_ptr<Model>> fModes;
            std::vector<std::string> fModeNames;
            Matrix fPi;                                  // (M,M) Markov transition
            Vector fMu0;                                 // (M) initial mode probabilities
            std::shared_ptr<const Routers::Router> fDecision;   // who decides; never null
            mutable std::size_t fLeading = 0;            // the mode this model answers with

            void Sync() const;
            std::pair<std::size_t, std::string> Resolve(const std::string &name) const;

        public:
            /**
             * \param modes  a model per regime (shared); all with the same n, m and l.
             * \param Pi     (M,M) Markov transition, rows summing to 1.
             * \param mu0    the initial mode probabilities (normalised); uniform when empty.
             * \param names  what each mode is called here; its model's name by default.
             * \throws std::invalid_argument for fewer than two modes, a null mode, modes
             *         of different dimensions or sample rates, a Pi of the wrong shape,
             *         with a negative entry or a row that does not sum to 1, a mu0 of the
             *         wrong size, or two modes of one given name.
             */
            Router(std::vector<std::shared_ptr<Model>> modes, const Matrix &Pi, const Vector &mu0 = Vector(),
                   std::vector<std::string> names = {});

            const char *Title() const override { return "Router Model"; }
            const char *Description() const override
            { return "Several models of one system, with a probability on each."; }

            std::size_t NModes() const { return fModes.size(); }
            const Model &Mode(std::size_t j) const { return *fModes.at(j); }
            Model &Mode(std::size_t j) { return *fModes.at(j); }
            const std::string &ModeName(std::size_t j) const { return fModeNames.at(j); }
            const Matrix &Transition() const { return fPi; }
            const Vector &InitialProbabilities() const { return fMu0; }
            /// Set the initial mode probabilities (normalised on the way in).
            void SetInitialProbabilities(const Vector &mu0);

            /// The decision layer (Routers::Bayesian unless set).
            const Routers::Router &Decision() const { return *fDecision; }
            /// Swap the decision layer (shared). \throws std::invalid_argument when null.
            void SetDecision(std::shared_ptr<const Routers::Router> decision);
            /// The same for a decision layer the caller keeps alive.
            void SetDecision(const Routers::Router *decision);

            /// The mode this model answers with; set by the engine at each step.
            std::size_t Leading() const { return fLeading; }
            void SetLeading(std::size_t j) const;

            // --- seen as one model: the leading mode's ---------------------------
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override { Sync(); return Lead().f(x, u, t); }
            Vector h(const Vector &x, double t = 0.) const override { Sync(); return Lead().h(x, t); }
            Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const override { Sync(); return Lead().Jacobian(x, u, t); }
            Matrix JacobianH(const Vector &x, double t = 0.) const override { Sync(); return Lead().JacobianH(x, t); }
            using Model::f;
            bool HasProcessNoiseInput() const override { return Lead().HasProcessNoiseInput(); }
            Vector f(const Vector &x, const Vector &u, const Vector &w, double t) const override { Sync(); return Lead().f(x, u, w, t); }
            bool HasMeasurementNoiseInput() const override { return Lead().HasMeasurementNoiseInput(); }
            Vector MeasureWithNoise(const Vector &x, const Vector &v, double t) const override { Sync(); return Lead().MeasureWithNoise(x, v, t); }
            Vector LimitControl(const Vector &u) const override { return Lead().LimitControl(u); }
            Matrix F(double t = 0.) const override { Sync(); return Lead().F(t); }
            Matrix H(double t = 0.) const override { Sync(); return Lead().H(t); }
            Matrix B(double t = 0.) const override { return Lead().B(t); }
            Matrix L(double t = 0.) const override { return Lead().L(t); }
            Matrix Q(double t = 0.) const override { Sync(); return Lead().Q(t); }
            Matrix R(double t = 0.) const override { Sync(); return Lead().R(t); }
            void CheckDelays() const override;

            // --- the parameters are the modes', as "<mode>.<name>" -------------------
            std::vector<std::string> ParameterNames() const override;
            bool HasParameter(const std::string &name) const override;
            double Parameter(const std::string &name) const override;
            void SetParameter(const std::string &name, double value) override;

            /// The modes' initial states, combined with the initial probabilities.
            void AdoptInitialState() override;
            void CheckShapes() const override;
            std::string Describe() const override;

        private:
            const Model &Lead() const { return *fModes[fLeading]; }
    };
}

#endif // KALMAN_MODELS_ROUTER_H
