/**
 * *********************************************
 *
 * \file Joint.h
 * \brief Header of the Kalman::Models::Joint class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Joint
 * \brief A model whose state carries some of another model's parameters, so
 * the filter estimates them along with the motion.
 *
 * \details Joint estimation of state and parameters by state augmentation. The
 * inner model declares its scalars by name (Model::DeclareParameter); the
 * joint model appends the chosen ones to the state,
 *
 * \code{.unparsed}
 *     x = [ s ; theta ]          s: the inner state (n),  theta: p parameters
 *
 *     s_{k+1}     = f_inner(s_k, u_k, t ; theta_k)
 *     theta_{k+1} = theta_k + w_theta,     w_theta ~ N(0, diag(drift^2))
 *     z_k         = h_inner(s_k, t ; theta_k)
 * \endcode
 *
 * so the parameters are constants that may drift a little (a random walk),
 * which keeps the filter able to move them. A parameter given a time `tau`
 * wanders and comes back instead (a first-order Gauss-Markov process, for a
 * sensor bias or a clock offset):
 *
 * \code{.unparsed}
 *     theta_{k+1} = mu + exp(-dt / tau) (theta_k - mu) + w_theta
 * \endcode
 *
 * with `mu` the value the inner model held when this one took its initial
 * state and `dt` the sample period, in the model's unit of time. Its variance
 * then settles at `drift^2 / (1 - exp(-2 dt / tau))` instead of growing
 * without bound. Each map sets the inner model's
 * parameters to the values carried in the state, evaluates, and puts the
 * previous values back: the inner model is left as it was found.
 *
 * The Jacobian is assembled by blocks,
 *
 * \code{.unparsed}
 *     J = [ df/ds   df/dtheta ]       df/ds:      the inner model's own Jacobian, at theta
 *         [   0         I     ]       df/dtheta:  differenced, p extra evaluations of f
 * \endcode
 *
 * so the motion is linearised exactly as the inner model linearises it (its
 * analytic Jacobian if it has one; the scheme it declares otherwise), and only
 * the p parameter columns are differenced, with the inner model's scheme and
 * step. dh/dtheta is obtained the same way.
 *
 * The covariances are block-diagonal: Q = diag(Q_inner, drift^2), R = R_inner.
 * Q_inner is asked for with the parameters of the last state stepped from, so
 * a process noise that depends on them sees the estimate.
 * The initial state is the inner model's, followed by the current values of the
 * parameters; the initial covariance is diag(P0_inner, sigma^2). Both are taken
 * at construction: set the inner model first, or call SetInitialState() and
 * SetInitialCovariance() on this model afterwards.
 *
 * The inner model is shared, not copied. The sample rate set on this model is
 * passed on to it before each evaluation, so the two step alike.
 *
 * \note Nothing here is specific to a domain: the drag coefficient of a shell,
 * the gain of a sensor and the damping of an oscillator are all a declared
 * parameter of some model.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_JOINT_H
#define KALMAN_MODELS_JOINT_H

#include "Kalman/Model.h"

#include <memory>
#include <string>
#include <vector>

namespace Kalman
{
    /// One parameter estimated with the state: its name in the model, and its noise.
    struct EstimatedParameter
    {
        std::string name;
        double sigma = 0.;  ///< standard deviation of the initial guess (> 0)
        double drift = 0.;  ///< standard deviation of the noise added to it per step (>= 0)
        double tau = 0.;    ///< time over which it returns to its starting value; 0: never (a random walk)
    };
}

namespace Kalman::Models
{
    class Joint : public Model
    {
        public:
            /// One estimated parameter (Kalman::EstimatedParameter).
            using Parameter = EstimatedParameter;

        private:
            std::shared_ptr<Model> fInner;
            std::vector<Parameter> fEstimated;
            std::size_t fNInner = 0;   // n of the inner model
            std::vector<double> fMean; // the value each parameter started from (what a tau returns to)

            /// `exp(-dt / tau)` of parameter \p k (1 for a random walk).
            double Decay(std::size_t k) const;
            // The state of the last f() call: an engine asks for Q(t) right after
            // stepping, and a Q that depends on the parameters needs their values.
            mutable Vector fLastX;

            /// Refuse a parameter that changes neither f nor h (see the constructor).
            void CheckActs() const;

            /// The inner model at this model's sample rate (it may have been changed since).
            void Sync() const;
            /// Split x into the inner state and the parameters.
            Vector Head(const Vector &x) const;
            /// Set the parameters to the tail of \p x; returns their previous values.
            std::vector<double> Apply(const Vector &x) const;
            void Restore(const std::vector<double> &previous) const;
            /// The scheme and step the parameter columns are differenced with.
            DifferentiatorKind ParameterScheme() const;
            double ParameterStep() const;

        public:
            /**
             * \param inner       the model whose parameters are estimated (shared).
             * \param parameters  which of its declared parameters, with their noise.
             * \throws std::invalid_argument for a null model, no parameter, a name
             *         given twice, a sigma <= 0, a drift < 0 or a tau < 0, or a parameter that
             *         changes neither f nor h at the inner initial state nor at a
             *         point shifted from it (a model that reads it only when built
             *         would leave the estimate on its prior).
             * \throws std::out_of_range for a name the inner model does not declare.
             */
            Joint(std::shared_ptr<Model> inner, std::vector<Parameter> parameters);

            const char *Title() const override { return "Joint Model"; }
            const char *Description() const override
            { return "A model whose state also carries some of its parameters, estimated with the motion."; }

            const Model &Inner() const { return *fInner; }
            Model &Inner() { return *fInner; }
            /// The estimated parameters, in the order they follow the inner state.
            const std::vector<Parameter> &Estimated() const { return fEstimated; }
            /// The state dimension of the inner model: the parameters start at this index.
            std::size_t NInner() const { return fNInner; }

            // --- maps ----------------------------------------------------------
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;
            Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const override;
            Matrix JacobianH(const Vector &x, double t = 0.) const override;

            using Model::f;
            bool HasProcessNoiseInput() const override { return fInner->HasProcessNoiseInput(); }
            Vector f(const Vector &x, const Vector &u, const Vector &w, double t) const override;
            bool HasMeasurementNoiseInput() const override { return fInner->HasMeasurementNoiseInput(); }
            Vector MeasureWithNoise(const Vector &x, const Vector &v, double t) const override;
            Vector LimitControl(const Vector &u) const override { return fInner->LimitControl(u); }

            // --- coefficients --------------------------------------------------
            Matrix B(double t = 0.) const override;
            Matrix L(double t = 0.) const override;
            Matrix Q(double t = 0.) const override;
            Matrix R(double t = 0.) const override;

            /// The inner initial state, then the parameters as the inner model holds them now.
            void AdoptInitialState() override;
            void CheckShapes() const override;
            std::string Describe() const override;
    };
}

#endif // KALMAN_MODELS_JOINT_H
