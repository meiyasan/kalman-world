/**
 * *********************************************
 *
 * \file Lag.h
 * \brief Header of the Kalman::Models::Lag class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Lag
 * \brief A model that carries its last `L` states along with the present one.
 *
 * \details The state is the inner model's, `L + 1` times over:
 *
 * \code{.unparsed}
 *     X_k = [ x_k ; x_{k-1} ; ... ; x_{k-L} ]
 *
 *     X_{k+1} = [ f_inner(x_k, u_k, t) ; x_k ; ... ; x_{k-L+1} ]
 * \endcode
 *
 * Each step computes the new present state and shifts the others down. Two
 * things come of it, with no change to any filter:
 *
 * - **a sensor that reports late.** A sensor with a `delay` (Sensor::delay)
 *   reads the state as it was `delay` ago: its rows of the measurement are
 *   evaluated at block `round(delay * fs)` instead of block 0. The filter's
 *   update then corrects the past state it concerns and, through the
 *   covariance between the blocks, the present one.
 * - **fixed-lag smoothing.** After the update at step `k`, block `j` is the
 *   estimate of `x_{k-j}` given the data up to `k`: the state `j` steps ago,
 *   improved by the `j` measurements that came after it.
 *
 * The sensors are the inner model's (Model::Sensors()), added before the model
 * is lagged; each keeps its place in `z`. The noise enters the present state
 * only: `Q = diag(Q_inner, 0, ...)`, `R = R_inner`. The initial state is the
 * inner one in every block, with the inner covariance on each block and none
 * between them. A model of matrices stays one, on the linear engine. The
 * named parameters are the inner model's.
 *
 * The inner model is shared, not copied, and steps at this model's sample rate.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_LAG_H
#define KALMAN_MODELS_LAG_H

#include "Kalman/Model.h"

#include <memory>
#include <string>
#include <vector>

namespace Kalman::Models
{
    class Lag : public Model
    {
        private:
            std::shared_ptr<Model> fInner;
            std::size_t fNInner = 0;            // n of the inner model
            std::size_t fDepth = 0;             // L: the number of past states carried
            mutable double fDelaysAt = -1.;     // the sample rate fRowBlock was computed for
            mutable std::vector<std::size_t> fRowBlock;   // for each row of z, the block it reads
            mutable std::vector<std::size_t> fBlocksRead; // the distinct blocks, ascending
            mutable bool fDelayed = false;      // whether any row reads a past block

            /// The inner model at this model's sample rate, and the delays in steps of it.
            void Sync() const;
            Vector Shift(const Vector &x, const Vector &head) const;

        public:
            /**
             * \param inner  the model whose past states are carried (shared), with its sensors.
             * \param depth  `L`, the number of past states (> 0).
             * \throws std::invalid_argument for a null model, `depth = 0`, or a sensor
             *         whose delay is longer than `depth` steps.
             */
            Lag(std::shared_ptr<Model> inner, std::size_t depth);

            const char *Title() const override { return "Lag Model"; }
            const char *Description() const override
            { return "A model that carries its last states: for sensors that report late, and fixed-lag smoothing."; }

            const Model &Inner() const { return *fInner; }
            Model &Inner() { return *fInner; }
            /// `L`: the number of past states carried.
            std::size_t Depth() const { return fDepth; }
            /// The state dimension of the inner model: the size of one block.
            std::size_t NInner() const { return fNInner; }
            /// The state as it was \p steps ago, out of \p x. \throws std::out_of_range beyond Depth().
            Vector Block(const Vector &x, std::size_t steps) const;
            /// The number of steps the sensor reading row \p i of the measurement is late by.
            std::size_t Delay(std::size_t i) const;

            /// The inner model's sensors, each reading its own block of this model's state:
            /// on that state none is late any more (their `delay` is 0; see Delay()).
            std::vector<Sensor> Sensors() const override;
            /// The delays are what this model is for (see Model::CheckDelays()).
            void CheckDelays() const override {}

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
            Matrix F(double t = 0.) const override;
            Matrix H(double t = 0.) const override;
            Matrix B(double t = 0.) const override;
            Matrix L(double t = 0.) const override;
            Matrix Q(double t = 0.) const override;
            Matrix R(double t = 0.) const override { Sync(); return fInner->R(t); }

            // --- the parameters are the inner model's ---------------------------
            std::vector<std::string> ParameterNames() const override { return fInner->ParameterNames(); }
            bool HasParameter(const std::string &name) const override { return fInner->HasParameter(name); }
            double Parameter(const std::string &name) const override { return fInner->Parameter(name); }
            void SetParameter(const std::string &name, double value) override { fInner->SetParameter(name, value); }

            /// The inner initial state in every block; its covariance on each block.
            void AdoptInitialState() override;
            void CheckShapes() const override;
            std::string Describe() const override;
    };
}

#endif // KALMAN_MODELS_LAG_H
