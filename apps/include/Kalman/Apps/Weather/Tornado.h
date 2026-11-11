/**
 * *********************************************
 *
 * \file Tornado.h
 * \brief Header of the Kalman::Apps::Weather::Tornado class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Weather::Tornado
 * \brief Mean-reverting tornado-environment latent — the twin of
 * `kalman.apps.weather.TornadoEnvironmentModel`.
 *
 * \details One latent, a significant-tornado-parameter proxy, relaxing toward
 * the equilibrium of its regime:
 *
 * \code{.unparsed}
 *   x_{k+1} = mu + a (x_k - mu)
 * \endcode
 *
 * A benign and a tornadic instance in a Kalman::Filters::IKF give the
 * probability that the environment is in the tornadic regime.
 *
 * | Parameter     | Meaning                           |
 * |---------------|-----------------------------------|
 * | `mu`          | regime equilibrium                |
 * | `persistence` | AR(1) coefficient `a`, in (0, 1)  |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_WEATHER_TORNADO_H
#define KALMAN_APPS_WEATHER_TORNADO_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Weather
{
    class Tornado : public Kalman::Apps::Model
    {
        private:
            double fMu = 0.3;
            double fPersistence = 0.85;

        public:
            Tornado();

            const char *Title() const override { return "Tornado Environment"; }
            const char *Description() const override
            { return "Mean-reverting significant-tornado-parameter latent (one IKF regime)."; }

            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_WEATHER_TORNADO_H
