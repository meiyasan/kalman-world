/**
 * *********************************************
 *
 * \file LinearModel.h
 * \brief Header of the Kalman::Apps::Ballistics::LinearModel class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Ballistics::LinearModel
 * \brief One axis at constant acceleration: position, velocity, acceleration.
 *
 * The counterpart of `BallisticLinearModel` (`modules/apps/ballistics/linear.py`),
 * with the same matrices. The state is `(p, v, a)` and the position is measured:
 *
 *     dp/dt = v      dv/dt = a + w      da/dt = 0
 *
 * `w` is a white acceleration of spectral density `q^2`, with `q` the process
 * uncertainty [m/s^1.5]: after a time t the velocity has wandered by `q sqrt(t)`.
 * The acceleration takes no disturbance of its own and stays what it was set to
 * (gravity, for the vertical axis). Both are discretized exactly over one
 * sample `dt`:
 *
 *     F = [1 dt dt^2/2; 0 1 dt; 0 0 1]      Q = q^2 [dt^3/3 dt^2/2 0; dt^2/2 dt 0; 0 0 0]
 *
 * `R = sigma^2`, with `sigma` the measurement uncertainty [m], and the initial
 * covariance is `diag(sigma^2, (sigma/10)^2, q^2)`. Three of them side by side
 * (Kalman::Compose) track a body in space when no force links the axes; drag
 * does, and that is Ballistics::ExtendedModel.
 *
 * A model of matrices: it runs on the linear engine. Registered as `ballistic-linear`.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_BALLISTICS_LINEARMODEL_H
#define KALMAN_APPS_BALLISTICS_LINEARMODEL_H

#include <Kalman/Models/Linear.h>

namespace Kalman::Apps::Ballistics
{
    class LinearModel : public Kalman::Models::Linear
    {
        private:
            double fSigma = 0.5;
            double fQ = 0.5;

        public:
            /**
             * \param fs                      the sample rate [Hz].
             * \param measurement_uncertainty the standard deviation of the position reading [m].
             * \param process_uncertainty     the strength `q` of the white acceleration [m/s^1.5].
             * \throws std::invalid_argument for a rate or an uncertainty that is not positive.
             */
            explicit LinearModel(double fs = 128., double measurement_uncertainty = 0.5,
                                 double process_uncertainty = 0.5);

            const char *Title() const override { return "Ballistic State Model"; }
            const char *Description() const override
            { return "Linear constant-acceleration model of one axis; position measured."; }

            double MeasurementUncertainty() const { return fSigma; }
            double ProcessUncertainty() const { return fQ; }

            /// F, H, Q and R for the current sample rate. Call it after SetSampleRate().
            void Build();
    };
}

#endif // KALMAN_APPS_BALLISTICS_LINEARMODEL_H
