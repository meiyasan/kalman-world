/**
 * *********************************************
 *
 * \file Tuner.h
 * \brief Header of the Kalman::Tuner class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Tuner
 * \brief Retunes the filter's own noise model as it runs; commands nothing.
 *
 * \details A controller acts on the system: it computes the command `u`. A
 * tuner acts on the *filter*: once per step, with the innovation in hand, it
 * adjusts the model's `Q` or `R` (a filter that is surprised trusts its
 * measurements more; a process noise that no longer explains the innovations
 * is rescaled). The two are different roles, and a filter takes one of each:
 *
 * \code{.cpp}
 * Kalman::Controllers::LQR lqr(&model);
 * Kalman::Tuners::AdaptiveQR tune(&model);
 * filter.Forward(measurements, lqr, tune);     // a command from lqr, the noise retuned by tune
 * filter.Forward(measurements, tune);          // no command, the noise retuned
 * \endcode
 *
 * A tuner is called at the same point of the step as a controller and shares
 * its machinery (the belief it is handed, Reset(), the logs), which is why it
 * derives from Kalman::Controller. Its Policy() is fixed: zero. What a
 * concrete tuner writes is Update().
 *
 * *********************************************
 */

#ifndef KALMAN_TUNER_H
#define KALMAN_TUNER_H

#include "Kalman/Controller.h"

namespace Kalman
{
    class Tuner : public Controller
    {
        public:
            using Controller::Controller;

            const char *Title() const override { return "Kalman Tuner"; }
            const char *Description() const override
            { return "Retunes the model's noise from the innovations; commands nothing."; }

            /// A tuner commands nothing: zero, of the command dimension.
            Vector Policy(const Vector &, const Matrix &, double, const Vector *) final { return Vector(NControl(), 0.); }
    };
}

#endif // KALMAN_TUNER_H
