/**
 * *********************************************
 *
 * \file Polarization.h
 * \brief Header of the Kalman::Apps::Gravitational::Polarization class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Gravitational::Polarization
 * \brief Online estimation of a gravitational-wave polarization (Jones
 * formalism, Amblard and Michel) -- the twin of
 * `kalman.apps.gravitational.PolarizationModel`.
 *
 * \details The state is eight numbers, four angles each with its rate:
 *
 * \code{.unparsed}
 *   x = [ log a, d(log a)/dt, psi, dpsi/dt, theta, dtheta/dt, phi, dphi/dt ]
 * \endcode
 *
 * Each pair moves at constant rate over one sample, `[[1, dt], [0, 1]]` with
 * `dt = 1/fs`, so the transition is linear. The measurement is the bivariate
 * complex signal, as four real numbers:
 *
 * \code{.unparsed}
 *   z1 = a exp(i psi) cos(theta/2)
 *   z2 = a exp(i psi) sin(theta/2) exp(i phi)
 *   z  = [ Re z1, Im z1, Re z2, Im z2 ],   a = exp(x0)
 * \endcode
 *
 * The amplitude is tracked as its logarithm, so it cannot change sign (the
 * signal is invariant under `(a, psi) -> (-a, psi + pi)`). `dh/dx` is given
 * analytically, as the Python class gives it; `df/dx` is differenced, as the
 * Python framework differences its `f`.
 *
 * The process noise of each pair is a constant-acceleration block of width
 * `delta` (a bandwidth knob in seconds, not the sample interval):
 * `sigma^2 [[d^3/3, d^2/2], [d^2/2, d]]`; the measurement noise is
 * `sigma_n^2 I`. Both are computed here from the parameters for a standalone
 * model; a Python instance pushes its own matrices.
 *
 * | Parameter     | Default | Meaning                                        |
 * |---------------|---------|------------------------------------------------|
 * | `delta`       | 0.5     | bandwidth of the process noise [s]             |
 * | `sigma_a`     | 0.1     | log-amplitude process noise                    |
 * | `sigma_psi`   | 0.5     | phase process noise                            |
 * | `sigma_theta` | 0.1     | theta process noise                            |
 * | `sigma_phi`   | 0.1     | phi process noise                              |
 * | `sigma_n`     | 1       | measurement noise, per real component          |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_GRAVITATIONAL_POLARIZATION_H
#define KALMAN_APPS_GRAVITATIONAL_POLARIZATION_H

#include "Kalman/Apps/Model.h"

namespace Kalman::Apps::Gravitational
{
    class Polarization : public Kalman::Apps::Model
    {
        public:
            Polarization();

            const char *Title() const override { return "Polarization EKF"; }
            const char *Description() const override
            { return "EKF for gravitational-wave polarization (Jones formalism): 8-D log-amplitude state, "
                     "4-D real observation."; }

            /// The transition over one sample: four constant-rate pairs, dt = 1/fs.
            Matrix F(double t = 0.) const override;
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            /// The Jones vector, as [Re z1, Im z1, Re z2, Im z2].
            Vector h(const Vector &x, double t = 0.) const override;
            /// Analytic dh/dx (4 x 8).
            Matrix JacobianH(const Vector &x, double t = 0.) const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_GRAVITATIONAL_POLARIZATION_H
