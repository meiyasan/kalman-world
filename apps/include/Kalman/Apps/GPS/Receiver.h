/**
 * *********************************************
 *
 * \file Receiver.h
 * \brief Header of the Kalman::Apps::GPS::Receiver class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::GPS::Receiver
 * \brief A GPS receiver solving for its position and clock from pseudoranges.
 *
 * \details The twin of the Python `GPSModel` (modules/apps/gps/model.py). The
 * state is `[x, y, z, b, bdot]`: the receiver's position in an Earth-centred
 * frame, its clock bias and clock drift, in metres and metres per second.
 * The receiver stands still and its clock bias follows its drift, `db/dt = bdot`:
 * a linear system stated in continuous time (a generator), with white noise
 * on the position and on the drift rate, both decoded for the sample rate.
 *
 * The measurement is one pseudorange per satellite,
 * `rho_i = |s_i(t) - p| + b`, with the line-of-sight Jacobian given
 * analytically. The satellites follow Kepler orbits; each is given by six
 * elements, held in six named arrays of one entry per satellite. The number
 * of measurements is the number of satellites: setting the arrays resizes the
 * model.
 *
 * The relativistic corrections are applied to the pseudoranges before the
 * filter (the Python class's preprocess()), not here.
 *
 * | Array | Unit | |
 * |---|---|---|
 * | `semi_major_axis` | m | |
 * | `eccentricity` | | |
 * | `inclination`, `raan`, `arg_perigee` | rad | orientation of the orbit |
 * | `mean_anomaly` | rad | at t = 0 |
 *
 * | Parameter | Default | |
 * |---|---|---|
 * | `pseudorange_uncertainty` | 1 | m, standard deviation of a pseudorange |
 * | `process_uncertainty` | 0.2 | m per sqrt(s), random walk of the position |
 * | `clock_drift_uncertainty` | 1e-3 | m/s per sqrt(s), random walk of the drift |
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_GPS_RECEIVER_H
#define KALMAN_APPS_GPS_RECEIVER_H

#include "Kalman/Apps/Model.h"

#include <array>
#include <vector>

namespace Kalman::Apps::GPS
{
    class Receiver : public Kalman::Apps::Model
    {
        public:
            static constexpr double kMuEarth = 3.986004418e14;   ///< [m^3/s^2]
            static constexpr double kEarthRadius = 6371000.;     ///< mean radius [m]
            static constexpr double kOrbitRadius = 26560000.;    ///< GPS semi-major axis [m]

        private:
            struct Satellite
            {
                double a, e, meanMotion, m0;
                std::array<double, 9> rotation;   // orbital plane to the Earth-centred frame, row-major
            };
            std::vector<Satellite> fSatellites;
            bool fConsistent = true;              // the six arrays have the same length

        public:
            Receiver();

            const char *Title() const override { return "GPS Relativistic Positioning Model"; }
            const char *Description() const override
            { return "EKF over [x,y,z,clock-bias,clock-drift] from satellite pseudoranges."; }

            std::size_t NSatellites() const { return fSatellites.size(); }
            /// Position of satellite \p i at time \p t [m], written to \p position (3 values).
            void Position(std::size_t i, double t, double *position) const;

            /// One pseudorange per satellite: the range to it plus the clock bias.
            Vector h(const Vector &x, double t = 0.) const override;
            /// The transition matrix: the dynamics are linear.
            Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const override;
            /// The lines of sight, and 1 for the clock bias.
            Matrix JacobianH(const Vector &x, double t = 0.) const override;

            /// \throws std::runtime_error when the six orbit arrays do not have the same length.
            void CheckShapes() const override;

        protected:
            void ParametersChanged() override;
    };
}

#endif // KALMAN_APPS_GPS_RECEIVER_H
