/**
 * \file Doppler.cc
 * \brief Implementation of the Kalman::Sensors::Doppler class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 */

#include "Kalman/Sensors/Doppler.h"
#include "Kalman/SensorFactory.h"
#include "Detail.h"

namespace Kalman::Sensors
{
    namespace
    {
        const char *const AXES[] = {"x", "y", "z"};

        Sensor Make(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                    const std::vector<std::size_t> &velocity, double sigma, const Vector &at)
        {
            const std::string who = "Sensors::Doppler '" + name + "'";
            Detail::CheckIndices(who, position, n);
            Detail::CheckIndices(who, velocity, n);
            const std::size_t d = position.size();
            if (d > 3) throw std::invalid_argument(who + ": at most three dimensions.");
            if (velocity.size() != d)
                throw std::invalid_argument(who + ": as many velocity components as position components are expected.");
            if (at.Size() && at.Size() != d)
                throw std::invalid_argument(who + ": where the sensor is takes " + std::to_string(d) + " numbers.");

            // Where the sensor is: its settings, read at every call.
            auto settings = std::make_shared<Sensor::Settings>();
            for (std::size_t i = 0; i < d; ++i) settings->emplace_back(AXES[i], at.Size() ? at(i) : 0.);

            // The line of sight u = (p - at) / r and the reading v . u.
            struct Geometry { Vector u; double r, vr; };
            const auto geometry = [position, velocity, settings, d](const Vector &x) {
                Geometry g{Vector(d, 0.), 0., 0.};
                for (std::size_t i = 0; i < d; ++i)
                {
                    g.u(i) = x(position[i]) - (*settings)[i].second;
                    g.r += g.u(i) * g.u(i);
                }
                g.r = std::sqrt(g.r);
                if (g.r > 0.)
                    for (std::size_t i = 0; i < d; ++i)
                    {
                        g.u(i) /= g.r;
                        g.vr += x(velocity[i]) * g.u(i);
                    }
                return g;
            };

            Sensor::Measure measure = [geometry](const Vector &x, double) {
                Vector z(1);
                z(0) = geometry(x).vr;
                return z;
            };
            // d v_r / d v = u,   d v_r / d p = (v - v_r u) / r
            Sensor::MeasureJacobian jacobian = [geometry, position, velocity, n, d](const Vector &x, double) {
                const Geometry g = geometry(x);
                Matrix J(1, n, 0.);
                if (!(g.r > 0.)) return J;
                for (std::size_t i = 0; i < d; ++i)
                {
                    J(0, velocity[i]) += g.u(i);
                    J(0, position[i]) += (x(velocity[i]) - g.vr * g.u(i)) / g.r;
                }
                return J;
            };
            Sensor sensor(std::move(name), std::move(measure), Detail::Diagonal(1, Detail::Variance(who, sigma)),
                          std::move(jacobian));
            sensor.settings = std::move(settings);
            return sensor;
        }

        const bool gRegistered = SensorFactory::Instance().Register(
            "doppler", [](const std::string &name, std::size_t n, const std::vector<double> &numbers) {
                Detail::Reader in{"sensor kind 'doppler'", numbers};
                const std::size_t d = in.Index();
                const std::vector<std::size_t> position = in.Indices(d), velocity = in.Indices(d);
                const double sigma = in.Number();
                const Vector at = in.Left() ? in.Numbers(d) : Vector();
                in.Done();
                return Sensor(Doppler(name, n, position, velocity, sigma, at));
            });
    }

    Doppler::Doppler(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                     const std::vector<std::size_t> &velocity, double sigma, const Vector &at)
        : Kalman::Sensor(Make(std::move(name), n, position, velocity, sigma, at))
    {
    }
}
