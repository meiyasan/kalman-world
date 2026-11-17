/**
 * *********************************************
 *
 * \file Sensors.cc
 * \brief Implementation of Kalman::Apps::Fusion
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Fusion/Sensors.h"
#include "Kalman/SensorFactory.h"

#include <cmath>
#include <stdexcept>

namespace Kalman::Apps::Fusion
{
    namespace
    {
        /// \throws std::invalid_argument when an index is not a component of a state of n.
        void CheckIndices(const std::string &who, const std::vector<std::size_t> &indices, std::size_t n)
        {
            if (indices.empty()) throw std::invalid_argument(who + ": no state component to read.");
            for (std::size_t i : indices)
                if (i >= n)
                    throw std::invalid_argument(who + ": component " + std::to_string(i)
                                                + " is not in a state of " + std::to_string(n) + ".");
        }

        double Variance(const std::string &who, double sigma)
        {
            if (!(sigma > 0.)) throw std::invalid_argument(who + ": sigma must be positive.");
            return sigma * sigma;
        }

        Matrix Diagonal(std::size_t k, double value)
        {
            Matrix R(k, k, 0.);
            for (std::size_t i = 0; i < k; ++i) R(i, i) = value;
            return R;
        }

        // --- reading the settings of a registered kind -----------------------------------
        struct Reader
        {
            const std::string &who;
            const std::vector<double> &values;
            std::size_t next = 0;

            std::size_t Left() const { return values.size() - next; }
            double Number()
            {
                if (next >= values.size()) throw std::invalid_argument(who + ": too few settings.");
                return values[next++];
            }
            std::size_t Index()
            {
                const double value = Number();
                if (value < 0. || value != std::floor(value))
                    throw std::invalid_argument(who + ": an index is expected, got " + std::to_string(value) + ".");
                return static_cast<std::size_t>(value);
            }
            std::vector<std::size_t> Indices(std::size_t count)
            {
                std::vector<std::size_t> indices(count);
                for (std::size_t &i : indices) i = Index();
                return indices;
            }
            Vector Numbers(std::size_t count)
            {
                Vector v(count);
                for (std::size_t i = 0; i < count; ++i) v(i) = Number();
                return v;
            }
            Matrix Square(std::size_t k)
            {
                Matrix m(k, k, 0.);
                for (std::size_t i = 0; i < k; ++i)
                    for (std::size_t j = 0; j < k; ++j) m(i, j) = Number();
                return m;
            }
            void Done() const
            {
                if (next != values.size()) throw std::invalid_argument(who + ": too many settings.");
            }
        };
    }

    // ---------------------------------------------------------------------------------------
    // PositionSensor
    // ---------------------------------------------------------------------------------------
    namespace
    {
        Matrix Selection(const std::string &name, std::size_t n, const std::vector<std::size_t> &axes,
                         const Matrix &rotation)
        {
            const std::string who = "PositionSensor '" + name + "'";
            CheckIndices(who, axes, n);
            const std::size_t k = axes.size();
            if (rotation.Rows() && (rotation.Rows() != k || rotation.Cols() != k))
                throw std::invalid_argument(who + ": the rotation must be " + std::to_string(k) + " x "
                                            + std::to_string(k) + ".");
            Matrix G(k, n, 0.);
            for (std::size_t i = 0; i < k; ++i)
                for (std::size_t j = 0; j < k; ++j)
                    G(i, axes[j]) += rotation.Rows() ? rotation(i, j) : (i == j ? 1. : 0.);
            return G;
        }
    }

    PositionSensor::PositionSensor(std::string name, std::size_t n, const std::vector<std::size_t> &axes,
                                   double sigma, const Matrix &rotation)
        : Kalman::Sensor(name, Selection(name, n, axes, rotation),
                         Diagonal(axes.size(), Variance("PositionSensor '" + name + "'", sigma)))
    {
    }

    // ---------------------------------------------------------------------------------------
    // DopplerRadar
    // ---------------------------------------------------------------------------------------
    Sensor DopplerRadar::Reading(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                                 const std::vector<std::size_t> &velocity, double sigma, const Vector &at)
    {
        const std::string who = "DopplerRadar '" + name + "'";
        CheckIndices(who, position, n);
        CheckIndices(who, velocity, n);
        const std::size_t d = position.size();
        if (velocity.size() != d)
            throw std::invalid_argument(who + ": as many velocity components as position components are expected.");
        if (at.Size() && at.Size() != d)
            throw std::invalid_argument(who + ": where the radar is takes " + std::to_string(d) + " numbers.");
        Vector origin(d, 0.);
        for (std::size_t i = 0; i < at.Size(); ++i) origin(i) = at(i);

        // The line of sight u = (p - at) / r and the reading v . u.
        struct Geometry { Vector u; double r, vr; };
        const auto geometry = [position, velocity, origin, d](const Vector &x) {
            Geometry g{Vector(d, 0.), 0., 0.};
            for (std::size_t i = 0; i < d; ++i)
            {
                g.u(i) = x(position[i]) - origin(i);
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
        return Sensor(std::move(name), std::move(measure), Diagonal(1, Variance(who, sigma)), std::move(jacobian));
    }

    DopplerRadar::DopplerRadar(std::string name, std::size_t n, const std::vector<std::size_t> &position,
                               const std::vector<std::size_t> &velocity, double sigma, const Vector &at)
        : Kalman::Instrument(name, {Reading(name, n, position, velocity, sigma, at)})
    {
    }

    // ---------------------------------------------------------------------------------------
    // Camera
    // ---------------------------------------------------------------------------------------
    Sensor Camera::Reading(std::string name, std::size_t n, const std::vector<std::size_t> &position, double focal,
                           double sigma, const Vector &at, const Matrix &rotation)
    {
        const std::string who = "Camera '" + name + "'";
        CheckIndices(who, position, n);
        if (position.size() != 3) throw std::invalid_argument(who + ": a point in space has three coordinates.");
        if (!(focal > 0.)) throw std::invalid_argument(who + ": the focal length must be positive.");
        if (at.Size() && at.Size() != 3) throw std::invalid_argument(who + ": where the camera is takes 3 numbers.");
        if (rotation.Rows() && (rotation.Rows() != 3 || rotation.Cols() != 3))
            throw std::invalid_argument(who + ": the rotation must be 3 x 3.");
        Vector origin(3, 0.);
        for (std::size_t i = 0; i < at.Size(); ++i) origin(i) = at(i);
        const Matrix W = rotation.Rows() ? rotation : Matrix::Identity(3);

        // The point in the camera's axes, its depth kept above NEAR.
        const auto seen = [position, origin, W](const Vector &x) {
            Vector q(3, 0.);
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j) q(i) += W(i, j) * (x(position[j]) - origin(j));
            if (q(2) < NEAR) q(2) = NEAR;
            return q;
        };

        Sensor::Measure measure = [seen, focal](const Vector &x, double) {
            const Vector q = seen(x);
            Vector z(2);
            z(0) = focal * q(0) / q(2);
            z(1) = focal * q(1) / q(2);
            return z;
        };
        // d(u, v) / dq = (focal / q_z) [ 1 0 -q_x/q_z ; 0 1 -q_y/q_z ],  dq / dp = W
        Sensor::MeasureJacobian jacobian = [seen, focal, position, W, n](const Vector &x, double) {
            const Vector q = seen(x);
            const double s = focal / q(2);
            Matrix J(2, n, 0.);
            for (std::size_t j = 0; j < 3; ++j)
            {
                J(0, position[j]) += s * (W(0, j) - q(0) / q(2) * W(2, j));
                J(1, position[j]) += s * (W(1, j) - q(1) / q(2) * W(2, j));
            }
            return J;
        };
        return Sensor(std::move(name), std::move(measure), Diagonal(2, Variance(who, sigma)), std::move(jacobian));
    }

    Camera::Camera(std::string name, std::size_t n, const std::vector<std::size_t> &position, double focal,
                   double sigma, const Vector &at, const Matrix &rotation)
        : Kalman::Instrument(name, {Reading(name, n, position, focal, sigma, at, rotation)})
    {
    }

    // ---------------------------------------------------------------------------------------
    // By name, for a program that holds no C++ (Kalman::SensorFactory)
    // ---------------------------------------------------------------------------------------
    namespace
    {
        const bool gRegistered = [] {
            SensorFactory &factory = SensorFactory::Instance();
            factory.Register("position", [](const std::string &name, std::size_t n, const std::vector<double> &settings) {
                const std::string who = "sensor kind 'position'";
                Reader in{who, settings};
                const std::size_t k = in.Index();
                const std::vector<std::size_t> axes = in.Indices(k);
                const double sigma = in.Number();
                const Matrix rotation = in.Left() ? in.Square(k) : Matrix();
                in.Done();
                return Sensor(PositionSensor(name, n, axes, sigma, rotation));
            });
            factory.Register("doppler", [](const std::string &name, std::size_t n, const std::vector<double> &settings) {
                const std::string who = "sensor kind 'doppler'";
                Reader in{who, settings};
                const std::size_t d = in.Index();
                const std::vector<std::size_t> position = in.Indices(d), velocity = in.Indices(d);
                const double sigma = in.Number();
                const Vector at = in.Left() ? in.Numbers(d) : Vector();
                in.Done();
                return DopplerRadar::Reading(name, n, position, velocity, sigma, at);
            });
            factory.Register("camera", [](const std::string &name, std::size_t n, const std::vector<double> &settings) {
                const std::string who = "sensor kind 'camera'";
                Reader in{who, settings};
                const std::vector<std::size_t> position = in.Indices(3);
                const double focal = in.Number(), sigma = in.Number();
                Vector at;
                Matrix rotation;
                if (in.Left())
                {
                    at = in.Numbers(3);
                    rotation = in.Square(3);
                }
                in.Done();
                return Camera::Reading(name, n, position, focal, sigma, at, rotation);
            });
            return true;
        }();
    }
}
