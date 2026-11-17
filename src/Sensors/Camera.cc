/**
 * \file Camera.cc
 * \brief Implementation of the Kalman::Sensors::Camera class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 */

#include "Kalman/Sensors/Camera.h"
#include "Kalman/SensorFactory.h"
#include "Detail.h"

namespace Kalman::Sensors
{
    namespace
    {
        Sensor Make(std::string name, std::size_t n, const std::vector<std::size_t> &position, double focal,
                    double sigma, const Vector &at, const Matrix &rotation)
        {
            const std::string who = "Sensors::Camera '" + name + "'";
            Detail::CheckIndices(who, position, n);
            if (position.size() != 3) throw std::invalid_argument(who + ": a point in space has three coordinates.");
            if (!(focal > 0.)) throw std::invalid_argument(who + ": the focal length must be positive.");
            if (at.Size() && at.Size() != 3) throw std::invalid_argument(who + ": where the camera is takes 3 numbers.");
            if (rotation.Rows() && (rotation.Rows() != 3 || rotation.Cols() != 3))
                throw std::invalid_argument(who + ": the rotation must be 3 x 3.");
            const Matrix W = rotation.Rows() ? rotation : Matrix::Identity(3);

            // Its settings, read at every call: [0] the focal length, [1..3] where it is.
            auto settings = std::make_shared<Sensor::Settings>();
            settings->emplace_back("focal", focal);
            for (std::size_t i = 0; i < 3; ++i)
                settings->emplace_back(i == 0 ? "x" : i == 1 ? "y" : "z", at.Size() ? at(i) : 0.);

            // The point in the camera's axes, its depth kept above NEAR.
            const auto seen = [position, settings, W](const Vector &x) {
                Vector q(3, 0.);
                for (std::size_t i = 0; i < 3; ++i)
                    for (std::size_t j = 0; j < 3; ++j)
                        q(i) += W(i, j) * (x(position[j]) - (*settings)[1 + j].second);
                if (q(2) < Camera::NEAR) q(2) = Camera::NEAR;
                return q;
            };

            Sensor::Measure measure = [seen, settings](const Vector &x, double) {
                const Vector q = seen(x);
                const double f = (*settings)[0].second;
                Vector z(2);
                z(0) = f * q(0) / q(2);
                z(1) = f * q(1) / q(2);
                return z;
            };
            // d(u, v) / dq = (focal / q_z) [ 1 0 -q_x/q_z ; 0 1 -q_y/q_z ],  dq / dp = W
            Sensor::MeasureJacobian jacobian = [seen, settings, position, W, n](const Vector &x, double) {
                const Vector q = seen(x);
                const double s = (*settings)[0].second / q(2);
                Matrix J(2, n, 0.);
                for (std::size_t j = 0; j < 3; ++j)
                {
                    J(0, position[j]) += s * (W(0, j) - q(0) / q(2) * W(2, j));
                    J(1, position[j]) += s * (W(1, j) - q(1) / q(2) * W(2, j));
                }
                return J;
            };
            Sensor sensor(std::move(name), std::move(measure), Detail::Diagonal(2, Detail::Variance(who, sigma)),
                          std::move(jacobian));
            sensor.settings = std::move(settings);
            return sensor;
        }

        const bool gRegistered = SensorFactory::Instance().Register(
            "camera", [](const std::string &name, std::size_t n, const std::vector<double> &numbers) {
                Detail::Reader in{"sensor kind 'camera'", numbers};
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
                return Sensor(Camera(name, n, position, focal, sigma, at, rotation));
            });
    }

    Camera::Camera(std::string name, std::size_t n, const std::vector<std::size_t> &position, double focal,
                   double sigma, const Vector &at, const Matrix &rotation)
        : Kalman::Sensor(Make(std::move(name), n, position, focal, sigma, at, rotation))
    {
    }
}
