/**
 * \file Position.cc
 * \brief Implementation of the Kalman::Sensors::Position class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 */

#include "Kalman/Sensors/Position.h"
#include "Kalman/SensorFactory.h"
#include "Detail.h"

namespace Kalman::Sensors
{
    namespace
    {
        Matrix Selection(const std::string &who, std::size_t n, const std::vector<std::size_t> &axes,
                         const Matrix &rotation)
        {
            Detail::CheckIndices(who, axes, n);
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

        const bool gRegistered = SensorFactory::Instance().Register(
            "position", [](const std::string &name, std::size_t n, const std::vector<double> &numbers) {
                Detail::Reader in{"sensor kind 'position'", numbers};
                const std::size_t k = in.Index();
                const std::vector<std::size_t> axes = in.Indices(k);
                const double sigma = in.Number();
                const Matrix rotation = in.Left() ? in.Square(k) : Matrix();
                in.Done();
                return Sensor(Position(name, n, axes, sigma, rotation));
            });
    }

    Position::Position(std::string name, std::size_t n, const std::vector<std::size_t> &axes, double sigma,
                       const Matrix &rotation)
        : Kalman::Sensor(name, Selection("Sensors::Position '" + name + "'", n, axes, rotation),
                         Detail::Diagonal(axes.size(), Detail::Variance("Sensors::Position '" + name + "'", sigma)))
    {
    }
}
