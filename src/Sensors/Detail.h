/**
 * \file Detail.h
 * \brief What the sensors of Kalman::Sensors share: checks, and reading the numbers of a registered kind.
 *
 * Not installed: included by the sources of this directory only.
 */
#ifndef KALMAN_SENSORS_DETAIL_H
#define KALMAN_SENSORS_DETAIL_H

#include "Kalman/Matrix.h"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace Kalman::Sensors::Detail
{
    /// \throws std::invalid_argument when an index is not a component of a state of n.
    inline void CheckIndices(const std::string &who, const std::vector<std::size_t> &indices, std::size_t n)
    {
        if (indices.empty()) throw std::invalid_argument(who + ": no state component to read.");
        for (std::size_t i : indices)
            if (i >= n)
                throw std::invalid_argument(who + ": component " + std::to_string(i) + " is not in a state of "
                                            + std::to_string(n) + ".");
    }

    inline double Variance(const std::string &who, double sigma)
    {
        if (!(sigma > 0.)) throw std::invalid_argument(who + ": sigma must be positive.");
        return sigma * sigma;
    }

    inline Matrix Diagonal(std::size_t k, double value)
    {
        Matrix R(k, k, 0.);
        for (std::size_t i = 0; i < k; ++i) R(i, i) = value;
        return R;
    }

    /// The numbers a registered kind is built from (Kalman::SensorFactory), read in order.
    struct Reader
    {
        std::string who;
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

#endif // KALMAN_SENSORS_DETAIL_H
