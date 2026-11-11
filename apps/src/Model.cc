/**
 * *********************************************
 *
 * \file Model.cc
 * \brief Implementation of the Kalman::Apps::Model class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Model.h"

#include <algorithm>
#include <cmath>

namespace Kalman::Apps
{
    namespace
    {
        // helpers.numerical_jacobian, line for line: one extra evaluation per
        // state component, each column (f(x + step e_j) - f(x)) / step.
        template <typename Map>
        Matrix ForwardDifference(const Map &map, const Vector &x)
        {
            const std::size_t n = x.Size();
            const Vector y0 = map(x);
            Matrix J(y0.Size(), n, 0.);

            for (std::size_t j = 0; j < n; ++j)
            {
                const double step = Model::kJacobianStep * std::max(1., std::abs(x(j)));
                Vector xp(x);
                xp(j) += step;
                const Vector y1 = map(xp);
                for (std::size_t i = 0; i < J.Rows(); ++i) J(i, j) = (y1(i) - y0(i)) / step;
            }
            return J;
        }
    }

    Matrix Model::Jacobian(const Vector &x, const Vector &u, double t) const
    {
        return ForwardDifference([this, &u, t](const Vector &xx) { return this->f(xx, u, t); }, x);
    }

    Matrix Model::JacobianH(const Vector &x, double t) const
    {
        return ForwardDifference([this, t](const Vector &xx) { return this->h(xx, t); }, x);
    }
}
