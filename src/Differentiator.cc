/**
 * *********************************************
 *
 * \file Differentiator.cc
 * \brief Implementation of the Kalman::Differentiator schemes
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Differentiator.h"
#include "Kalman/Differentiators/Central.h"
#include "Kalman/Differentiators/Forward.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace Kalman::Differentiators
{
    double DefaultStep(DifferentiatorKind kind)
    {
        // Forward: the step of the Python framework's numerical_jacobian.
        // Central: its error falls as the square of the step, so it can be larger.
        return kind == DifferentiatorKind::Forward ? 1e-6 : 1e-5;
    }

    Matrix Forward::Jacobian(const Map &fn, const Vector &x, const Vector *y0) const
    {
        const std::size_t n = x.Size();
        const Vector base = y0 ? *y0 : fn(x);
        Matrix J(base.Size(), n, 0.);
        for (std::size_t j = 0; j < n; ++j)
        {
            const double step = fStep * std::max(1., std::abs(x(j)));
            Vector xp(x);
            xp(j) += step;
            const Vector y1 = fn(xp);
            for (std::size_t i = 0; i < J.Rows(); ++i) J(i, j) = (y1(i) - base(i)) / step;
        }
        return J;
    }

    Matrix Central::Jacobian(const Map &fn, const Vector &x, const Vector *y0) const
    {
        const std::size_t n = x.Size();
        // The value at x is not part of the formula; it only gives the number of rows.
        std::size_t rows = y0 ? y0->Size() : 0;
        Matrix J;
        for (std::size_t j = 0; j < n; ++j)
        {
            const double step = fStep * std::max(1., std::abs(x(j)));
            Vector xp(x), xm(x);
            xp(j) += step;
            xm(j) -= step;
            const Vector fp = fn(xp), fm = fn(xm);
            if (J.Empty())
            {
                rows = fp.Size();
                J = Matrix(rows, n, 0.);
            }
            for (std::size_t i = 0; i < rows; ++i) J(i, j) = (fp(i) - fm(i)) / (2. * step);
        }
        if (J.Empty()) J = Matrix(y0 ? y0->Size() : fn(x).Size(), n, 0.);
        return J;
    }

    std::unique_ptr<Differentiator> Make(DifferentiatorKind kind, double step)
    {
        switch (kind)
        {
            case DifferentiatorKind::Forward: return std::make_unique<Forward>(step);
            case DifferentiatorKind::Central: return std::make_unique<Central>(step);
            case DifferentiatorKind::Analytic:
            case DifferentiatorKind::Automatic: break;
        }
        std::ostringstream os;
        os << "Kalman::Differentiators::Make: '" << ToString(kind) << "' is not a numerical scheme"
           << (kind == DifferentiatorKind::Automatic ? " (automatic differentiation is reserved for libKalmanLearn)."
                                                      : " (an analytic Jacobian is given by the model).");
        throw std::runtime_error(os.str());
    }

    Matrix Jacobian(DifferentiatorKind kind, const Map &fn, const Vector &x, double step, const Vector *y0)
    {
        if (kind == DifferentiatorKind::Forward) return Forward(step).Jacobian(fn, x, y0);
        if (kind == DifferentiatorKind::Central) return Central(step).Jacobian(fn, x, y0);
        return Make(kind, step)->Jacobian(fn, x, y0);      // throws with the reason
    }
}
