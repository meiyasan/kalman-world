/**
 * *********************************************
 *
 * \file Observability.cc
 * \brief Source of the Kalman::Observability class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Observability.h"
#include "Kalman/Algebra/Cholesky.h"
#include "Kalman/Algebra/Symmetric.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace Kalman
{
    Observability::Observability(const Model &model, std::size_t steps, const Vector *x0, double tolerance)
        : fTolerance(tolerance)
    {
        const std::size_t n = model.NState();
        if (steps == 0) throw std::invalid_argument("Kalman::Observability: consider at least one step.");
        Vector x = x0 ? *x0 : model.X0();
        if (x.Size() != n)
            throw std::invalid_argument("Kalman::Observability: the initial state has " + std::to_string(x.Size())
                                        + " components, " + std::to_string(n) + " expected.");

        const Vector u(model.NControl(), 0.);
        const double dt = model.DeltaT();
        double t = model.T0();

        Matrix G(n, n, 0.);
        Matrix Phi = Matrix::Identity(n);
        for (std::size_t k = 0; k < steps; ++k)
        {
            // The filter measures after it steps: the first measurement is of x_1.
            const Matrix F = model.Jacobian(x, u, t);
            x = model.f(x, u, t);
            Phi = F * Phi;
            t += dt;

            const Matrix HPhi = model.JacobianH(x, t) * Phi;
            G += HPhi.Transpose() * (Algebra::Inverse(model.R(t)) * HPhi);
        }

        // Dimensionless: in units of the initial standard deviations.
        const Matrix &P0 = model.P0();
        Vector d(n, 1.);
        for (std::size_t i = 0; i < n; ++i)
            if (P0.Rows() == n && P0(i, i) > 0.) d(i) = std::sqrt(P0(i, i));
        fGramian = Matrix(n, n, 0.);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) fGramian(i, j) = d(i) * G(i, j) * d(j);
        Algebra::Symmetrize(fGramian);

        Vector values;
        Matrix vectors;
        if (!Algebra::SymmetricEigen(fGramian, values, vectors))
            throw std::runtime_error("Kalman::Observability: the eigendecomposition did not converge.");

        // Smallest first, whatever order the decomposition returns.
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), std::size_t{0});
        std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return values(a) < values(b); });
        fValues = Vector(n);
        fVectors = Matrix(n, n, 0.);
        for (std::size_t c = 0; c < n; ++c)
        {
            fValues(c) = values(order[c]);
            for (std::size_t i = 0; i < n; ++i) fVectors(i, c) = vectors(i, order[c]);
        }

        const double largest = n ? fValues(n - 1) : 0.;
        fRank = 0;
        for (std::size_t c = 0; c < n; ++c)
            if (largest > 0. && fValues(c) > fTolerance * largest) ++fRank;
    }

    double Observability::Weakness(std::size_t i) const
    {
        if (i >= fValues.Size()) throw std::out_of_range("Kalman::Observability: no such state component.");
        // The squared length of the component's axis projected on the weak
        // subspace: the columns are orthonormal, so the squares add up.
        double share = 0.;
        for (std::size_t c = 0; c < NWeak(); ++c) share += fVectors(i, c) * fVectors(i, c);
        return std::min(1., share);
    }

    std::vector<std::size_t> Observability::WeakComponents(double threshold) const
    {
        std::vector<std::size_t> out;
        for (std::size_t i = 0; i < fValues.Size(); ++i)
            if (Weakness(i) > threshold) out.push_back(i);
        return out;
    }
}
