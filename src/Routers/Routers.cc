/**
 * *********************************************
 *
 * \file Routers.cc
 * \brief Implementation of the rule-based routers
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Routers/Bayesian.h"
#include "Kalman/Routers/Hybrid.h"
#include "Kalman/Routers/Threshold.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Routers
{
    // --- Router (shared) -----------------------------------------------------

    Vector Router::Normalise(Vector mu)
    {
        double sum = 0.;
        for (std::size_t i = 0; i < mu.Size(); ++i)
        {
            // A mode that reaches exactly zero can never come back: every later
            // update multiplies it by something finite. The floor is what keeps a
            // regime recoverable after the filter has stopped believing in it.
            if (!(mu(i) > 1e-12)) mu(i) = 1e-12;
            sum += mu(i);
        }
        if (!(sum > 0.))                       // every entry was NaN
            return Vector(mu.Size(), 1. / static_cast<double>(mu.Size()));

        mu *= 1. / sum;
        return mu;
    }

    Matrix Router::NisWindow(const Matrix &nis, std::size_t window)
    {
        const std::size_t M = nis.Cols();
        const std::size_t have = std::min(window, nis.Rows());
        const std::size_t pad = window - have;

        Matrix out(window, M, 1.0);            // 1.0 = the chi2(1) mean
        for (std::size_t r = 0; r < have; ++r)
            for (std::size_t j = 0; j < M; ++j)
                out(pad + r, j) = nis(nis.Rows() - have + r, j);
        return out;
    }

    // --- Bayesian ------------------------------------------------------------

    Vector Bayesian::Route(const Matrix &, const Vector &c, const Vector &likelihoods) const
    {
        if (c.Size() != likelihoods.Size())
            throw std::runtime_error("Kalman::Routers::Bayesian: c and likelihoods differ in length.");

        Vector mu(c.Size());
        for (std::size_t j = 0; j < c.Size(); ++j) mu(j) = c(j) * likelihoods(j);
        return Normalise(mu);
    }

    // --- Threshold -----------------------------------------------------------

    Vector Threshold::Route(const Matrix &nis, const Vector &c, const Vector &) const
    {
        const std::size_t M = nis.Cols() ? nis.Cols() : c.Size();
        if (nis.Rows() == 0)
            return Vector(M, 1. / static_cast<double>(M));   // nothing observed yet

        // The mean of the last `window` rows, or of all of them while there are
        // fewer: a short history is averaged as it is, not padded (padding with the
        // chi2(1) mean would pull every model toward 1 whatever m is). The same
        // rule as routers/threshold.py, which forwards here; a window of 0 is the
        // whole history, as `nis[-0:]` is in numpy.
        const std::size_t rows = fWindow == 0 ? nis.Rows() : std::min(fWindow, nis.Rows());
        Vector dist(M);
        for (std::size_t j = 0; j < M; ++j)
        {
            double mean = 0.;
            for (std::size_t r = nis.Rows() - rows; r < nis.Rows(); ++r) mean += nis(r, j);
            mean /= static_cast<double>(rows);
            dist(j) = std::abs(mean - static_cast<double>(fM));
        }

        if (fSoft)
        {
            Vector mu(M);
            for (std::size_t j = 0; j < M; ++j) mu(j) = 1. / (dist(j) + 1e-6);
            return Normalise(mu);
        }

        std::size_t best = 0;
        for (std::size_t j = 1; j < M; ++j)
            if (dist(j) < dist(best)) best = j;

        Vector mu(M, 0.);
        mu(best) = 1.;
        return mu;                              // hard: deliberately not floored
    }

    // --- Hybrid --------------------------------------------------------------

    Hybrid::Hybrid(const Router *a, const Router *b, double alpha) : fA(a), fB(b)
    {
        if (!a || !b) throw std::runtime_error("Kalman::Routers::Hybrid: both routers are required.");
        SetAlpha(alpha);
    }

    void Hybrid::SetAlpha(double alpha)
    {
        if (!(alpha >= 0. && alpha <= 1.))
            throw std::runtime_error("Kalman::Routers::Hybrid: alpha must lie in [0, 1].");
        fAlpha = alpha;
    }

    Vector Hybrid::Route(const Matrix &nis, const Vector &c, const Vector &likelihoods) const
    {
        const Vector a = Normalise(fA->Route(nis, c, likelihoods));
        const Vector b = Normalise(fB->Route(nis, c, likelihoods));

        Vector mu(a.Size());
        for (std::size_t j = 0; j < a.Size(); ++j)
            mu(j) = std::pow(a(j), 1. - fAlpha) * std::pow(b(j), fAlpha);
        return Normalise(mu);
    }
}
