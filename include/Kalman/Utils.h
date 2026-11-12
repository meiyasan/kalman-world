/**
 * *********************************************
 *
 * \file Utils.h
 * \brief Small dense linear-algebra helpers shared by the engines
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details The C++ counterpart of the numeric helpers in ``modules/helpers.py``
 * and ``modules/framework/native/kalman.c``. Everything here is a free function
 * over Kalman::Vector / Kalman::Matrix: no state, no allocation policy, nothing
 * that belongs to a model or an engine.
 *
 * Two conventions are load-bearing and shared with the Python framework:
 *
 * - **NaN marks a missing measurement channel.** ValidChannels() reports the
 *   channels that are present, and the Compress* helpers cut `z`/`H`/`R` down to
 *   them, so a partially-blind sensor costs no special case upstream.
 * - **Symmetric inverses degrade instead of failing.** Inverse() tries Cholesky,
 *   retries with a growing Tikhonov ridge, and finally falls back to an
 *   eigenvalue pseudo-inverse. A covariance that has drifted slightly
 *   non-positive-definite must not abort a run.
 *
 * *********************************************
 */

#ifndef KALMAN_UTILS_H
#define KALMAN_UTILS_H

#include "Kalman/Matrix.h"
#include "Kalman/Algebra/Algebra.h"
#include "Kalman/Differentiator.h"

#include <cstddef>
#include <functional>
#include <vector>

namespace Kalman::Utils
{
    // The linear algebra lives in Kalman::Algebra and the Jacobians in
    // Kalman::Differentiators. These names are kept so that code written
    // against Kalman::Utils keeps compiling.
    using Algebra::Similarity;
    using Algebra::Symmetrize;
    using Algebra::SymmetricEigen;
    using Algebra::ConditionNumber;
    using Algebra::CholeskyFactor;
    using Algebra::Inverse;
    using Algebra::Solve;
    using Algebra::Exponential;
    using Algebra::VanLoan;

    /// Central-difference Jacobian `df/dx` at \p x (Differentiators::Central).
    inline Matrix NumericalJacobian(const std::function<Vector(const Vector &)> &fn,
                                    const Vector &x, double eps = 1e-5)
    { return Differentiators::Jacobian(DifferentiatorKind::Central, fn, x, eps); }


    /// Indices of the channels of \p z that are present (not NaN).
    std::vector<std::size_t> ValidChannels(const Vector &z);

    /// True when every channel of \p z is present.
    bool IsComplete(const Vector &z);

    /// \p v restricted to \p keep.
    Vector Compress(const Vector &v, const std::vector<std::size_t> &keep);

    /// Rows of \p M restricted to \p keep (for the observation matrix H).
    Matrix CompressRows(const Matrix &M, const std::vector<std::size_t> &keep);

    /// Rows *and* columns of \p M restricted to \p keep (for the noise matrix R).
    Matrix CompressBoth(const Matrix &M, const std::vector<std::size_t> &keep);

    /// Quadratic form `y^T S^-1 y`, given an already-inverted \p Sinv.
    double Mahalanobis(const Vector &y, const Matrix &Sinv);

    /**
     * \brief Joseph-form covariance update `(I-KH) P (I-KH)^T + K R K^T`.
     *
     * Preferred over the shorter `(I-KH) P`: it stays symmetric and positive
     * semi-definite under round-off, and it is valid for a suboptimal gain.
     */
    Matrix JosephForm(const Matrix &K, const Matrix &H, const Matrix &P, const Matrix &R);

    // --- the steps themselves, on already-compressed channels ------------------
    // These are the arithmetic behind Engine::Update, the sigma-point moments and
    // the particle resampler, exposed so the C kernels the Python engines call are
    // the same code and not a second implementation.

    /**
     * \brief The Kalman correction from a residual: gain, mean, Joseph-form covariance.
     * \param Hc  (p, n) observation rows of the channels present.
     * \param Rc  (p, p) their noise.
     * \param yc  (p) the residual `z - h(x)` on those channels.
     * \param x, P the prediction.
     * \param[out] xOut, POut the correction.
     */
    void Correct(const Matrix &Hc, const Matrix &Rc, const Vector &yc,
                 const Vector &x, const Matrix &P, Vector &xOut, Matrix &POut);

    /**
     * \brief `out = sum_s w[s] (A[s] - a)(B[s] - b)^T` over `nsigma` rows.
     *
     * Flat row-major inputs: A is (nsigma, na), B is (nsigma, nb), out is (na, nb).
     * The one accumulation every sigma-point moment needs.
     */
    void WeightedOuterSum(std::size_t nsigma, std::size_t na, std::size_t nb,
                          const double *w, const double *A, const double *a,
                          const double *B, const double *b, double *out);

    /**
     * \brief Systematic resampling: `indices[i]` is the particle at position `(i + u0) / n`
     * on the cumulative weight; \p u0 is one uniform draw in [0, 1).
     */
    void SystematicResample(std::size_t n, const double *weights, double u0, int *indices);
}

#endif // KALMAN_UTILS_H
