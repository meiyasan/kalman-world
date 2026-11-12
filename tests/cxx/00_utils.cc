/**
 * \file 00_utils.cc
 * \brief The dense types and the shared algebra: products, inverses, NaN masking, Jacobians.
 */
#include "Kalman/Matrix.h"
#include "Kalman/Utils.h"
#include "Kalman/Differentiators/Central.h"
#include "Kalman/Differentiators/Forward.h"
#include "Kalman/Algebra/Algebra.h"
#include "Check.h"

#include <cmath>
#include <stdexcept>
#include <limits>

using namespace Kalman;

int main()
{
    // --- Vector / Matrix basics ---------------------------------------------
    {
        Vector v{1., 2., 3.};
        CHECK(v.Size() == 3, "Vector: initializer list");
        CHECK_CLOSE(v(2), 3., 1e-15, "Vector: element access");

        Matrix I = Matrix::Identity(3);
        CHECK(I.Rows() == 3 && I.Cols() == 3, "Matrix: identity shape");
        CHECK_CLOSE(I(1, 1), 1., 1e-15, "Matrix: identity diagonal");
        CHECK_CLOSE(I(1, 2), 0., 1e-15, "Matrix: identity off-diagonal");

        const Vector Iv = I * v;
        CHECK_CLOSE(Iv(1), 2., 1e-15, "Matrix: I x = x");

        Matrix A(2, 3, 0.);
        A(0, 0) = 1.; A(0, 1) = 2.; A(0, 2) = 3.;
        A(1, 0) = 4.; A(1, 1) = 5.; A(1, 2) = 6.;
        const Matrix At = A.Transpose();
        CHECK(At.Rows() == 3 && At.Cols() == 2, "Matrix: transpose shape");
        CHECK_CLOSE(At(2, 1), 6., 1e-15, "Matrix: transpose element");

        // non-square product, to catch row/col-major slips
        const Matrix AAt = A * At;                 // (2x3)(3x2) = 2x2
        CHECK(AAt.Rows() == 2 && AAt.Cols() == 2, "Matrix: non-square product shape");
        CHECK_CLOSE(AAt(0, 0), 1. + 4. + 9., 1e-12, "Matrix: product (0,0)");
        CHECK_CLOSE(AAt(0, 1), 4. + 10. + 18., 1e-12, "Matrix: product (0,1)");
        CHECK_CLOSE(AAt(1, 1), 16. + 25. + 36., 1e-12, "Matrix: product (1,1)");
    }

    // --- Similarity ----------------------------------------------------------
    {
        Matrix A(2, 2);
        A(0, 0) = 1.; A(0, 1) = 2.;
        A(1, 0) = 0.; A(1, 1) = 1.;

        Matrix P(2, 2);
        P(0, 0) = 4.; P(0, 1) = 1.;
        P(1, 0) = 1.; P(1, 1) = 9.;

        const Matrix S = Utils::Similarity(A, P);  // A P A^T, worked out by hand
        CHECK_CLOSE(S(0, 0), 4. + 4. + 36., 1e-12, "Similarity (0,0)");
        CHECK_CLOSE(S(0, 1), 1. + 18., 1e-12, "Similarity (0,1)");
        CHECK_CLOSE(S(1, 0), S(0, 1), 1e-12, "Similarity is symmetric here");
        CHECK_CLOSE(S(1, 1), 9., 1e-12, "Similarity (1,1)");
    }

    // --- Inverse: exact on a well-conditioned matrix --------------------------
    {
        Matrix S(2, 2);
        S(0, 0) = 4.; S(0, 1) = 1.;
        S(1, 0) = 1.; S(1, 1) = 3.;

        bool ok = false;
        const Matrix Sinv = Utils::Inverse(S, &ok);
        CHECK(ok, "Inverse: Cholesky succeeds on a positive-definite matrix");

        const Matrix I = S * Sinv;
        CHECK_CLOSE(I(0, 0), 1., 1e-12, "Inverse: S S^-1 = I (0,0)");
        CHECK_CLOSE(I(0, 1), 0., 1e-12, "Inverse: S S^-1 = I (0,1)");
        CHECK_CLOSE(I(1, 1), 1., 1e-12, "Inverse: S S^-1 = I (1,1)");
    }

    // --- Inverse: a singular matrix must degrade, not explode ------------------
    {
        Matrix S(2, 2, 0.);
        S(0, 0) = 1.;                              // rank 1

        bool ok = true;
        const Matrix Sinv = Utils::Inverse(S, &ok);
        CHECK(!ok, "Inverse: reports that the plain Cholesky failed");
        CHECK(Sinv.AllFinite(), "Inverse: singular input still gives finite output");
        // The pseudo-inverse of diag(1, 0) is diag(1, 0).
        CHECK_CLOSE(Sinv(0, 0), 1., 1e-9, "Inverse: pseudo-inverse keeps the live direction");
        CHECK_CLOSE(Sinv(1, 1), 0., 1e-9, "Inverse: pseudo-inverse drops the null direction");
    }

    // --- Symmetric eigendecomposition ------------------------------------------
    {
        Matrix S(2, 2);
        S(0, 0) = 2.; S(0, 1) = 1.;
        S(1, 0) = 1.; S(1, 1) = 2.;                // eigenvalues 1 and 3

        Vector values;
        Matrix vectors;
        CHECK(Utils::SymmetricEigen(S, values, vectors), "Eigen: converged");

        double lo = std::min(values(0), values(1)), hi = std::max(values(0), values(1));
        CHECK_CLOSE(lo, 1., 1e-10, "Eigen: smaller eigenvalue");
        CHECK_CLOSE(hi, 3., 1e-10, "Eigen: larger eigenvalue");
        CHECK_CLOSE(Utils::ConditionNumber(S), 3., 1e-9, "ConditionNumber: ratio of extremes");
    }

    // --- Cholesky factor of a positive SEMI-definite matrix --------------------
    {
        // The white-noise-acceleration block is exactly rank 1: sampling from it
        // must degrade via jitter rather than fail.
        Matrix Q(2, 2);
        Q(0, 0) = 0.25; Q(0, 1) = 0.5;
        Q(1, 0) = 0.5;  Q(1, 1) = 1.0;             // = [0.5;1][0.5;1]^T, singular

        const Matrix L = Utils::CholeskyFactor(Q);
        CHECK(!L.Empty(), "Cholesky: a semi-definite matrix still factors (jittered)");
        const Matrix LLt = L * L.Transpose();
        CHECK_CLOSE(LLt(0, 0), Q(0, 0), 1e-6, "Cholesky: L L^T reproduces Q (0,0)");
        CHECK_CLOSE(LLt(1, 1), Q(1, 1), 1e-6, "Cholesky: L L^T reproduces Q (1,1)");
    }

    // --- NaN channel masking ---------------------------------------------------
    {
        Vector z(3);
        z(0) = 1.;
        z(1) = std::numeric_limits<double>::quiet_NaN();
        z(2) = 3.;

        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        CHECK(keep.size() == 2, "ValidChannels: drops the NaN channel");
        CHECK(keep[0] == 0 && keep[1] == 2, "ValidChannels: keeps the right indices");
        CHECK(!Utils::IsComplete(z), "IsComplete: false when a channel is missing");

        const Vector zc = Utils::Compress(z, keep);
        CHECK(zc.Size() == 2, "Compress: shrinks the vector");
        CHECK_CLOSE(zc(1), 3., 1e-15, "Compress: keeps the values in order");

        Matrix R(3, 3, 0.);
        for (std::size_t i = 0; i < 3; ++i) R(i, i) = static_cast<double>(i + 1);
        const Matrix Rc = Utils::CompressBoth(R, keep);
        CHECK(Rc.Rows() == 2 && Rc.Cols() == 2, "CompressBoth: shrinks both axes");
        CHECK_CLOSE(Rc(1, 1), 3., 1e-15, "CompressBoth: keeps the right diagonal");
    }

    // --- Numerical Jacobian -----------------------------------------------------
    {
        // f(x) = (x0^2, x0 x1)  =>  J = [[2 x0, 0], [x1, x0]]
        auto fn = [](const Vector &x)
        {
            Vector out(2);
            out(0) = x(0) * x(0);
            out(1) = x(0) * x(1);
            return out;
        };

        Vector x(2); x(0) = 3.; x(1) = -2.;
        const Matrix J = Utils::NumericalJacobian(fn, x);

        CHECK_CLOSE(J(0, 0), 6., 1e-6, "Jacobian: d(x0^2)/dx0 = 2 x0");
        CHECK_CLOSE(J(0, 1), 0., 1e-6, "Jacobian: d(x0^2)/dx1 = 0");
        CHECK_CLOSE(J(1, 0), -2., 1e-6, "Jacobian: d(x0 x1)/dx0 = x1");
        CHECK_CLOSE(J(1, 1), 3., 1e-6, "Jacobian: d(x0 x1)/dx1 = x0");
    }

    // --- Mahalanobis and the Joseph form -----------------------------------------
    {
        Vector y(2); y(0) = 1.; y(1) = 2.;
        Matrix Sinv(2, 2, 0.);
        Sinv(0, 0) = 1.;
        Sinv(1, 1) = 0.25;
        CHECK_CLOSE(Utils::Mahalanobis(y, Sinv), 2., 1e-12, "Mahalanobis: y^T S^-1 y");

        // With K = 0 the Joseph form must return P untouched.
        const Matrix P = Matrix::Identity(2);
        const Matrix H = Matrix::Identity(2);
        const Matrix R = Matrix::Identity(2);
        const Matrix K(2, 2, 0.);
        const Matrix out = Utils::JosephForm(K, H, P, R);
        CHECK_CLOSE(out(0, 0), 1., 1e-12, "JosephForm: zero gain leaves P alone");
        CHECK_CLOSE(out(0, 1), 0., 1e-12, "JosephForm: stays diagonal");
    }

    // --- continuous to discrete -------------------------------------------------
    {
        // A rotation generator: exp([[0, -w], [w, 0]] t) is the rotation by w t.
        const double w = 3., t = 0.7;
        Matrix A(2, 2, 0.);
        A(0, 1) = -w * t;
        A(1, 0) = w * t;
        const Matrix E = Utils::Exponential(A);
        CHECK_CLOSE(E(0, 0), std::cos(w * t), 1e-13, "Exponential: rotation, cosine");
        CHECK_CLOSE(E(1, 0), std::sin(w * t), 1e-13, "Exponential: rotation, sine");

        // A large norm goes through the squarings: exp(diag(-30, 5)).
        Matrix D(2, 2, 0.);
        D(0, 0) = -30.;
        D(1, 1) = 5.;
        const Matrix ED = Utils::Exponential(D);
        CHECK_CLOSE(ED(0, 0) / std::exp(-30.), 1., 1e-10, "Exponential: decaying mode");
        CHECK_CLOSE(ED(1, 1) / std::exp(5.), 1., 1e-12, "Exponential: growing mode");

        // Position and velocity driven by white acceleration of intensity q: the
        // textbook result F = [[1, dt], [0, 1]], Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]].
        const double dt = 0.25, q = 2.;
        Matrix Ac(2, 2, 0.), G(2, 2, 0.), F, Q;
        Ac(0, 1) = 1.;
        G(1, 1) = q;
        Utils::VanLoan(Ac, G, dt, F, Q);
        CHECK_CLOSE(F(0, 1), dt, 1e-14, "VanLoan: F of a constant-velocity model");
        CHECK_CLOSE(Q(0, 0), q * dt * dt * dt / 3., 1e-14, "VanLoan: position variance");
        CHECK_CLOSE(Q(0, 1), q * dt * dt / 2., 1e-14, "VanLoan: cross term");
        CHECK_CLOSE(Q(1, 1), q * dt, 1e-14, "VanLoan: velocity variance");
        CHECK_CLOSE(Q(0, 1), Q(1, 0), 1e-16, "VanLoan: Q is symmetric");
    }

    // --- the algebra and the differentiators, under their own names ------------
    {
        // Solve() by substitution agrees with the inverse, on a covariance-like matrix.
        Matrix S(3, 3, 0.);
        S(0, 0) = 4.; S(0, 1) = 1.; S(0, 2) = 0.5;
        S(1, 0) = 1.; S(1, 1) = 3.; S(1, 2) = 0.2;
        S(2, 0) = 0.5; S(2, 1) = 0.2; S(2, 2) = 2.;
        Vector b(3);
        b(0) = 1.; b(1) = -2.; b(2) = 0.5;
        const Vector x = Algebra::Solve(S, b);
        const Vector back = S * x;
        CHECK_CLOSE(back(0), b(0), 1e-13, "Algebra::Solve: S x = b, first row");
        CHECK_CLOSE(back(2), b(2), 1e-13, "Algebra::Solve: S x = b, last row");
        const Matrix L = Algebra::Cholesky(S);
        CHECK_CLOSE((L * L.Transpose())(1, 2), S(1, 2), 1e-14, "Algebra::Cholesky: L L^T = S");
        CHECK(Algebra::Cholesky(Matrix(2, 2, 0.)).Empty(), "Algebra::Cholesky: a singular matrix has no factor");

        // d/dx of [x0^2 x1, sin(x0)] at (1.3, -0.7): the two schemes against the formula.
        const Map fn = [](const Vector &v) {
            Vector y(2);
            y(0) = v(0) * v(0) * v(1);
            y(1) = std::sin(v(0));
            return y;
        };
        Vector at(2);
        at(0) = 1.3; at(1) = -0.7;
        const double exact = 2. * 1.3 * -0.7;
        const Matrix Jf = Differentiators::Jacobian(DifferentiatorKind::Forward, fn, at);
        const Matrix Jc = Differentiators::Jacobian(DifferentiatorKind::Central, fn, at);
        CHECK_CLOSE(Jf(0, 0), exact, 1e-5, "Forward: first order");
        CHECK_CLOSE(Jc(0, 0), exact, 1e-9, "Central: second order");
        CHECK(std::abs(Jc(0, 0) - exact) < std::abs(Jf(0, 0) - exact), "Central is closer than Forward");
        CHECK_CLOSE(Jc(1, 1), 0., 1e-12, "Central: a variable the map ignores");

        const Vector y0 = fn(at);
        int calls = 0;
        const Map counted = [&](const Vector &v) { ++calls; return fn(v); };
        Differentiators::Forward().Jacobian(counted, at, &y0);
        CHECK(calls == 2, "Forward: one evaluation per component when f(x) is given");

        bool threw = false;
        try { Differentiators::Make(DifferentiatorKind::Automatic); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "Automatic differentiation is reserved, and says so");
        CHECK(Differentiators::Make(DifferentiatorKind::Central)->Kind() == DifferentiatorKind::Central,
              "Make: builds the scheme asked for");
    }

    return Check::Summary("utils");
}
