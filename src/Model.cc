/**
 * *********************************************
 *
 * \file Model.cc
 * \brief Implementation of the Kalman::Model class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Model.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace Kalman
{
    namespace
    {
        /// Raise a shape error that names the offending matrix and both shapes.
        void ShapeError(const char *what, std::size_t gotR, std::size_t gotC,
                        std::size_t wantR, std::size_t wantC)
        {
            std::ostringstream os;
            os << "Kalman::Model: " << what << " has shape (" << gotR << ", " << gotC
               << "), expected (" << wantR << ", " << wantC << ").";
            throw std::runtime_error(os.str());
        }
    }

    Model::Model(std::string name, std::size_t n, std::size_t m, std::size_t l)
        : fName(std::move(name))
    {
        if (n == 0) throw std::runtime_error("Kalman::Model: the state dimension n must be positive.");
        if (m == 0) throw std::runtime_error("Kalman::Model: the measurement dimension m must be positive.");
        Allocate(n, m, l);
    }

    void Model::Allocate(std::size_t n, std::size_t m, std::size_t l)
    {
        fNState = n;
        fNMeasure = m;
        fNControl = l;

        // Editable placeholders, the same defaults as the Python matrix-construction
        // path: F = identity, H observes the leading min(m, n) states, Q small,
        // R unit, x0 = 0, P0 = identity.
        fF = Matrix::Identity(n);

        fH = Matrix(m, n, 0.);
        for (std::size_t i = 0; i < std::min(m, n); ++i) fH(i, i) = 1.;

        fQ = Matrix::Identity(n);
        fQ *= 1e-3;
        fR = Matrix::Identity(m);

        fB = (l > 0) ? Matrix(n, l, 0.) : Matrix();
        fL = Matrix();                            // empty => identity coupling

        fX0 = Vector(n, 0.);
        fP0 = Matrix::Identity(n);
    }

    void Model::SetSampleRate(double fs)
    {
        if (fs <= 0.) throw std::runtime_error("Kalman::Model: the sample rate must be positive.");
        fSampleRate = fs;
    }

    // --- coefficients --------------------------------------------------------

    Matrix Model::F(double) const { return fF; }
    Matrix Model::H(double) const { return fH; }
    Matrix Model::B(double) const { return fB; }
    Matrix Model::L(double) const { return fL; }
    Matrix Model::Q(double) const { return fQ; }
    Matrix Model::R(double) const { return fR; }

    // --- maps ----------------------------------------------------------------

    Vector Model::f(const Vector &x, const Vector &u, double t) const
    {
        Vector out = F(t) * x;
        if (fNControl > 0 && u.Size() == fNControl)
        {
            const Matrix Bm = B(t);
            // An empty B means "the control enters the state directly", which only
            // makes sense when l == n; otherwise it must be mapped.
            if (Bm.Empty()) out += u;
            else            out += Bm * u;
        }
        return out;
    }

    Vector Model::h(const Vector &x, double t) const
    {
        return H(t) * x;
    }

    Matrix Model::Jacobian(const Vector &x, const Vector &u, double t) const
    {
        if (fAnalyticJacobian) return F(t);

        const Vector uu(u);
        return Utils::NumericalJacobian(
            [this, &uu, t](const Vector &xx) { return this->f(xx, uu, t); }, x, fEpsilon);
    }

    Matrix Model::JacobianH(const Vector &x, double t) const
    {
        if (fAnalyticJacobian) return H(t);

        return Utils::NumericalJacobian(
            [this, t](const Vector &xx) { return this->h(xx, t); }, x, fEpsilon);
    }

    // --- setters -------------------------------------------------------------

    void Model::SetF(const Matrix &F)
    {
        if (F.Rows() != fNState || F.Cols() != fNState)
            ShapeError("F", F.Rows(), F.Cols(), fNState, fNState);
        fF = F;
    }

    void Model::SetH(const Matrix &H)
    {
        if (H.Rows() != fNMeasure || H.Cols() != fNState)
            ShapeError("H", H.Rows(), H.Cols(), fNMeasure, fNState);
        fH = H;
    }

    void Model::SetB(const Matrix &B)
    {
        if (B.Rows() != fNState || B.Cols() != fNControl)
            ShapeError("B", B.Rows(), B.Cols(), fNState, fNControl);
        fB = B;
    }

    void Model::SetL(const Matrix &L)
    {
        if (!L.Empty() && (L.Rows() != fNState || L.Cols() != fNState))
            ShapeError("L", L.Rows(), L.Cols(), fNState, fNState);
        fL = L;
    }

    void Model::SetQ(const Matrix &Q)
    {
        if (Q.Rows() != fNState || Q.Cols() != fNState)
            ShapeError("Q", Q.Rows(), Q.Cols(), fNState, fNState);
        fQ = Q;
    }

    void Model::SetR(const Matrix &R)
    {
        if (R.Rows() != fNMeasure || R.Cols() != fNMeasure)
            ShapeError("R", R.Rows(), R.Cols(), fNMeasure, fNMeasure);
        fR = R;
    }

    void Model::SetInitialState(const Vector &x0)
    {
        if (x0.Size() != fNState) ShapeError("x0", x0.Size(), 1, fNState, 1);
        fX0 = x0;
    }

    void Model::SetInitialCovariance(const Matrix &P0)
    {
        if (P0.Rows() != fNState || P0.Cols() != fNState)
            ShapeError("P0", P0.Rows(), P0.Cols(), fNState, fNState);
        fP0 = P0;
    }

    // --- validation ----------------------------------------------------------

    void Model::CheckShapes() const
    {
        if (fNState == 0 || fNMeasure == 0)
            throw std::runtime_error("Kalman::Model: the model was never sized (n, m must be > 0).");

        const double t = fT0;

        const Matrix Fm = F(t);
        if (Fm.Rows() != fNState || Fm.Cols() != fNState)
            ShapeError("F(t)", Fm.Rows(), Fm.Cols(), fNState, fNState);

        const Matrix Hm = H(t);
        if (Hm.Rows() != fNMeasure || Hm.Cols() != fNState)
            ShapeError("H(t)", Hm.Rows(), Hm.Cols(), fNMeasure, fNState);

        const Matrix Qm = Q(t);
        if (Qm.Rows() != fNState || Qm.Cols() != fNState)
            ShapeError("Q(t)", Qm.Rows(), Qm.Cols(), fNState, fNState);

        const Matrix Rm = R(t);
        if (Rm.Rows() != fNMeasure || Rm.Cols() != fNMeasure)
            ShapeError("R(t)", Rm.Rows(), Rm.Cols(), fNMeasure, fNMeasure);

        const Matrix Lm = L(t);
        if (!Lm.Empty() && (Lm.Rows() != fNState || Lm.Cols() != fNState))
            ShapeError("L(t)", Lm.Rows(), Lm.Cols(), fNState, fNState);

        if (fNControl > 0)
        {
            const Matrix Bm = B(t);
            if (!Bm.Empty() && (Bm.Rows() != fNState || Bm.Cols() != fNControl))
                ShapeError("B(t)", Bm.Rows(), Bm.Cols(), fNState, fNControl);
        }

        if (fX0.Size() != fNState) ShapeError("x0", fX0.Size(), 1, fNState, 1);
        if (fP0.Rows() != fNState || fP0.Cols() != fNState)
            ShapeError("P0", fP0.Rows(), fP0.Cols(), fNState, fNState);

        // The maps must agree with the declared dimensions too: a subclass that
        // overrides f()/h() can easily return the wrong length, and that is far
        // cheaper to catch here than mid-run.
        const Vector probe(fNState, 0.);
        const Vector u(fNControl, 0.);

        const Vector fx = f(probe, u, t);
        if (fx.Size() != fNState) ShapeError("f(x, u, t)", fx.Size(), 1, fNState, 1);

        const Vector hx = h(probe, t);
        if (hx.Size() != fNMeasure) ShapeError("h(x, t)", hx.Size(), 1, fNMeasure, 1);
    }

    std::string Model::Describe() const
    {
        std::ostringstream os;
        os << fName << " — n = " << fNState << ", m = " << fNMeasure
           << ", l = " << fNControl << ", fs = " << fSampleRate << " Hz, engine = "
           << ToString(fDefaultEngine) << ", Jacobian = "
           << (fAnalyticJacobian ? "analytic" : "numerical");
        return os.str();
    }
}
