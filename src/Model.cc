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
        Decode();
    }

    // --- coefficients --------------------------------------------------------

    Matrix Model::F(double) const { return fF; }
    Matrix Model::H(double) const { return fH; }
    Matrix Model::B(double) const { return fB; }
    Matrix Model::L(double) const { return fL; }
    Matrix Model::Q(double) const { return fQ; }
    Matrix Model::R(double) const { return fR; }

    // --- maps ----------------------------------------------------------------

    Vector Model::f(const Vector &x, const Vector &u, const Vector &w, double t) const
    {
        Vector out = f(x, u, t);
        if (w.Size() == out.Size()) out += w;
        return out;
    }

    Vector Model::MeasureWithNoise(const Vector &x, const Vector &v, double t) const
    {
        Vector out = h(x, t);
        if (v.Size() == out.Size()) out += v;
        return out;
    }

    Vector Model::f(const Vector &x, const Vector &u, double t) const
    {
        if (fContinuous)
        {
            // The ODE path: the integrator advances Derivative() across one sample,
            // in Substeps() pieces, holding the control across the step.
            const double dt = DeltaT() / static_cast<double>(fSubsteps);
            return GetIntegrator().Integrate(
                [this](const Vector &xx, const Vector &uu, double tt) { return this->Derivative(xx, uu, tt); },
                x, u, t, dt, fSubsteps);
        }

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

    Vector Model::Derivative(const Vector &, const Vector &, double) const
    {
        throw std::runtime_error("Kalman::Model: '" + fName + "' declares continuous dynamics "
                                 "but does not override Derivative(x, u, t).");
    }

    // --- time stepping -------------------------------------------------------

    void Model::UseContinuousDynamics(bool on)
    {
        fContinuous = on;
        if (on) fAnalyticJacobian = false;
    }

    void Model::SetGenerator(const Matrix &A)
    {
        if (A.Rows() != fNState || A.Cols() != fNState)
            ShapeError("A", A.Rows(), A.Cols(), fNState, fNState);
        fGenerator = A;
        Decode();
    }

    void Model::SetNoiseIntensity(const Matrix &Qc, const Matrix &Lc)
    {
        const std::size_t q = Qc.Rows();
        if (q == 0 || Qc.Cols() != q) throw std::runtime_error("Kalman::Model: the noise intensity must be square.");
        if (Lc.Empty() ? q != fNState : (Lc.Rows() != fNState || Lc.Cols() != q))
            throw std::runtime_error("Kalman::Model: the noise coupling must be (n, q).");
        fNoiseIntensity = Qc;
        fNoiseCoupling = Lc;
        Decode();
    }

    // What a continuous model stated, turned into one sample: F from the
    // generator, Q from the noise intensity.
    void Model::Decode()
    {
        if (fGenerator.Empty() && fNoiseIntensity.Empty()) return;
        const double dt = DeltaT();

        Matrix A = fGenerator;
        if (A.Empty())
        {
            // No generator: linearise the derivative at the initial state. The
            // leading term of Q does not depend on A at all, so this only
            // shapes the cross terms.
            if (fContinuous)
                A = Utils::NumericalJacobian(
                    [this](const Vector &x) { return this->Derivative(x, Vector(fNControl, 0.), fT0); }, fX0, fEpsilon);
            else
                A = Matrix(fNState, fNState, 0.);
        }

        Matrix G(fNState, fNState, 0.);
        if (!fNoiseIntensity.Empty())
            G = fNoiseCoupling.Empty() ? fNoiseIntensity
                                       : fNoiseCoupling * fNoiseIntensity * fNoiseCoupling.Transpose();

        Matrix F, Q;
        Utils::VanLoan(A, G, dt, F, Q);
        if (!fGenerator.Empty()) fF = F;
        if (!fNoiseIntensity.Empty())
        {
            fQ = Q;
            fL = Matrix();                       // the coupling is inside Q now
        }
    }

    void Model::SetIntegrator(IntegratorKind kind)
    {
        fDefaultIntegrator = kind;
        fIntegrator.reset();                    // rebuilt from the new kind on first use
    }

    void Model::SetIntegrator(std::shared_ptr<Integrator> integrator)
    {
        if (!integrator) throw std::runtime_error("Kalman::Model: the integrator must not be null.");
        fDefaultIntegrator = integrator->Kind();
        fIntegrator = std::move(integrator);
    }

    Integrator &Model::GetIntegrator() const
    {
        if (!fIntegrator) fIntegrator = Integrators::Make(fDefaultIntegrator);
        return *fIntegrator;
    }

    void Model::SetSubsteps(std::size_t n)
    {
        if (n == 0) throw std::runtime_error("Kalman::Model: at least one integrator substep.");
        fSubsteps = n;
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
        if (fGenerator.Empty() && !fNoiseIntensity.Empty()) Decode();   // linearised there
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

    void Model::DeclareParameter(const std::string &name, double value)
    {
        if (name.empty()) throw std::invalid_argument("Kalman::Model: a parameter needs a name.");
        fParameters[name] = value;
    }

    std::vector<std::string> Model::ParameterNames() const
    {
        std::vector<std::string> names;
        names.reserve(fParameters.size());
        for (const auto &entry : fParameters) names.push_back(entry.first);   // std::map: sorted
        return names;
    }

    namespace
    {
        [[noreturn]] void UnknownParameter(const Model &model, const std::string &name)
        {
            std::ostringstream os;
            os << "Kalman::Model '" << model.Name() << "' has no parameter '" << name << "'; it declares";
            const std::vector<std::string> names = model.ParameterNames();
            if (names.empty()) os << " none";
            for (std::size_t i = 0; i < names.size(); ++i) os << (i ? ", " : " ") << names[i];
            os << '.';
            throw std::out_of_range(os.str());
        }
    }

    double Model::Parameter(const std::string &name) const
    {
        const auto it = fParameters.find(name);
        if (it == fParameters.end()) UnknownParameter(*this, name);
        return it->second;
    }

    void Model::SetParameter(const std::string &name, double value)
    {
        const auto it = fParameters.find(name);
        if (it == fParameters.end()) UnknownParameter(*this, name);
        it->second = value;
        ParametersChanged();
    }

    std::string Model::Describe() const
    {
        std::ostringstream os;
        os << fName << " — n = " << fNState << ", m = " << fNMeasure
           << ", l = " << fNControl << ", fs = " << fSampleRate << " Hz, engine = "
           << ToString(fDefaultEngine) << ", Jacobian = "
           << (fAnalyticJacobian ? "analytic" : "numerical");
        if (fContinuous)
            os << ", integrator = " << ToString(fDefaultIntegrator)
               << (fSubsteps > 1 ? " x" + std::to_string(fSubsteps) : "");
        return os.str();
    }
}
