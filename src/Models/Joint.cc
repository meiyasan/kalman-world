/**
 * *********************************************
 *
 * \file Joint.cc
 * \brief Source of the Kalman::Models::Joint class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Joint.h"
#include "Kalman/Differentiator.h"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Kalman::Models
{
    namespace
    {
        std::size_t CheckedCount(const Model &inner, const std::vector<Joint::Parameter> &parameters)
        {
            if (parameters.empty())
                throw std::invalid_argument("Models::Joint: name at least one parameter to estimate.");
            std::set<std::string> seen;
            for (const auto &p : parameters)
            {
                if (!inner.HasParameter(p.name))
                    inner.Parameter(p.name);              // throws, naming the known ones
                if (!seen.insert(p.name).second)
                    throw std::invalid_argument("Models::Joint: the parameter '" + p.name + "' is given twice.");
                if (!(p.sigma > 0.))
                    throw std::invalid_argument("Models::Joint: the parameter '" + p.name
                                                + "' needs an initial standard deviation > 0.");
                if (!(p.drift >= 0.))
                    throw std::invalid_argument("Models::Joint: the drift of '" + p.name + "' must be >= 0.");
                if (!(p.tau >= 0.))
                    throw std::invalid_argument("Models::Joint: the tau of '" + p.name + "' must be >= 0.");
            }
            return parameters.size();
        }
    }

    Joint::Joint(std::shared_ptr<Model> inner, std::vector<Parameter> parameters)
        : fInner(std::move(inner)), fEstimated(std::move(parameters))
    {
        if (!fInner) throw std::invalid_argument("Models::Joint: the inner model is null.");
        const std::size_t n = fInner->NState(), p = CheckedCount(*fInner, fEstimated);
        fNInner = n;
        SetName(fInner->Name());
        Allocate(n + p, fInner->NMeasure(), fInner->NControl());

        SetSampleRate(fInner->SampleRate());
        SetT0(fInner->T0());
        SetTimeUnit(fInner->TimeUnit());

        // The parameters make the step nonlinear in the state even when the inner
        // model is linear: the linear engine would read F(t), which is not dJ/dx.
        const EngineKind engine = fInner->DefaultEngine();
        SetDefaultEngine(engine == EngineKind::Linear ? EngineKind::Extended : engine);
        SetAnalyticJacobian(false);
        SetStructure(StructureKind::Maps);

        AdoptInitialState();

        std::vector<Component> components = fInner->StateComponents();
        for (const auto &e : fEstimated) components.push_back({e.name, ""});
        DeclareState(std::move(components));
        DeclareMeasurement(fInner->MeasurementComponents());

        CheckActs();
    }

    void Joint::AdoptInitialState()
    {
        // The inner model's initial state, then the parameters as they are now.
        const std::size_t n = fNInner, p = fEstimated.size();
        const Vector &s0 = fInner->X0();
        const Matrix &P0 = fInner->P0();
        Vector x0(n + p, 0.);
        Matrix Px(n + p, n + p, 0.);
        for (std::size_t i = 0; i < n; ++i)
        {
            x0(i) = s0.Size() == n ? s0(i) : 0.;
            for (std::size_t j = 0; j < n; ++j)
                Px(i, j) = (P0.Rows() == n && P0.Cols() == n) ? P0(i, j) : (i == j ? 1. : 0.);
        }
        fMean.assign(p, 0.);
        for (std::size_t k = 0; k < p; ++k)
        {
            x0(n + k) = fInner->Parameter(fEstimated[k].name);
            fMean[k] = x0(n + k);
            Px(n + k, n + k) = fEstimated[k].sigma * fEstimated[k].sigma;
        }
        SetInitialState(x0);
        SetInitialCovariance(Px);
    }

    void Joint::CheckActs() const
    {
        Sync();
        const Vector s0 = Head(fX0);
        Vector shifted(s0);
        for (std::size_t i = 0; i < fNInner; ++i) shifted(i) += std::max(1., std::fabs(s0(i)));
        const Vector u(fNControl, 0.);
        const double t = fT0;

        for (std::size_t k = 0; k < fEstimated.size(); ++k)
        {
            const std::string &name = fEstimated[k].name;
            const double value = fInner->Parameter(name);
            const double step = 1e-3 * std::max(1., std::fabs(value));
            bool acts = false;
            for (const Vector *s : std::initializer_list<const Vector *>{&s0, &shifted})
            {
                Vector f0, h0, f1, h1;
                try
                {
                    f0 = fInner->f(*s, u, t); h0 = fInner->h(*s, t);
                    fInner->SetParameter(name, value + step);
                    f1 = fInner->f(*s, u, t); h1 = fInner->h(*s, t);
                }
                catch (...) { fInner->SetParameter(name, value); throw; }
                fInner->SetParameter(name, value);
                for (std::size_t i = 0; i < f0.Size() && !acts; ++i) acts = f0(i) != f1(i);
                for (std::size_t i = 0; i < h0.Size() && !acts; ++i) acts = h0(i) != h1(i);
                if (acts) break;
            }
            if (!acts)
                throw std::invalid_argument("Models::Joint: changing the parameter '" + name + "' of "
                                            + fInner->Name() + " changes neither f nor h. The model may read it "
                                            "only when it is built; the filter could not estimate it.");
        }
    }

    // --- helpers -------------------------------------------------------------

    void Joint::Sync() const
    {
        if (fInner->SampleRate() != fSampleRate) fInner->SetSampleRate(fSampleRate);
    }

    Vector Joint::Head(const Vector &x) const
    {
        if (x.Size() != fNState)
            throw std::invalid_argument("Models::Joint: the state has " + std::to_string(x.Size())
                                        + " components, " + std::to_string(fNState) + " expected.");
        Vector s(fNInner);
        for (std::size_t i = 0; i < fNInner; ++i) s(i) = x(i);
        return s;
    }

    std::vector<double> Joint::Apply(const Vector &x) const
    {
        std::vector<double> previous(fEstimated.size());
        for (std::size_t k = 0; k < fEstimated.size(); ++k)
        {
            previous[k] = fInner->Parameter(fEstimated[k].name);
            fInner->SetParameter(fEstimated[k].name, x(fNInner + k));
        }
        return previous;
    }

    void Joint::Restore(const std::vector<double> &previous) const
    {
        for (std::size_t k = 0; k < fEstimated.size(); ++k)
            fInner->SetParameter(fEstimated[k].name, previous[k]);
    }

    double Joint::Decay(std::size_t k) const
    {
        const double tau = fEstimated[k].tau;
        return tau > 0. ? std::exp(-DeltaT() / tau) : 1.;
    }

    DifferentiatorKind Joint::ParameterScheme() const
    {
        const DifferentiatorKind kind = fInner->Differentiator();
        // An analytic inner model has a formula for d/ds, not for d/dtheta.
        return (kind == DifferentiatorKind::Forward || kind == DifferentiatorKind::Central)
                   ? kind : DifferentiatorKind::Forward;
    }

    double Joint::ParameterStep() const
    {
        const DifferentiatorKind kind = fInner->Differentiator();
        return (kind == DifferentiatorKind::Forward || kind == DifferentiatorKind::Central)
                   ? fInner->Epsilon() : Differentiators::DefaultStep(DifferentiatorKind::Forward);
    }

    // --- maps ------------------------------------------------------------------

    Vector Joint::f(const Vector &x, const Vector &u, double t) const
    {
        Sync();
        const Vector s = Head(x);
        fLastX = x;
        const auto previous = Apply(x);
        Vector next;
        try { next = fInner->f(s, u, t); }
        catch (...) { Restore(previous); throw; }
        Restore(previous);

        Vector out(x);                                 // the parameters stay, or return towards their mean
        for (std::size_t i = 0; i < fNInner; ++i) out(i) = next(i);
        for (std::size_t k = 0; k < fEstimated.size(); ++k)
            out(fNInner + k) = fMean[k] + Decay(k) * (x(fNInner + k) - fMean[k]);
        return out;
    }

    Vector Joint::f(const Vector &x, const Vector &u, const Vector &w, double t) const
    {
        if (w.Size() != fNState)
            throw std::invalid_argument("Models::Joint: the process noise has " + std::to_string(w.Size())
                                        + " components, " + std::to_string(fNState) + " expected.");
        Sync();
        const Vector s = Head(x), ws = Head(w);
        fLastX = x;
        const auto previous = Apply(x);
        Vector next;
        try { next = fInner->f(s, u, ws, t); }
        catch (...) { Restore(previous); throw; }
        Restore(previous);

        Vector out(x);
        for (std::size_t i = 0; i < fNInner; ++i) out(i) = next(i);
        for (std::size_t k = 0; k < fEstimated.size(); ++k)
            out(fNInner + k) = fMean[k] + Decay(k) * (x(fNInner + k) - fMean[k]) + w(fNInner + k);
        return out;
    }

    Vector Joint::h(const Vector &x, double t) const
    {
        Sync();
        const Vector s = Head(x);
        const auto previous = Apply(x);
        Vector z;
        try { z = fInner->h(s, t); }
        catch (...) { Restore(previous); throw; }
        Restore(previous);
        return z;
    }

    std::vector<Sensor> Joint::Sensors() const
    {
        Sync();
        std::vector<Sensor> out = fInner->Sensors();
        std::size_t first = 0;
        for (Sensor &sensor : out)
        {
            const std::size_t k = sensor.Channels(), row = first;
            first += k;
            sensor.G = Matrix();
            sensor.calibrated = false;
            sensor.g = [this, row, k](const Vector &x, double t) {
                const Vector z = this->h(x, t);
                Vector part(k);
                for (std::size_t i = 0; i < k; ++i) part(i) = z(row + i);
                return part;
            };
            sensor.jacobian = [this, row, k](const Vector &x, double t) {
                const Matrix H = this->JacobianH(x, t);
                Matrix part(k, fNState, 0.);
                for (std::size_t i = 0; i < k; ++i)
                    for (std::size_t j = 0; j < fNState; ++j) part(i, j) = H(row + i, j);
                return part;
            };
        }
        return out;
    }

    Vector Joint::MeasureWithNoise(const Vector &x, const Vector &v, double t) const
    {
        Sync();
        const Vector s = Head(x);
        const auto previous = Apply(x);
        Vector z;
        try { z = fInner->MeasureWithNoise(s, v, t); }
        catch (...) { Restore(previous); throw; }
        Restore(previous);
        return z;
    }

    Matrix Joint::Jacobian(const Vector &x, const Vector &u, double t) const
    {
        Sync();
        const std::size_t n = fNInner, p = fEstimated.size();
        const Vector s = Head(x);
        Vector theta(p);
        for (std::size_t k = 0; k < p; ++k) theta(k) = x(n + k);

        const auto previous = Apply(x);
        Matrix Js, Jt;
        try
        {
            Js = fInner->Jacobian(s, u, t);
            const Vector uu(u);
            Jt = Differentiators::Jacobian(
                ParameterScheme(),
                [this, &s, &uu, t](const Vector &th) {
                    for (std::size_t k = 0; k < th.Size(); ++k) fInner->SetParameter(fEstimated[k].name, th(k));
                    return fInner->f(s, uu, t);
                },
                theta, ParameterStep());
        }
        catch (...) { Restore(previous); throw; }
        Restore(previous);

        Matrix J(n + p, n + p, 0.);
        for (std::size_t i = 0; i < n; ++i)
        {
            for (std::size_t j = 0; j < n; ++j) J(i, j) = Js(i, j);
            for (std::size_t k = 0; k < p; ++k) J(i, n + k) = Jt(i, k);
        }
        for (std::size_t k = 0; k < p; ++k) J(n + k, n + k) = Decay(k);
        return J;
    }

    Matrix Joint::JacobianH(const Vector &x, double t) const
    {
        Sync();
        const std::size_t n = fNInner, p = fEstimated.size(), m = fNMeasure;
        const Vector s = Head(x);
        Vector theta(p);
        for (std::size_t k = 0; k < p; ++k) theta(k) = x(n + k);

        const auto previous = Apply(x);
        Matrix Hs, Ht;
        try
        {
            Hs = fInner->JacobianH(s, t);
            Ht = Differentiators::Jacobian(
                ParameterScheme(),
                [this, &s, t](const Vector &th) {
                    for (std::size_t k = 0; k < th.Size(); ++k) fInner->SetParameter(fEstimated[k].name, th(k));
                    return fInner->h(s, t);
                },
                theta, ParameterStep());
        }
        catch (...) { Restore(previous); throw; }
        Restore(previous);

        Matrix H(m, n + p, 0.);
        for (std::size_t i = 0; i < m; ++i)
        {
            for (std::size_t j = 0; j < n; ++j) H(i, j) = Hs(i, j);
            for (std::size_t k = 0; k < p; ++k) H(i, n + k) = Ht(i, k);
        }
        return H;
    }

    // --- coefficients ------------------------------------------------------------

    Matrix Joint::B(double t) const
    {
        if (fNControl == 0) return Matrix();
        const Matrix Bs = fInner->B(t);                // empty: u is added to the state directly
        Matrix out(fNState, fNControl, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNControl; ++j)
                out(i, j) = Bs.Empty() ? (i == j ? 1. : 0.) : Bs(i, j);
        return out;
    }

    Matrix Joint::L(double t) const
    {
        const Matrix Ls = fInner->L(t);
        if (Ls.Empty()) return Matrix();               // identity coupling throughout
        Matrix out(fNState, fNState, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, j) = Ls(i, j);
        for (std::size_t i = fNInner; i < fNState; ++i) out(i, i) = 1.;
        return out;
    }

    Matrix Joint::Q(double t) const
    {
        Sync();
        Matrix Qs;
        if (fLastX.Size() == fNState)
        {
            const auto previous = Apply(fLastX);
            try { Qs = fInner->Q(t); }
            catch (...) { Restore(previous); throw; }
            Restore(previous);
        }
        else Qs = fInner->Q(t);
        Matrix out(fNState, fNState, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, j) = Qs(i, j);
        for (std::size_t k = 0; k < fEstimated.size(); ++k)
            out(fNInner + k, fNInner + k) = fEstimated[k].drift * fEstimated[k].drift;
        return out;
    }

    Matrix Joint::R(double t) const
    {
        Sync();
        return fInner->R(t);
    }

    void Joint::CheckShapes() const
    {
        Sync();
        fInner->CheckShapes();
        Model::CheckShapes();
    }

    std::string Joint::Describe() const
    {
        std::ostringstream os;
        os << Model::Describe() << "; estimates";
        for (const auto &e : fEstimated) os << ' ' << e.name;
        os << " of " << fInner->Describe();
        return os.str();
    }
}
