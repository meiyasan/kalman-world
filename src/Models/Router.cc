/**
 * *********************************************
 *
 * \file Router.cc
 * \brief Source of the Kalman::Models::Router class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Router.h"
#include "Kalman/Routers/Bayesian.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Kalman::Models
{
    Router::Router(std::vector<std::shared_ptr<Model>> modes, const Matrix &Pi, const Vector &mu0,
                   std::vector<std::string> names)
        : fModes(std::move(modes)), fModeNames(std::move(names)), fPi(Pi), fDecision(std::make_shared<Routers::Bayesian>())
    {
        const std::size_t M = fModes.size();
        if (M < 2) throw std::invalid_argument("Models::Router: at least two models are required.");
        for (const auto &mode : fModes)
            if (!mode) throw std::invalid_argument("Models::Router: a model is null.");
        if (Pi.Rows() != M || Pi.Cols() != M)
            throw std::invalid_argument("Models::Router: Pi has shape (" + std::to_string(Pi.Rows()) + ", "
                                        + std::to_string(Pi.Cols()) + "), expected (" + std::to_string(M) + ", "
                                        + std::to_string(M) + ").");
        for (std::size_t i = 0; i < M; ++i)
        {
            double row = 0.;
            for (std::size_t j = 0; j < M; ++j)
            {
                if (Pi(i, j) < 0.) throw std::invalid_argument("Models::Router: Pi has a negative entry.");
                row += Pi(i, j);
            }
            if (std::abs(row - 1.) > 1e-9)
                throw std::invalid_argument("Models::Router: row " + std::to_string(i) + " of Pi sums to "
                                            + std::to_string(row) + ", expected 1.");
        }

        // Mixing averages states across modes, so it only means something when
        // every mode means the same thing by "the state".
        const Model &first = *fModes.front();
        for (const auto &mode : fModes)
        {
            if (mode->NState() != first.NState() || mode->NMeasure() != first.NMeasure()
                || mode->NControl() != first.NControl())
                throw std::invalid_argument("Models::Router: model '" + mode->Name() + "' has (n, m, l) = ("
                                            + std::to_string(mode->NState()) + ", " + std::to_string(mode->NMeasure())
                                            + ", " + std::to_string(mode->NControl()) + ") but '" + first.Name()
                                            + "' has (" + std::to_string(first.NState()) + ", "
                                            + std::to_string(first.NMeasure()) + ", " + std::to_string(first.NControl())
                                            + "). Every mode must share them.");
            if (mode->SampleRate() != first.SampleRate())
                throw std::invalid_argument("Models::Router: the modes must share one sample rate.");
        }

        if (fModeNames.empty())
        {
            std::map<std::string, int> seen;
            for (const auto &mode : fModes)
            {
                const std::string base = mode->Name().empty() ? "mode" : mode->Name();
                const int k = ++seen[base];
                fModeNames.push_back(k == 1 ? base : base + std::to_string(k));
            }
        }
        if (fModeNames.size() != M) throw std::invalid_argument("Models::Router: one name per model.");
        {
            std::set<std::string> seen;
            for (const std::string &name : fModeNames)
                if (name.empty() || !seen.insert(name).second)
                    throw std::invalid_argument("Models::Router: every mode needs a name of its own ('" + name + "').");
        }

        SetName(first.Name());
        Allocate(first.NState(), first.NMeasure(), first.NControl());
        SetSampleRate(first.SampleRate());
        SetT0(first.T0());
        SetTimeUnit(first.TimeUnit());
        SetDefaultEngine(EngineKind::Interacting);
        SetStructure(StructureKind::Maps);
        SetBuiltInSensorName(first.BuiltInSensorName());
        if (first.HasAnalyticJacobian()) SetAnalyticJacobian(true);
        else SetDifferentiator(first.Differentiator(), first.Epsilon());
        DeclareState(first.StateComponents());
        DeclareMeasurement(first.MeasurementComponents());

        SetInitialProbabilities(mu0.Size() ? mu0 : Vector(M, 1. / static_cast<double>(M)));
    }

    void Router::SetInitialProbabilities(const Vector &mu0)
    {
        if (mu0.Size() != fModes.size())
            throw std::invalid_argument("Models::Router: one initial probability per model.");
        fMu0 = Routers::Router::Normalise(mu0);
        std::size_t best = 0;
        for (std::size_t j = 1; j < fMu0.Size(); ++j)
            if (fMu0(j) > fMu0(best)) best = j;
        fLeading = best;
        AdoptInitialState();
    }

    void Router::SetDecision(std::shared_ptr<const Routers::Router> decision)
    {
        if (!decision) throw std::invalid_argument("Models::Router: the decision layer must not be null.");
        fDecision = std::move(decision);
    }

    void Router::SetDecision(const Routers::Router *decision)
    {
        SetDecision(std::shared_ptr<const Routers::Router>(decision, [](const Routers::Router *) {}));
    }

    void Router::SetLeading(std::size_t j) const
    {
        if (j >= fModes.size()) throw std::out_of_range("Models::Router: no such mode.");
        fLeading = j;
    }

    void Router::AdoptInitialState()
    {
        const std::size_t n = fNState;
        Vector x(n, 0.);
        for (std::size_t j = 0; j < fModes.size(); ++j)
            if (fModes[j]->X0().Size() == n) x += fMu0(j) * fModes[j]->X0();
        Matrix P(n, n, 0.);
        for (std::size_t j = 0; j < fModes.size(); ++j)
        {
            const Matrix &Pj = fModes[j]->P0();
            for (std::size_t a = 0; a < n; ++a)
                for (std::size_t b = 0; b < n; ++b)
                {
                    const double da = fModes[j]->X0().Size() == n ? fModes[j]->X0()(a) - x(a) : 0.;
                    const double db = fModes[j]->X0().Size() == n ? fModes[j]->X0()(b) - x(b) : 0.;
                    const double p = (Pj.Rows() == n && Pj.Cols() == n) ? Pj(a, b) : (a == b ? 1. : 0.);
                    P(a, b) += fMu0(j) * (p + da * db);
                }
        }
        Utils::Symmetrize(P);
        SetInitialState(x);
        SetInitialCovariance(P);
    }

    void Router::Sync() const
    {
        for (const auto &mode : fModes)
            if (mode->SampleRate() != fSampleRate) mode->SetSampleRate(fSampleRate);
    }

    void Router::CheckDelays() const
    {
        for (const auto &mode : fModes) mode->CheckDelays();
    }

    // --- parameters ---------------------------------------------------------------------------

    std::pair<std::size_t, std::string> Router::Resolve(const std::string &name) const
    {
        std::size_t best = fModes.size(), length = 0;
        for (std::size_t j = 0; j < fModes.size(); ++j)
        {
            const std::string &mode = fModeNames[j];
            if (name.size() > mode.size() + 1 && name.compare(0, mode.size(), mode) == 0 && name[mode.size()] == '.'
                && mode.size() >= length && fModes[j]->HasParameter(name.substr(mode.size() + 1)))
            {
                best = j;
                length = mode.size();
            }
        }
        if (best == fModes.size())
        {
            std::string known;
            for (const std::string &n : ParameterNames()) known += (known.empty() ? "" : ", ") + n;
            throw std::out_of_range("Models::Router: no parameter '" + name + "'; it has: " + (known.empty() ? "none" : known) + ".");
        }
        return {best, name.substr(length + 1)};
    }

    std::vector<std::string> Router::ParameterNames() const
    {
        std::vector<std::string> out;
        for (std::size_t j = 0; j < fModes.size(); ++j)
            for (const std::string &name : fModes[j]->ParameterNames()) out.push_back(fModeNames[j] + "." + name);
        std::sort(out.begin(), out.end());
        return out;
    }

    bool Router::HasParameter(const std::string &name) const
    {
        try { Resolve(name); }
        catch (const std::out_of_range &) { return false; }
        return true;
    }

    double Router::Parameter(const std::string &name) const
    {
        const auto [mode, own] = Resolve(name);
        return fModes[mode]->Parameter(own);
    }

    void Router::SetParameter(const std::string &name, double value)
    {
        const auto [mode, own] = Resolve(name);
        fModes[mode]->SetParameter(own, value);
    }

    void Router::CheckShapes() const
    {
        Sync();
        for (const auto &mode : fModes) mode->CheckShapes();
        Model::CheckShapes();
    }

    std::string Router::Describe() const
    {
        std::ostringstream os;
        os << Model::Describe() << "; " << fModes.size() << " modes:";
        for (std::size_t j = 0; j < fModes.size(); ++j) os << " [" << fModeNames[j] << "] " << fModes[j]->Describe();
        return os.str();
    }
}
