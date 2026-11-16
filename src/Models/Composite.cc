/**
 * *********************************************
 *
 * \file Composite.cc
 * \brief Source of the Kalman::Models::Composite class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Composite.h"

#include <algorithm>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Kalman::Models
{
    namespace
    {
        /// Copy \p block into \p out with its top-left corner at (\p row, \p col).
        void Place(Matrix &out, const Matrix &block, std::size_t row, std::size_t col)
        {
            for (std::size_t i = 0; i < block.Rows(); ++i)
                for (std::size_t j = 0; j < block.Cols(); ++j) out(row + i, col + j) = block(i, j);
        }

        /// The names of all parts' components, a shared one carrying its part's name in front.
        std::vector<Component> Named(const std::vector<std::vector<Component>> &perPart,
                                     const std::vector<std::string> &parts)
        {
            std::map<std::string, std::set<std::size_t>> owners;
            for (std::size_t p = 0; p < perPart.size(); ++p)
                for (const Component &c : perPart[p]) owners[c.name].insert(p);
            std::vector<Component> out;
            for (std::size_t p = 0; p < perPart.size(); ++p)
            {
                std::set<std::string> mine;
                for (Component c : perPart[p])
                {
                    if (owners[c.name].size() > 1 || !mine.insert(c.name).second) c.name = parts[p] + "_" + c.name;
                    out.push_back(std::move(c));
                }
            }
            return out;
        }
    }

    Composite::Composite(std::vector<std::shared_ptr<Model>> parts, Readout readout, std::vector<std::string> names)
        : fParts(std::move(parts)), fPartNames(std::move(names)), fReadout(readout)
    {
        if (fParts.empty()) throw std::invalid_argument("Models::Composite: give at least one model.");
        for (const auto &part : fParts)
            if (!part) throw std::invalid_argument("Models::Composite: a part is null.");
        if (!fPartNames.empty() && fPartNames.size() != fParts.size())
            throw std::invalid_argument("Models::Composite: " + std::to_string(fParts.size()) + " parts and "
                                        + std::to_string(fPartNames.size()) + " names.");
        const bool summed = fReadout == Readout::Summed;

        if (fPartNames.empty())
        {
            // Called by their model's name; numbered from the second that shares one.
            std::map<std::string, int> seen;
            for (const auto &part : fParts)
            {
                const std::string base = part->Name().empty() ? "part" : part->Name();
                const int k = ++seen[base];
                fPartNames.push_back(k == 1 ? base : base + std::to_string(k));
            }
        }
        {
            std::set<std::string> seen;
            for (const std::string &name : fPartNames)
                if (name.empty() || !seen.insert(name).second)
                    throw std::invalid_argument("Models::Composite: every part needs a name of its own ('" + name + "').");
        }

        std::size_t n = 0, m = 0, l = 0;
        const double fs = fParts.front()->SampleRate();
        for (const auto &part : fParts)
        {
            if (part->SampleRate() != fs)
                throw std::invalid_argument("Models::Composite: the parts must share one sample rate.");
            if (summed && part->NMeasure() != fParts.front()->NMeasure())
                throw std::invalid_argument("Models::Composite: summed parts must have the same number of measurements.");
            fState.push_back(n); fMeasure.push_back(summed ? 0 : m); fControl.push_back(l);
            n += part->NState(); m += part->NMeasure(); l += part->NControl();
        }
        if (summed) m = fParts.front()->NMeasure();
        fState.push_back(n); fMeasure.push_back(m); fControl.push_back(l);

        SetName(fParts.size() == 1 ? fParts.front()->Name() : "composite");
        Allocate(n, m, l);
        SetSampleRate(fs);
        SetT0(fParts.front()->T0());
        SetTimeUnit(fParts.front()->TimeUnit());

        // What the parts are, together: matrices when all are; code as soon as one is.
        StructureKind structure = StructureKind::Matrices;
        EngineKind engine = EngineKind::Linear;
        const Model *differenced = nullptr;
        for (const auto &part : fParts)
        {
            if (part->Structure() == StructureKind::Maps) structure = StructureKind::Maps;
            else if (part->Structure() == StructureKind::TimeVarying && structure == StructureKind::Matrices)
                structure = StructureKind::TimeVarying;
            if (engine == EngineKind::Linear && part->DefaultEngine() != EngineKind::Linear) engine = part->DefaultEngine();
            if (!differenced && !part->HasAnalyticJacobian()) differenced = part.get();
        }
        SetStructure(structure);
        SetDefaultEngine(engine);
        if (differenced) SetDifferentiator(differenced->Differentiator(), differenced->Epsilon());
        else SetAnalyticJacobian(true);

        AdoptInitialState();

        std::vector<std::vector<Component>> states, measurements;
        for (const auto &part : fParts)
        {
            states.push_back(part->StateComponents());
            if (!summed) measurements.push_back(part->MeasurementComponents());
        }
        DeclareState(Named(states, fPartNames));
        DeclareMeasurement(summed ? fParts.front()->MeasurementComponents() : Named(measurements, fPartNames));
        if (summed) SetBuiltInSensorName(fParts.front()->BuiltInSensorName());
    }

    void Composite::AdoptInitialState()
    {
        Vector x0(fNState, 0.);
        Matrix P0(fNState, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const std::size_t n = fParts[p]->NState(), at = fState[p];
            const Vector &s0 = fParts[p]->X0();
            const Matrix &Ps = fParts[p]->P0();
            for (std::size_t i = 0; i < n; ++i)
            {
                x0(at + i) = s0.Size() == n ? s0(i) : 0.;
                for (std::size_t j = 0; j < n; ++j)
                    P0(at + i, at + j) = (Ps.Rows() == n && Ps.Cols() == n) ? Ps(i, j) : (i == j ? 1. : 0.);
            }
        }
        SetInitialState(x0);
        SetInitialCovariance(P0);
    }

    void Composite::Sync() const
    {
        for (const auto &part : fParts)
            if (part->SampleRate() != fSampleRate) part->SetSampleRate(fSampleRate);
    }

    Vector Composite::Cut(const Vector &v, const std::vector<std::size_t> &starts, std::size_t i, const char *what) const
    {
        if (v.Size() != starts.back())
            throw std::invalid_argument(std::string("Models::Composite: the ") + what + " has " + std::to_string(v.Size())
                                        + " components, " + std::to_string(starts.back()) + " expected.");
        const std::size_t from = starts[i], size = starts[i + 1] - from;
        Vector out(size);
        for (std::size_t k = 0; k < size; ++k) out(k) = v(from + k);
        return out;
    }

    // --- maps ---------------------------------------------------------------------------

    Vector Composite::f(const Vector &x, const Vector &u, double t) const
    {
        Sync();
        Vector out(fNState);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const Vector up = (fNControl && u.Size()) ? Cut(u, fControl, p, "control") : Vector();
            const Vector next = fParts[p]->f(Cut(x, fState, p, "state"), up, t);
            for (std::size_t i = 0; i < next.Size(); ++i) out(fState[p] + i) = next(i);
        }
        return out;
    }

    bool Composite::HasProcessNoiseInput() const
    {
        return std::any_of(fParts.begin(), fParts.end(), [](const auto &p) { return p->HasProcessNoiseInput(); });
    }

    Vector Composite::f(const Vector &x, const Vector &u, const Vector &w, double t) const
    {
        Sync();
        Vector out(fNState);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const Vector up = (fNControl && u.Size()) ? Cut(u, fControl, p, "control") : Vector();
            const Vector xp = Cut(x, fState, p, "state"), wp = Cut(w, fState, p, "process noise");
            Vector next;
            if (fParts[p]->HasProcessNoiseInput()) next = fParts[p]->f(xp, up, wp, t);
            else
            {
                next = fParts[p]->f(xp, up, t);                 // a part that takes none: the noise is added
                for (std::size_t i = 0; i < next.Size(); ++i) next(i) += wp(i);
            }
            for (std::size_t i = 0; i < next.Size(); ++i) out(fState[p] + i) = next(i);
        }
        return out;
    }

    Vector Composite::h(const Vector &x, double t) const
    {
        Sync();
        Vector z(fNMeasure, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const Vector zp = fParts[p]->h(Cut(x, fState, p, "state"), t);
            for (std::size_t i = 0; i < zp.Size(); ++i) z(fMeasure[p] + i) += zp(i);    // summed: every part starts at 0
        }
        return z;
    }

    bool Composite::HasMeasurementNoiseInput() const
    {
        if (fReadout == Readout::Summed) return false;          // one channel: its noise is added once
        return std::any_of(fParts.begin(), fParts.end(), [](const auto &p) { return p->HasMeasurementNoiseInput(); });
    }

    Vector Composite::MeasureWithNoise(const Vector &x, const Vector &v, double t) const
    {
        if (v.Size() != fNMeasure)
            throw std::invalid_argument("Models::Composite: the measurement noise has " + std::to_string(v.Size())
                                        + " components, " + std::to_string(fNMeasure) + " expected.");
        Sync();
        if (fReadout == Readout::Summed)
        {
            Vector z = h(x, t);
            for (std::size_t i = 0; i < fNMeasure; ++i) z(i) += v(i);
            return z;
        }
        Vector z(fNMeasure);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const Vector zp = fParts[p]->MeasureWithNoise(Cut(x, fState, p, "state"), Cut(v, fMeasure, p, "noise"), t);
            for (std::size_t i = 0; i < zp.Size(); ++i) z(fMeasure[p] + i) = zp(i);
        }
        return z;
    }

    Vector Composite::LimitControl(const Vector &u) const
    {
        if (fNControl == 0 || u.Size() != fNControl) return u;
        Vector out(fNControl);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            if (fControl[p + 1] == fControl[p]) continue;
            const Vector up = fParts[p]->LimitControl(Cut(u, fControl, p, "control"));
            for (std::size_t i = 0; i < up.Size(); ++i) out(fControl[p] + i) = up(i);
        }
        return out;
    }

    Matrix Composite::Jacobian(const Vector &x, const Vector &u, double t) const
    {
        Sync();
        Matrix out(fNState, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const Vector up = (fNControl && u.Size()) ? Cut(u, fControl, p, "control") : Vector();
            Place(out, fParts[p]->Jacobian(Cut(x, fState, p, "state"), up, t), fState[p], fState[p]);
        }
        return out;
    }

    Matrix Composite::JacobianH(const Vector &x, double t) const
    {
        Sync();
        Matrix out(fNMeasure, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p)
            Place(out, fParts[p]->JacobianH(Cut(x, fState, p, "state"), t), fMeasure[p], fState[p]);
        return out;
    }

    // --- coefficients ---------------------------------------------------------------------

    Matrix Composite::F(double t) const
    {
        Sync();
        Matrix out(fNState, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p) Place(out, fParts[p]->F(t), fState[p], fState[p]);
        return out;
    }

    Matrix Composite::H(double t) const
    {
        Sync();
        Matrix out(fNMeasure, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p) Place(out, fParts[p]->H(t), fMeasure[p], fState[p]);
        return out;
    }

    Matrix Composite::B(double t) const
    {
        if (fNControl == 0) return Matrix();
        Matrix out(fNState, fNControl, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const std::size_t l = fControl[p + 1] - fControl[p], n = fState[p + 1] - fState[p];
            if (l == 0) continue;
            const Matrix Bp = fParts[p]->B(t);              // empty: u is added to the state directly
            if (!Bp.Empty()) Place(out, Bp, fState[p], fControl[p]);
            else
                for (std::size_t i = 0; i < std::min(n, l); ++i) out(fState[p] + i, fControl[p] + i) = 1.;
        }
        return out;
    }

    Matrix Composite::L(double t) const
    {
        std::vector<Matrix> blocks;
        bool any = false;
        for (const auto &part : fParts)
        {
            blocks.push_back(part->L(t));
            any = any || !blocks.back().Empty();
        }
        if (!any) return Matrix();                          // identity coupling throughout
        Matrix out(fNState, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            if (!blocks[p].Empty()) Place(out, blocks[p], fState[p], fState[p]);
            else
                for (std::size_t i = fState[p]; i < fState[p + 1]; ++i) out(i, i) = 1.;
        }
        return out;
    }

    Matrix Composite::Q(double t) const
    {
        Sync();
        Matrix out(fNState, fNState, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p) Place(out, fParts[p]->Q(t), fState[p], fState[p]);
        return out;
    }

    Matrix Composite::R(double t) const
    {
        Sync();
        if (fReadout == Readout::Summed) return fParts.front()->R(t);
        Matrix out(fNMeasure, fNMeasure, 0.);
        for (std::size_t p = 0; p < fParts.size(); ++p) Place(out, fParts[p]->R(t), fMeasure[p], fMeasure[p]);
        return out;
    }

    // --- sensors ----------------------------------------------------------------------------

    std::vector<Sensor> Composite::Sensors() const
    {
        if (fReadout == Readout::Summed) return Model::Sensors();
        std::vector<Sensor> out;
        std::map<std::string, int> seen;
        for (std::size_t p = 0; p < fParts.size(); ++p)
            for (const Sensor &s : fParts[p]->Sensors()) ++seen[s.name];
        std::size_t row = 0;
        const std::vector<Component> channels = MeasurementComponents();
        for (std::size_t p = 0; p < fParts.size(); ++p)
            for (Sensor sensor : fParts[p]->Sensors())
            {
                const std::size_t k = sensor.Channels(), at = fState[p], n = fState[p + 1] - at;
                if (seen[sensor.name] > 1) sensor.name = fPartNames[p] + "_" + sensor.name;
                if (row + k <= channels.size())
                    sensor.channels.assign(channels.begin() + static_cast<std::ptrdiff_t>(row),
                                           channels.begin() + static_cast<std::ptrdiff_t>(row + k));
                row += k;
                if (sensor.IsLinear())
                {
                    Matrix G(k, fNState, 0.);
                    Place(G, sensor.G, 0, at);
                    sensor.G = G;
                }
                else
                {
                    const Sensor::Measure g = sensor.g;
                    sensor.g = [this, g, p](const Vector &x, double t) { return g(Cut(x, fState, p, "state"), t); };
                    if (sensor.jacobian)
                    {
                        const Sensor::MeasureJacobian dg = sensor.jacobian;
                        sensor.jacobian = [this, dg, p, k, at, n](const Vector &x, double t) {
                            Matrix G(k, fNState, 0.);
                            (void)n;
                            Place(G, dg(Cut(x, fState, p, "state"), t), 0, at);
                            return G;
                        };
                    }
                }
                out.push_back(std::move(sensor));
            }
        return out;
    }

    void Composite::CheckDelays() const
    {
        for (const auto &part : fParts) part->CheckDelays();
    }

    // --- parameters ---------------------------------------------------------------------------

    std::pair<std::size_t, std::string> Composite::Resolve(const std::string &name) const
    {
        // The longest part name that fits: a part may itself carry dotted names.
        std::size_t best = fParts.size(), length = 0;
        for (std::size_t p = 0; p < fParts.size(); ++p)
        {
            const std::string &part = fPartNames[p];
            if (name.size() > part.size() + 1 && name.compare(0, part.size(), part) == 0 && name[part.size()] == '.'
                && part.size() >= length && fParts[p]->HasParameter(name.substr(part.size() + 1)))
            {
                best = p;
                length = part.size();
            }
        }
        if (best == fParts.size())
        {
            std::string known;
            for (const std::string &n : ParameterNames()) known += (known.empty() ? "" : ", ") + n;
            throw std::out_of_range("Models::Composite: no parameter '" + name + "'; it has: " + (known.empty() ? "none" : known) + ".");
        }
        return {best, name.substr(length + 1)};
    }

    std::vector<std::string> Composite::ParameterNames() const
    {
        std::vector<std::string> out;
        for (std::size_t p = 0; p < fParts.size(); ++p)
            for (const std::string &name : fParts[p]->ParameterNames()) out.push_back(fPartNames[p] + "." + name);
        std::sort(out.begin(), out.end());
        return out;
    }

    bool Composite::HasParameter(const std::string &name) const
    {
        try { Resolve(name); }
        catch (const std::out_of_range &) { return false; }
        return true;
    }

    double Composite::Parameter(const std::string &name) const
    {
        const auto [part, own] = Resolve(name);
        return fParts[part]->Parameter(own);
    }

    void Composite::SetParameter(const std::string &name, double value)
    {
        const auto [part, own] = Resolve(name);
        fParts[part]->SetParameter(own, value);
    }

    void Composite::CheckShapes() const
    {
        Sync();
        for (const auto &part : fParts) part->CheckShapes();
        Model::CheckShapes();
    }

    std::string Composite::Describe() const
    {
        std::ostringstream os;
        os << Model::Describe() << "; " << fParts.size() << " part" << (fParts.size() == 1 ? "" : "s") << " ("
           << (fReadout == Readout::Summed ? "summed" : "stacked") << "):";
        for (std::size_t p = 0; p < fParts.size(); ++p) os << " [" << fPartNames[p] << "] " << fParts[p]->Describe();
        return os.str();
    }
}
