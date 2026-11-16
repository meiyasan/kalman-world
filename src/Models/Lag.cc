/**
 * *********************************************
 *
 * \file Lag.cc
 * \brief Source of the Kalman::Models::Lag class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Lag.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Kalman::Models
{
    Lag::Lag(std::shared_ptr<Model> inner, std::size_t depth) : fInner(std::move(inner)), fDepth(depth)
    {
        if (!fInner) throw std::invalid_argument("Models::Lag: the inner model is null.");
        if (fDepth == 0) throw std::invalid_argument("Models::Lag: carry at least one past state (depth > 0).");
        fNInner = fInner->NState();
        SetName(fInner->Name());
        Allocate(fNInner * (fDepth + 1), fInner->NMeasure(), fInner->NControl());

        SetSampleRate(fInner->SampleRate());
        SetT0(fInner->T0());
        SetTimeUnit(fInner->TimeUnit());
        SetDefaultEngine(fInner->DefaultEngine());
        SetBuiltInSensorName(fInner->BuiltInSensorName());
        // Still what the inner model is: its matrices in a larger frame, or code.
        if (fInner->HasAnalyticJacobian()) SetAnalyticJacobian(true);
        else SetDifferentiator(fInner->Differentiator(), fInner->Epsilon());
        SetStructure(fInner->Structure());

        AdoptInitialState();

        // The present state keeps its names; the state j steps ago carries "@-j".
        const std::vector<Component> inner_components = fInner->StateComponents();
        std::vector<Component> components;
        for (std::size_t j = 0; j <= fDepth; ++j)
            for (std::size_t i = 0; i < fNInner; ++i)
            {
                Component c = i < inner_components.size() ? inner_components[i] : Component{"x" + std::to_string(i), ""};
                if (j > 0) c.name += "@-" + std::to_string(j);
                components.push_back(std::move(c));
            }
        DeclareState(std::move(components));
        DeclareMeasurement(fInner->MeasurementComponents());

        Sync();
    }

    void Lag::AdoptInitialState()
    {
        const Vector &s0 = fInner->X0();
        const Matrix &P0 = fInner->P0();
        const std::size_t n = fNInner;
        Vector x0(fNState, 0.);
        Matrix Px(fNState, fNState, 0.);
        for (std::size_t j = 0; j <= fDepth; ++j)
            for (std::size_t i = 0; i < n; ++i)
            {
                x0(j * n + i) = s0.Size() == n ? s0(i) : 0.;
                for (std::size_t c = 0; c < n; ++c)
                    Px(j * n + i, j * n + c) = (P0.Rows() == n && P0.Cols() == n) ? P0(i, c) : (i == c ? 1. : 0.);
            }
        SetInitialState(x0);
        SetInitialCovariance(Px);
    }

    void Lag::Sync() const
    {
        if (fInner->SampleRate() != fSampleRate) fInner->SetSampleRate(fSampleRate);
        if (fDelaysAt == fSampleRate) return;

        // The block each row of z reads: its sensor's delay, in steps of the model.
        std::vector<std::size_t> rows;
        for (const Sensor &sensor : fInner->Sensors())
        {
            const double steps = sensor.delay * fSampleRate;
            const std::size_t block = static_cast<std::size_t>(std::llround(steps));
            if (!(sensor.delay >= 0.) || block > fDepth)
                throw std::invalid_argument("Models::Lag: the sensor '" + sensor.name + "' is late by "
                                            + std::to_string(steps) + " steps and the model carries "
                                            + std::to_string(fDepth) + ".");
            rows.insert(rows.end(), sensor.Channels(), block);
        }
        if (rows.size() != fNMeasure)
            throw std::runtime_error("Models::Lag: the sensors of the inner model have " + std::to_string(rows.size())
                                     + " channels and its measurement " + std::to_string(fNMeasure) + ".");
        fRowBlock = std::move(rows);
        fBlocksRead = fRowBlock;
        std::sort(fBlocksRead.begin(), fBlocksRead.end());
        fBlocksRead.erase(std::unique(fBlocksRead.begin(), fBlocksRead.end()), fBlocksRead.end());
        fDelayed = fBlocksRead.back() != 0;
        fDelaysAt = fSampleRate;
    }

    Vector Lag::Block(const Vector &x, std::size_t steps) const
    {
        if (x.Size() != fNState)
            throw std::invalid_argument("Models::Lag: the state has " + std::to_string(x.Size())
                                        + " components, " + std::to_string(fNState) + " expected.");
        if (steps > fDepth)
            throw std::out_of_range("Models::Lag: the model carries " + std::to_string(fDepth) + " past states, not "
                                    + std::to_string(steps) + ".");
        Vector s(fNInner);
        for (std::size_t i = 0; i < fNInner; ++i) s(i) = x(steps * fNInner + i);
        return s;
    }

    std::size_t Lag::Delay(std::size_t i) const
    {
        Sync();
        return fRowBlock.at(i);
    }

    Vector Lag::Shift(const Vector &x, const Vector &head) const
    {
        Vector out(fNState);
        for (std::size_t i = 0; i < fNInner; ++i) out(i) = head(i);
        for (std::size_t i = fNInner; i < fNState; ++i) out(i) = x(i - fNInner);
        return out;
    }

    // --- maps ---------------------------------------------------------------------------

    Vector Lag::f(const Vector &x, const Vector &u, double t) const
    {
        Sync();
        return Shift(x, fInner->f(Block(x, 0), u, t));
    }

    Vector Lag::f(const Vector &x, const Vector &u, const Vector &w, double t) const
    {
        if (w.Size() != fNState)
            throw std::invalid_argument("Models::Lag: the process noise has " + std::to_string(w.Size())
                                        + " components, " + std::to_string(fNState) + " expected.");
        Sync();
        Vector ws(fNInner);
        for (std::size_t i = 0; i < fNInner; ++i) ws(i) = w(i);
        Vector out = Shift(x, fInner->f(Block(x, 0), u, ws, t));
        for (std::size_t i = fNInner; i < fNState; ++i) out(i) += w(i);
        return out;
    }

    Matrix Lag::Jacobian(const Vector &x, const Vector &u, double t) const
    {
        Sync();
        const Matrix Js = fInner->Jacobian(Block(x, 0), u, t);
        Matrix out(fNState, fNState, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, j) = Js(i, j);
        for (std::size_t i = fNInner; i < fNState; ++i) out(i, i - fNInner) = 1.;
        return out;
    }

    Matrix Lag::F(double t) const
    {
        Sync();
        const Matrix Fs = fInner->F(t);
        Matrix out(fNState, fNState, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, j) = Fs(i, j);
        for (std::size_t i = fNInner; i < fNState; ++i) out(i, i - fNInner) = 1.;
        return out;
    }

    Vector Lag::h(const Vector &x, double t) const
    {
        Sync();
        if (!fDelayed) return fInner->h(Block(x, 0), t);
        // One reading of the inner model per block that a sensor reads; each row
        // is taken from the reading of its own block, at its own time.
        Vector z(fNMeasure);
        for (const std::size_t block : fBlocksRead)
        {
            const Vector zb = fInner->h(Block(x, block), t - static_cast<double>(block) * DeltaT());
            for (std::size_t i = 0; i < fNMeasure; ++i)
                if (fRowBlock[i] == block) z(i) = zb(i);
        }
        return z;
    }

    Vector Lag::MeasureWithNoise(const Vector &x, const Vector &v, double t) const
    {
        Sync();
        if (!fDelayed) return fInner->MeasureWithNoise(Block(x, 0), v, t);
        Vector z(fNMeasure);
        for (const std::size_t block : fBlocksRead)
        {
            const Vector zb = fInner->MeasureWithNoise(Block(x, block), v, t - static_cast<double>(block) * DeltaT());
            for (std::size_t i = 0; i < fNMeasure; ++i)
                if (fRowBlock[i] == block) z(i) = zb(i);
        }
        return z;
    }

    Matrix Lag::JacobianH(const Vector &x, double t) const
    {
        Sync();
        Matrix out(fNMeasure, fNState, 0.);
        for (const std::size_t block : fBlocksRead)
        {
            const Matrix Hb = fInner->JacobianH(Block(x, block), t - static_cast<double>(block) * DeltaT());
            for (std::size_t i = 0; i < fNMeasure; ++i)
                if (fRowBlock[i] == block)
                    for (std::size_t j = 0; j < fNInner; ++j) out(i, block * fNInner + j) = Hb(i, j);
        }
        return out;
    }

    Matrix Lag::H(double t) const
    {
        Sync();
        const Matrix Hs = fInner->H(t);
        Matrix out(fNMeasure, fNState, 0.);
        if (Hs.Empty()) return out;
        for (std::size_t i = 0; i < fNMeasure; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, fRowBlock[i] * fNInner + j) = Hs(i, j);
        return out;
    }

    // --- coefficients ---------------------------------------------------------------------

    Matrix Lag::B(double t) const
    {
        if (fNControl == 0) return Matrix();
        const Matrix Bs = fInner->B(t);                // empty: u is added to the state directly
        Matrix out(fNState, fNControl, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNControl; ++j)
                out(i, j) = Bs.Empty() ? (i == j ? 1. : 0.) : Bs(i, j);
        return out;
    }

    Matrix Lag::L(double t) const
    {
        const Matrix Ls = fInner->L(t);
        if (Ls.Empty()) return Matrix();               // identity coupling throughout
        Matrix out(fNState, fNState, 0.);
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, j) = Ls(i, j);
        for (std::size_t i = fNInner; i < fNState; ++i) out(i, i) = 1.;
        return out;
    }

    Matrix Lag::Q(double t) const
    {
        Sync();
        const Matrix Qs = fInner->Q(t);
        Matrix out(fNState, fNState, 0.);              // the past states take no noise: they are copies
        for (std::size_t i = 0; i < fNInner; ++i)
            for (std::size_t j = 0; j < fNInner; ++j) out(i, j) = Qs(i, j);
        return out;
    }

    std::vector<Sensor> Lag::Sensors() const
    {
        Sync();
        std::vector<Sensor> out = fInner->Sensors();
        std::size_t row = 0;
        for (Sensor &sensor : out)
        {
            const std::size_t block = fRowBlock[row], n = fNInner, k = sensor.Channels();
            const double late = static_cast<double>(block) * DeltaT();
            row += k;
            sensor.delay = 0.;                         // dealt with: on this model's state it reads on time
            if (sensor.IsLinear())
            {
                Matrix G(k, fNState, 0.);
                for (std::size_t i = 0; i < k; ++i)
                    for (std::size_t j = 0; j < n; ++j) G(i, block * n + j) = sensor.G(i, j);
                sensor.G = G;
                continue;
            }
            const Sensor::Measure g = sensor.g;
            sensor.g = [this, g, block, late](const Vector &x, double t) { return g(Block(x, block), t - late); };
            if (sensor.jacobian)
            {
                const Sensor::MeasureJacobian dg = sensor.jacobian;
                sensor.jacobian = [this, dg, block, late, k, n](const Vector &x, double t) {
                    const Matrix Gb = dg(Block(x, block), t - late);
                    Matrix G(k, fNState, 0.);
                    for (std::size_t i = 0; i < k; ++i)
                        for (std::size_t j = 0; j < n; ++j) G(i, block * n + j) = Gb(i, j);
                    return G;
                };
            }
        }
        return out;
    }

    void Lag::CheckShapes() const
    {
        Sync();
        if (!fDelayed) fInner->CheckDelays();          // a late sensor this model cannot see is refused
        Model::CheckShapes();
    }

    std::string Lag::Describe() const
    {
        std::ostringstream os;
        os << Model::Describe() << "; the last " << fDepth << " state" << (fDepth == 1 ? "" : "s") << " of "
           << fInner->Describe();
        return os.str();
    }
}
