/**
 * *********************************************
 *
 * \file Fusion.cc
 * \brief Source of the Kalman::Models::Fusion class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Fusion.h"
#include "Kalman/Differentiator.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace Kalman::Models
{
    Fusion::Fusion(std::shared_ptr<Model> inner, Sensor sensor)
        : fInner(std::move(inner)), fSensor(std::move(sensor))
    {
        if (!fInner) throw std::invalid_argument("Models::Fusion: the inner model is null.");
        fSensor.Check(fInner->NState());
        fNExtra = fSensor.Channels();
        fNInnerMeasure = fInner->NMeasure();
        SetName(fInner->Name());
        Allocate(fInner->NState(), fNInnerMeasure + fNExtra, fInner->NControl());

        SetSampleRate(fInner->SampleRate());
        SetT0(fInner->T0());
        SetTimeUnit(fInner->TimeUnit());
        SetDefaultEngine(fInner->DefaultEngine());
        SetBuiltInSensorName(fInner->BuiltInSensorName());
        AdoptInitialState();

        Settle();

        DeclareState(fInner->StateComponents());
        // An observable keeps its name unless the model already has one of that
        // name: it then takes its sensor's name in front ("gyro2_RATE").
        std::vector<Component> channels = fInner->MeasurementComponents();
        std::vector<Component> added = fSensor.Observables();
        for (auto &c : added)
        {
            for (const auto &known : channels)
                if (known.name == c.name) { c.name = fSensor.name + "_" + c.name; break; }
            channels.push_back(c);
        }
        fSensor.channels = added;                       // what Sensors() reports is what z carries
        DeclareMeasurement(std::move(channels));
    }

    void Fusion::Settle()
    {
        const bool formulas = fSensor.IsLinear() || fSensor.jacobian;
        if (fSensor.IsLinear() && !fSensor.calibrated)
        {
            // Still what the inner model is: its matrices with one more block, or code.
            if (fInner->HasAnalyticJacobian()) SetAnalyticJacobian(true);
            else SetDifferentiator(fInner->Differentiator(), fInner->Epsilon());
            SetStructure(fInner->Structure());
            return;
        }
        // A function on top of a linear model is no longer its matrices; nor is a
        // calibrated reading, `gain * G x + offset`, which no matrix H carries. The
        // rows of the Jacobian without a formula are differenced with the inner
        // scheme (forward, at its default step, under an inner model that has formulas).
        if (fInner->DefaultEngine() == EngineKind::Linear) SetDefaultEngine(EngineKind::Extended);
        if (fInner->HasAnalyticJacobian() && formulas && fSensor.IsLinear()) SetAnalyticJacobian(true);
        else if (fInner->HasAnalyticJacobian()) SetDifferentiator(DifferentiatorKind::Forward);
        else SetDifferentiator(fInner->Differentiator(), fInner->Epsilon());
        SetStructure(StructureKind::Maps);
    }

    void Fusion::AdoptInitialState()
    {
        if (fInner->X0().Size() == fNState) SetInitialState(fInner->X0());
        if (fInner->P0().Rows() == fNState) SetInitialCovariance(fInner->P0());
    }

    std::vector<Sensor> Fusion::Sensors() const
    {
        std::vector<Sensor> out = fInner->Sensors();
        out.push_back(fSensor);
        return out;
    }

    // --- parameters: the inner model's, and the calibration of the added sensor ---------

    int Fusion::Own(const std::string &name) const
    {
        if (!fSensor.calibrated) return 0;
        const std::string &owner = fSensor.Owner();
        if (name == owner + ".gain") return 1;
        if (name == owner + ".offset") return 2;
        return 0;
    }

    void Fusion::Calibrate(double gain, double offset)
    {
        fSensor.calibrated = true;
        fSensor.gain = gain;
        fSensor.offset = offset;
        Settle();
    }

    std::vector<std::string> Fusion::ParameterNames() const
    {
        std::vector<std::string> names = fInner->ParameterNames();
        if (fSensor.calibrated)
            for (const char *suffix : {".gain", ".offset"})
            {
                const std::string name = fSensor.Owner() + suffix;
                if (std::find(names.begin(), names.end(), name) == names.end()) names.push_back(name);
            }
        std::sort(names.begin(), names.end());
        return names;
    }

    bool Fusion::HasParameter(const std::string &name) const { return Own(name) != 0 || fInner->HasParameter(name); }

    double Fusion::Parameter(const std::string &name) const
    {
        const int own = Own(name);
        return own == 1 ? fSensor.gain : own == 2 ? fSensor.offset : fInner->Parameter(name);
    }

    void Fusion::SetParameter(const std::string &name, double value)
    {
        const int own = Own(name);
        if (own == 1) fSensor.gain = value;
        if (own == 2) fSensor.offset = value;
        // Sensors of one instrument share its calibration: the layers below take it too.
        if (own == 0 || fInner->HasParameter(name)) fInner->SetParameter(name, value);
    }

    void Fusion::Sync() const
    {
        if (fInner->SampleRate() != fSampleRate) fInner->SetSampleRate(fSampleRate);
    }

    Vector Fusion::Extra(const Vector &x, double t) const
    {
        return fSensor.Read(x, t);
    }

    Vector Fusion::Stack(const Vector &head, const Vector &tail) const
    {
        Vector z(fNMeasure);
        for (std::size_t i = 0; i < fNInnerMeasure; ++i) z(i) = head(i);
        for (std::size_t i = 0; i < fNExtra; ++i) z(fNInnerMeasure + i) = tail(i);
        return z;
    }

    Vector Fusion::h(const Vector &x, double t) const
    {
        Sync();
        return Stack(fInner->h(x, t), Extra(x, t));
    }

    Vector Fusion::MeasureWithNoise(const Vector &x, const Vector &v, double t) const
    {
        if (v.Size() != fNMeasure)
            throw std::invalid_argument("Models::Fusion: the measurement noise has " + std::to_string(v.Size())
                                        + " components, " + std::to_string(fNMeasure) + " expected.");
        Sync();
        Vector head(fNInnerMeasure);
        for (std::size_t i = 0; i < fNInnerMeasure; ++i) head(i) = v(i);
        Vector tail = Extra(x, t);
        for (std::size_t i = 0; i < fNExtra; ++i) tail(i) += v(fNInnerMeasure + i);
        return Stack(fInner->MeasureWithNoise(x, head, t), tail);
    }

    Matrix Fusion::JacobianH(const Vector &x, double t) const
    {
        Sync();
        const Matrix Hs = fInner->JacobianH(x, t);
        Matrix G;
        const double gain = fSensor.calibrated ? fSensor.gain : 1.;
        if (fSensor.IsLinear()) { G = fSensor.G; G *= gain; }
        else if (fSensor.jacobian) { G = fSensor.jacobian(x, t); G *= gain; }
        else
            G = Differentiators::Jacobian(
                fNumericalScheme, [this, t](const Vector &xx) { return Extra(xx, t); }, x, fEpsilon);
        if (G.Rows() != fNExtra || G.Cols() != fNState)
            throw std::runtime_error("Models::Fusion: the Jacobian of the extra channels is not k x n.");

        Matrix out(fNMeasure, fNState, 0.);
        for (std::size_t j = 0; j < fNState; ++j)
        {
            for (std::size_t i = 0; i < fNInnerMeasure; ++i) out(i, j) = Hs(i, j);
            for (std::size_t i = 0; i < fNExtra; ++i) out(fNInnerMeasure + i, j) = G(i, j);
        }
        return out;
    }

    Matrix Fusion::H(double t) const
    {
        // The matrix a linear engine reads. With a function for the extra
        // channels there is none: their rows are left at zero, and the engines
        // that linearise ask JacobianH().
        const Matrix Hs = fInner->H(t);
        Matrix out(fNMeasure, fNState, 0.);
        for (std::size_t j = 0; j < fNState; ++j)
        {
            for (std::size_t i = 0; i < fNInnerMeasure; ++i) out(i, j) = Hs(i, j);
            if (fSensor.IsLinear())
                for (std::size_t i = 0; i < fNExtra; ++i)
                    out(fNInnerMeasure + i, j) = (fSensor.calibrated ? fSensor.gain : 1.) * fSensor.G(i, j);
        }
        return out;
    }

    Matrix Fusion::R(double t) const
    {
        Sync();
        const Matrix Rs = fInner->R(t);
        Matrix out(fNMeasure, fNMeasure, 0.);
        for (std::size_t i = 0; i < fNInnerMeasure; ++i)
            for (std::size_t j = 0; j < fNInnerMeasure; ++j) out(i, j) = Rs(i, j);
        for (std::size_t i = 0; i < fNExtra; ++i)
            for (std::size_t j = 0; j < fNExtra; ++j) out(fNInnerMeasure + i, fNInnerMeasure + j) = fSensor.R(i, j);
        return out;
    }

    void Fusion::SetDelay(double delay)
    {
        if (!(delay >= 0.)) throw std::invalid_argument("Models::Fusion: a delay cannot be negative.");
        fSensor.delay = delay;
    }

    void Fusion::CheckDelays() const
    {
        if (fSensor.delay > 0.)
            throw std::invalid_argument("Models::Fusion: the sensor '" + fSensor.name + "' reports late (delay "
                                        + std::to_string(fSensor.delay) + ") and the model carries no past state "
                                        "before it: add the sensors first, then lag the model (Models::Lag).");
        fInner->CheckDelays();
    }

    void Fusion::CheckShapes() const
    {
        Sync();
        fInner->CheckShapes();
        Model::CheckShapes();
    }

    std::string Fusion::Describe() const
    {
        std::ostringstream os;
        os << Model::Describe() << "; sensor '" << fSensor.name << "' (" << fNExtra << " channel"
           << (fNExtra == 1 ? "" : "s") << ") added to "
           << fInner->Describe();
        return os.str();
    }
}
