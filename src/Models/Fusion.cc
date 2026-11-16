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

        if (fSensor.IsLinear())
        {
            // Still what the inner model is: its matrices with one more block, or code.
            if (fInner->HasAnalyticJacobian()) SetAnalyticJacobian(true);
            else SetDifferentiator(fInner->Differentiator(), fInner->Epsilon());
            SetStructure(fInner->Structure());
        }
        else
        {
            // A function on top of a linear model is no longer its matrices. Its rows
            // of the Jacobian are differenced with the inner scheme (forward, at its
            // default step, under an inner model that has formulas).
            if (fInner->DefaultEngine() == EngineKind::Linear) SetDefaultEngine(EngineKind::Extended);
            if (fInner->HasAnalyticJacobian()) SetDifferentiator(DifferentiatorKind::Forward);
            else SetDifferentiator(fInner->Differentiator(), fInner->Epsilon());
            SetStructure(StructureKind::Maps);
        }

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

    void Fusion::Sync() const
    {
        if (fInner->SampleRate() != fSampleRate) fInner->SetSampleRate(fSampleRate);
    }

    Vector Fusion::Extra(const Vector &x, double t) const
    {
        Vector g = fSensor.g ? fSensor.g(x, t) : fSensor.G * x;
        if (g.Size() != fNExtra)
            throw std::runtime_error("Models::Fusion: the sensor '" + fSensor.name + "' returned "
                                     + std::to_string(g.Size()) + " values, " + std::to_string(fNExtra) + " expected.");
        return g;
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
        if (fSensor.IsLinear()) G = fSensor.G;
        else if (fSensor.jacobian) G = fSensor.jacobian(x, t);
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
                for (std::size_t i = 0; i < fNExtra; ++i) out(fNInnerMeasure + i, j) = fSensor.G(i, j);
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
