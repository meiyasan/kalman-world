/**
 * *********************************************
 *
 * \file Filter.cc
 * \brief Implementation of the Kalman::Filter class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Filter.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Engines/Extended.h"
#include "Kalman/Engines/Linear.h"
#include "Kalman/Engines/Particle.h"
#include "Kalman/Engines/Unscented.h"
#include "Kalman/Diagnostics.h"
#include "Kalman/Utils.h"

#include <cmath>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace Kalman
{
    Filter::Filter(Model *model, Engine *engine)
        : Filter(model,
                 engine ? engine : MakeEngine(model ? model->DefaultEngine() : EngineKind::Linear),
                 engine == nullptr)
    {
    }

    Filter::Filter(Model *model, Engine *engine, bool owns)
        : fModel(model), fEngine(engine), fOwnsEngine(owns)
    {
        if (!fModel)
        {
            if (owns) delete fEngine;
            throw std::runtime_error("Kalman::Filter: a model is required.");
        }

        fModel->CheckShapes();
        Reset();
    }

    Filter::~Filter()
    {
        if (fOwnsEngine) delete fEngine;
    }

    Engine *Filter::MakeEngine(EngineKind kind)
    {
        switch (kind)
        {
            case EngineKind::Linear:    return new Engines::Linear();
            case EngineKind::Extended:  return new Engines::Extended();
            case EngineKind::Unscented: return new Engines::Unscented();
            case EngineKind::Particle:  return new Engines::Particle();
            case EngineKind::Ensemble:  return new Engines::Ensemble();
            // As a per-step engine the combinatorial one is an Extended step; the
            // track-finding pass itself is reached through Filters::CKF.
            case EngineKind::Combinatorial: return new Engines::Combinatorial();
        }

        // Fail loudly rather than silently substituting a different algorithm: an
        // integer that is not a kind (a corrupt spec, a version skew) must not
        // quietly become a linear step.
        std::ostringstream os;
        os << "Kalman::Filter: no engine implements kind " << static_cast<int>(kind)
           << " ('" << ToString(kind) << "').";
        throw std::runtime_error(os.str());
    }

    void Filter::SetEngine(Engine *engine)
    {
        if (!engine) throw std::runtime_error("Kalman::Filter: the engine must not be null.");
        if (fOwnsEngine) delete fEngine;
        fEngine = engine;
        fOwnsEngine = false;
        fEngine->Reset(*fModel);
    }

    void Filter::Reset()
    {
        fX = fModel->X0();
        fP = fModel->P0();
        if (fEngine) fEngine->Reset(*fModel);

        fPriorX.clear(); fPriorP.clear();
        fPostX.clear();  fPostP.clear();
        fTransition.clear();
        fMeasure.clear();
        fInnovation.clear();
        fTime.clear();
        fSmoothX.clear(); fSmoothP.clear();
    }

    Record Filter::Forward(const std::vector<Vector> &zk, const std::vector<Vector> *uk,
                           bool reset)
    {
        if (zk.empty()) throw std::runtime_error("Kalman::Filter::Forward: no measurements given.");
        fModel->CheckShapes();

        const std::size_t m = fModel->NMeasure();
        const std::size_t l = fModel->NControl();

        if (uk && l > 0 && uk->size() != zk.size())
            throw std::runtime_error("Kalman::Filter::Forward: uk and zk must have the same length.");

        if (reset) Reset();

        const std::size_t N = zk.size();
        const double dt = fModel->DeltaT();
        double t = fModel->T0();

        fPriorX.reserve(fPriorX.size() + N);
        fPostX.reserve(fPostX.size() + N);

        for (std::size_t k = 0; k < N; ++k)
        {
            const Vector &z = zk[k];
            if (z.Size() != m)
            {
                std::ostringstream os;
                os << "Kalman::Filter::Forward: measurement " << k << " has " << z.Size()
                   << " channels, expected " << m << ".";
                throw std::runtime_error(os.str());
            }

            const Vector *u = (uk && l > 0) ? &(*uk)[k] : nullptr;

            // --- predict --------------------------------------------------
            Vector xPrior;
            Matrix PPrior, F;
            fEngine->Predict(*fModel, fX, fP, u, t, xPrior, PPrior, F);

            // --- innovation (kept for diagnostics, before the correction) --
            Vector nu;
            Matrix S;
            fEngine->Innovation(*fModel, xPrior, PPrior, z, t, nu, S);

            // --- update ---------------------------------------------------
            Vector xPost;
            Matrix PPost;
            fEngine->Update(*fModel, xPrior, PPrior, z, t, xPost, PPost);

            fPriorX.push_back(xPrior);
            fPriorP.push_back(PPrior);
            fPostX.push_back(xPost);
            fPostP.push_back(PPost);
            fTransition.push_back(F);
            fMeasure.push_back(z);
            fInnovation.push_back(nu);
            fTime.push_back(t);

            fX = xPost;
            fP = PPost;

            t += dt;
        }

        return MakeRecord(fPostX, fPostP, StateFlavor::Posterior);
    }

    Record Filter::Backward()
    {
        const std::size_t N = NSteps();
        if (N == 0) throw std::runtime_error("Kalman::Filter::Backward: run Forward() first.");
        if (!fEngine->HasSmoother())
            throw std::runtime_error(std::string("Kalman::Filter::Backward: the ") + fEngine->Title()
                                     + " has no RTS smoother; use Forward() only.");

        fSmoothX = fPostX;
        fSmoothP = fPostP;

        // The last posterior is already the smoothed estimate; walk backwards from it.
        for (std::size_t k = N - 1; k-- > 0;)
        {
            Vector xs;
            Matrix Ps;
            fEngine->Smooth(fPostX[k], fPostP[k],
                            fPriorX[k + 1], fPriorP[k + 1],
                            fSmoothX[k + 1], fSmoothP[k + 1],
                            fTransition[k + 1],
                            xs, Ps);
            fSmoothX[k] = xs;
            fSmoothP[k] = Ps;
        }

        return MakeRecord(fSmoothX, fSmoothP, StateFlavor::Smoothed);
    }

    Record Filter::Filtering(const std::vector<Vector> &zk, const std::vector<Vector> *uk,
                             bool smooth)
    {
        Record forward = Forward(zk, uk);
        return smooth ? Backward() : forward;
    }

    Record Filter::PriorRecord() const
    {
        return MakeRecord(fPriorX, fPriorP, StateFlavor::Prior);
    }

    Record Filter::MakeRecord(const std::vector<Vector> &x, const std::vector<Matrix> &P,
                              StateFlavor flavor) const
    {
        Record record(fModel->Name(), flavor);
        record.Reserve(x.size());
        for (std::size_t k = 0; k < x.size(); ++k)
            record.Add(fTime[k], x[k], P[k], fMeasure[k]);
        return record;
    }

    // --- diagnostics: delegated to Kalman::Diagnostics -----------------------
    // These are questions asked *about* a run, not steps *of* one, so the code
    // lives in its own class (mirroring framework/diagnostics.py). The forwarders
    // stay because `filter.MeanNIS()` is what a caller wants to write.

    std::vector<double> Filter::NIS() const { return GetDiagnostics().NIS(); }

    double Filter::MeanNIS() const { return GetDiagnostics().MeanNIS(); }

    std::vector<double> Filter::NEES(const std::vector<Vector> &truth) const
    {
        return GetDiagnostics().NEES(truth);
    }

    double Filter::MeanNEES(const std::vector<Vector> &truth) const
    {
        return GetDiagnostics().MeanNEES(truth);
    }

    void Filter::Print(std::ostream &os) const
    {
        os << (fModel ? fModel->Name() : "filter")
           << " — engine = " << (fEngine ? fEngine->Title() : "(none)")
           << ", " << NSteps() << " steps run"
           << (fSmoothX.empty() ? "" : ", smoothed");
    }
}
