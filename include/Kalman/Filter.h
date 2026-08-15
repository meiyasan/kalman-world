/**
 * *********************************************
 *
 * \file Filter.h
 * \brief Header of the Kalman::Filter class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Filter
 * \brief The **workflow** — WHEN each step runs, and what is kept.
 *
 * \details A filter composes a model (WHAT) with an engine (HOW) and owns the
 * run: the current belief `(x, P)` and the prior / posterior / transition logs
 * that the RTS backward pass and the consistency diagnostics need. It implements
 * no covariance math of its own — every matrix operation happens in the engine.
 *
 * \code{.cpp}
 * MyModel model;                          // your system; libKalman ships no physics
 * Kalman::Filters::KF filter(&model);
 *
 * auto forward  = filter.Forward(measurements);   // filtered estimate
 * auto smoothed = filter.Backward();              // RTS over the same run
 * \endcode
 *
 * Missing data needs no special handling: a NaN channel in a measurement is
 * skipped by the update, so a step with an entirely-NaN measurement is a pure
 * prediction and the covariance grows accordingly.
 *
 * \note The filter does **not** own the model — it is passed by pointer and must
 * outlive the filter. It *does* own the engine it creates for itself; an engine
 * passed in explicitly stays the caller's.
 *
 * *********************************************
 */

#ifndef KALMAN_FILTER_H
#define KALMAN_FILTER_H

#include "Kalman/Abstract/Metadata.h"
#include "Kalman/Diagnostics.h"
#include "Kalman/Engine.h"
#include "Kalman/Matrix.h"
#include "Kalman/Model.h"
#include "Kalman/IO/Record.h"
#include "Kalman/Constants.h"

#include <iosfwd>
#include <vector>

namespace Kalman
{
    class Filter : public Abstract::Metadata
    {
        protected:
            Model *fModel = nullptr;      // the system (not owned)
            Engine *fEngine = nullptr;    // the algorithm
            bool fOwnsEngine = false;     // true when the engine was built here

            Vector fX;                    // current state estimate
            Matrix fP;                    // current covariance

            std::vector<Vector> fPriorX;      // per step: predicted mean
            std::vector<Matrix> fPriorP;      // per step: predicted covariance
            std::vector<Vector> fPostX;       // per step: corrected mean
            std::vector<Matrix> fPostP;       // per step: corrected covariance
            std::vector<Matrix> fTransition;  // per step: the F used
            std::vector<Vector> fMeasure;     // per step: the measurement fed in
            std::vector<Vector> fInnovation;  // per step: nu = z - h(x_prior)
            std::vector<double> fTime;        // per step: time tag

            std::vector<Vector> fSmoothX;     // filled by Backward()
            std::vector<Matrix> fSmoothP;

        public:
            /**
             * \param model  the system; must outlive the filter.
             * \param engine the algorithm, or nullptr to build the one the model asks
             *               for through Kalman::Model::DefaultEngine().
             */
            explicit Filter(Model *model, Engine *engine = nullptr);
            virtual ~Filter();

            Filter(const Filter &) = delete;
            Filter &operator=(const Filter &) = delete;

            const char *Title() const override { return "Kalman Filter"; }
            const char *Description() const override
            { return "Forward/backward workflow over a composed model + engine."; }

            Model *GetModel() const { return fModel; }
            Engine *GetEngine() const { return fEngine; }
            /// Adopt a different engine; the filter never takes ownership here.
            void SetEngine(Engine *engine);

            const Vector &State() const { return fX; }
            const Matrix &Covariance() const { return fP; }

            /// Drop the run and rewind the belief to the model's x0 / P0.
            void Reset();

            /**
             * \brief Run the forward pass over \p zk.
             * \param zk     one measurement (m) per step; NaN marks a missing channel.
             * \param uk     one control vector (l) per step, or nullptr.
             * \param reset  start a clean run (default) rather than continuing.
             * \return the posterior record.
             */
            Record Forward(const std::vector<Vector> &zk,
                           const std::vector<Vector> *uk = nullptr,
                           bool reset = true);

            /**
             * \brief Rauch–Tung–Striebel smoother over the stored forward pass.
             * \return the smoothed record.
             * \throws std::runtime_error when Forward() has not been run.
             */
            Record Backward();

            /// Forward, then optionally Backward — the one-call entry point.
            Record Filtering(const std::vector<Vector> &zk,
                             const std::vector<Vector> *uk = nullptr,
                             bool smooth = false);

            /// The prior (predicted) record of the last forward pass.
            Record PriorRecord() const;

            // --- consistency diagnostics -------------------------------------
            /**
             * \brief Normalised Innovation Squared, `nu^T S^-1 nu` per step.
             *
             * Needs no truth. For a consistent filter it averages the number of
             * measurement channels; persistently larger means the filter trusts the
             * model too much (Q too small), persistently smaller means Q is too big.
             * Steps with no measurement at all are reported as NaN so they stay out
             * of averages instead of biasing them toward zero.
             */
            std::vector<double> NIS() const;
            /// Mean of NIS() over the run, ignoring the NaN (unmeasured) steps.
            double MeanNIS() const;

            /**
             * \brief Normalised Estimation Error Squared against a known truth.
             * \param truth one true state (n) per step.
             *
             * Averages the state dimension for a consistent filter.
             */
            std::vector<double> NEES(const std::vector<Vector> &truth) const;
            /// Mean of NEES() over the run.
            double MeanNEES(const std::vector<Vector> &truth) const;

            std::size_t NSteps() const { return fPostX.size(); }

            // --- the stored run, which Kalman::Diagnostics reads ---------------
            const std::vector<Vector> &PriorStates() const { return fPriorX; }
            const std::vector<Matrix> &PriorCovariances() const { return fPriorP; }
            const std::vector<Vector> &PosteriorStates() const { return fPostX; }
            const std::vector<Matrix> &PosteriorCovariances() const { return fPostP; }
            const std::vector<Matrix> &Transitions() const { return fTransition; }
            const std::vector<Vector> &Measurements() const { return fMeasure; }
            const std::vector<Vector> &Innovations() const { return fInnovation; }
            const std::vector<double> &Times() const { return fTime; }
            const std::vector<Vector> &SmoothedStates() const { return fSmoothX; }
            const std::vector<Matrix> &SmoothedCovariances() const { return fSmoothP; }

            /// The consistency diagnostics for this run — see Kalman::Diagnostics.
            Diagnostics GetDiagnostics() const { return Diagnostics(*this); }

            void Print(std::ostream &os) const;

        protected:
            /**
             * \brief Constructor for the named leaves, which supply their own engine.
             * \param model  the system; must outlive the filter.
             * \param engine the algorithm to drive it with.
             * \param owns   when true the filter deletes \p engine in its destructor.
             */
            Filter(Model *model, Engine *engine, bool owns);

            /// Build the engine the model asks for. Throws for kinds with no C++ engine.
            static Engine *MakeEngine(EngineKind kind);

            /// Assemble a record from one of the stored logs.
            Record MakeRecord(const std::vector<Vector> &x, const std::vector<Matrix> &P,
                              StateFlavor flavor) const;
    };
}

#endif // KALMAN_FILTER_H
