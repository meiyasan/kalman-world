/**
 * *********************************************
 *
 * \file Composite.h
 * \brief Header of the Kalman::Models::Composite class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Composite
 * \brief Several models side by side, each with its own state.
 *
 * \details The parts evolve independently, so the state is theirs one after
 * the other and the motion is block-diagonal:
 *
 * \code{.unparsed}
 *     x = [ x1 ; x2 ; ... ]          f(x) = [ f1(x1) ; f2(x2) ; ... ]
 *     F = diag(F1, F2, ...)          Q = diag(Q1, Q2, ...)
 * \endcode
 *
 * What is measured depends on the Kalman::Readout:
 *
 * \code{.unparsed}
 *     Stacked:  z = [ h1(x1) ; h2(x2) ; ... ]     R = diag(R1, R2, ...)     each part brings its sensors
 *     Summed:   z = h1(x1) + h2(x2) + ...         R = R1                    the parts share one measurement
 * \endcode
 *
 * Summed is several signals on one channel: the lines of a spectrum, a signal
 * and its background. The parts then have the same number of measurements, and
 * the noise of the channel is the first part's.
 *
 * The controls are the parts' one after the other. A named parameter of a part
 * is `<part>.<name>` here, the part being called by its model's name (numbered
 * from the second when two share one: `osc`, `osc2`), or by the names given. A
 * state or measurement component keeps its name unless two parts share it; it
 * then carries its part's in front (`osc_position`, `osc2_position`).
 *
 * When every part is its matrices the composite is too, and the linear engine
 * runs it; one part that is code makes it code. Each part linearises itself:
 * the Jacobians are the parts' own, in blocks.
 *
 * The parts are shared, not copied, and step at this model's sample rate.
 * This is not Models::Fusion, where a second sensor reads the *same* state.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_COMPOSITE_H
#define KALMAN_MODELS_COMPOSITE_H

#include "Kalman/Model.h"

#include <memory>
#include <string>
#include <vector>

namespace Kalman::Models
{
    class Composite : public Model
    {
        private:
            std::vector<std::shared_ptr<Model>> fParts;
            std::vector<std::string> fPartNames;
            Readout fReadout = Readout::Stacked;
            std::vector<std::size_t> fState, fMeasure, fControl;   // where each part starts; one more entry: the end

            void Sync() const;
            Vector Cut(const Vector &v, const std::vector<std::size_t> &starts, std::size_t i, const char *what) const;
            /// The part and its own name for `<part>.<name>`. \throws std::out_of_range otherwise.
            std::pair<std::size_t, std::string> Resolve(const std::string &name) const;

        public:
            /**
             * \param parts    the models (shared), in the order of their states.
             * \param readout  stacked measurements, or one measurement they sum into.
             * \param names    what each part is called here; its model's name by default.
             * \throws std::invalid_argument for no part, a null part, parts at different
             *         sample rates, two parts of one given name, or, summed, parts with
             *         different numbers of measurements.
             */
            explicit Composite(std::vector<std::shared_ptr<Model>> parts, Readout readout = Readout::Stacked,
                               std::vector<std::string> names = {});

            const char *Title() const override { return "Composite Model"; }
            const char *Description() const override { return "Several models side by side, each with its own state."; }

            std::size_t NParts() const { return fParts.size(); }
            const Model &Part(std::size_t i) const { return *fParts.at(i); }
            Model &Part(std::size_t i) { return *fParts.at(i); }
            const std::string &PartName(std::size_t i) const { return fPartNames.at(i); }
            Readout GetReadout() const { return fReadout; }
            /// Where part \p i starts in the state (\p i = NParts(): the end).
            std::size_t StateStart(std::size_t i) const { return fState.at(i); }
            /// The state of part \p i, out of \p x.
            Vector PartState(const Vector &x, std::size_t i) const { return Cut(x, fState, i, "state"); }

            /// Stacked: every part's sensors, each reading its part of the state. Summed: the one measurement.
            std::vector<Sensor> Sensors() const override;
            void CheckDelays() const override;

            // --- maps ----------------------------------------------------------
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            Vector h(const Vector &x, double t = 0.) const override;
            Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const override;
            Matrix JacobianH(const Vector &x, double t = 0.) const override;

            using Model::f;
            bool HasProcessNoiseInput() const override;
            Vector f(const Vector &x, const Vector &u, const Vector &w, double t) const override;
            bool HasMeasurementNoiseInput() const override;
            Vector MeasureWithNoise(const Vector &x, const Vector &v, double t) const override;
            Vector LimitControl(const Vector &u) const override;

            // --- coefficients --------------------------------------------------
            Matrix F(double t = 0.) const override;
            Matrix H(double t = 0.) const override;
            Matrix B(double t = 0.) const override;
            Matrix L(double t = 0.) const override;
            Matrix Q(double t = 0.) const override;
            Matrix R(double t = 0.) const override;

            // --- the parameters are the parts', as "<part>.<name>" -------------------
            std::vector<std::string> ParameterNames() const override;
            bool HasParameter(const std::string &name) const override;
            double Parameter(const std::string &name) const override;
            void SetParameter(const std::string &name, double value) override;

            /// The parts' initial states one after the other; their covariances in blocks.
            void AdoptInitialState() override;
            void CheckShapes() const override;
            std::string Describe() const override;
    };
}

#endif // KALMAN_MODELS_COMPOSITE_H
