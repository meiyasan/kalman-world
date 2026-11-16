/**
 * *********************************************
 *
 * \file Fusion.h
 * \brief Header of the Kalman::Models::Fusion class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Fusion
 * \brief A model read by one more sensor: several sensors on the same state.
 *
 * \details The state and its dynamics are the inner model's, untouched. The
 * measurement grows by the channels of the added Kalman::Sensor,
 *
 * \code{.unparsed}
 *     z = [ h_inner(x, t) ; g(x, t) ]          R = diag(R_inner, R_sensor)
 * \endcode
 *
 * and one filter update weighs every sensor by its noise: that is sensor
 * fusion. The sensor reads through a function `g` (its Jacobian given, or
 * differenced with the inner model's scheme) or a matrix `G`. With a matrix on
 * an inner model that is its matrices, the result is still its matrices: H(t)
 * is the inner one with `G` below it, and the linear engine runs it.
 *
 * Sensors() lists the inner model's sensors, then the added one, so adding
 * sensors one after another gives one flat list.
 *
 * This is not Models::Composite, which puts models with their own states side
 * by side: here a second instrument reads the same state (a Doppler radar on a
 * tracked shell, a second camera).
 *
 * The inner model is shared, not copied; it receives this model's sample rate
 * before each evaluation, and its named parameters are this model's, so the
 * operations nest (a parameter of the inner model can be estimated through
 * Models::Joint around this one).
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_FUSION_H
#define KALMAN_MODELS_FUSION_H

#include "Kalman/Model.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Kalman::Models
{
    class Fusion : public Model
    {
        private:
            std::shared_ptr<Model> fInner;
            Sensor fSensor;                   // the added sensor
            std::size_t fNInnerMeasure = 0;   // m of the inner model
            std::size_t fNExtra = 0;          // its number of channels, k

            void Sync() const;
            Vector Extra(const Vector &x, double t) const;
            Vector Stack(const Vector &head, const Vector &tail) const;

        public:
            /**
             * \param inner   the model that gains a sensor (shared).
             * \param sensor  the sensor; checked against the inner state (Sensor::Check).
             * \throws std::invalid_argument for a null model, or a sensor that does not fit.
             */
            Fusion(std::shared_ptr<Model> inner, Sensor sensor);

            const char *Title() const override { return "Fusion Model"; }
            const char *Description() const override
            { return "A model read by one more sensor on the same state."; }

            const Model &Inner() const { return *fInner; }
            Model &Inner() { return *fInner; }
            /// The sensor that was added.
            const Sensor &Added() const { return fSensor; }
            /// The number of channels it added.
            std::size_t NExtra() const { return fNExtra; }
            std::vector<Sensor> Sensors() const override;
            /// The inner model's initial state: the state is the same.
            void AdoptInitialState() override;

            // --- the dynamics are the inner model's ---------------------------------
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override { Sync(); return fInner->f(x, u, t); }
            Matrix Jacobian(const Vector &x, const Vector &u, double t = 0.) const override
            { Sync(); return fInner->Jacobian(x, u, t); }
            using Model::f;
            bool HasProcessNoiseInput() const override { return fInner->HasProcessNoiseInput(); }
            Vector f(const Vector &x, const Vector &u, const Vector &w, double t) const override
            { Sync(); return fInner->f(x, u, w, t); }
            Vector LimitControl(const Vector &u) const override { return fInner->LimitControl(u); }
            Matrix F(double t = 0.) const override { Sync(); return fInner->F(t); }
            Matrix B(double t = 0.) const override { return fInner->B(t); }
            Matrix L(double t = 0.) const override { return fInner->L(t); }
            Matrix Q(double t = 0.) const override { Sync(); return fInner->Q(t); }

            // --- the measurement grows ---------------------------------------------
            Vector h(const Vector &x, double t = 0.) const override;
            Matrix JacobianH(const Vector &x, double t = 0.) const override;
            Matrix H(double t = 0.) const override;
            Matrix R(double t = 0.) const override;
            bool HasMeasurementNoiseInput() const override { return fInner->HasMeasurementNoiseInput(); }
            Vector MeasureWithNoise(const Vector &x, const Vector &v, double t) const override;

            // --- the parameters are the inner model's ---------------------------------
            std::vector<std::string> ParameterNames() const override { return fInner->ParameterNames(); }
            bool HasParameter(const std::string &name) const override { return fInner->HasParameter(name); }
            double Parameter(const std::string &name) const override { return fInner->Parameter(name); }
            void SetParameter(const std::string &name, double value) override { fInner->SetParameter(name, value); }

            void CheckShapes() const override;
            std::string Describe() const override;
    };
}

#endif // KALMAN_MODELS_FUSION_H
