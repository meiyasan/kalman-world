/**
 * *********************************************
 *
 * \file Callback.cc
 * \brief Implementation of the Kalman::Models::Callback class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Callback.h"

#include <stdexcept>
#include <vector>

namespace Kalman::Models
{
    namespace
    {
        void Check(int status, const char *what)
        {
            if (status != 0)
                throw std::runtime_error(std::string("Kalman::Models::Callback: the caller's ") + what + " failed.");
        }

        Vector ToVector(const std::vector<double> &data)
        {
            Vector out(data.size());
            for (std::size_t i = 0; i < data.size(); ++i) out(i) = data[i];
            return out;
        }

        Matrix ToMatrix(const std::vector<double> &data, std::size_t rows, std::size_t cols)
        {
            Matrix out(rows, cols, 0.);
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t j = 0; j < cols; ++j) out(i, j) = data[i * cols + j];
            return out;
        }

        std::vector<double> Flat(const Vector &v)
        {
            std::vector<double> out(v.Size());
            for (std::size_t i = 0; i < v.Size(); ++i) out[i] = v(i);
            return out;
        }
    }

    Callback::Callback(std::string name, std::size_t n, std::size_t m, std::size_t l,
                       StepFn step, MeasureFn measure,
                       StepJacobianFn stepJacobian, MeasureJacobianFn measureJacobian, void *user)
        : Extended(std::move(name), n, m, l), fStep(step), fMeasure(measure),
          fStepJacobian(stepJacobian), fMeasureJacobian(measureJacobian), fUser(user)
    {
        if (!step || !measure)
            throw std::invalid_argument("Kalman::Models::Callback: both f and h must be given.");
    }

    Vector Callback::f(const Vector &x, const Vector &u, double t) const
    {
        const std::vector<double> xin = Flat(x), uin = Flat(u);
        std::vector<double> out(NState());
        if (fProcessNoise) { fLastX = x; fLastU = u; }
        Check(fStep(fUser, xin.data(), uin.size() == NControl() && NControl() > 0 ? uin.data() : nullptr, t, out.data()), "f");
        return ToVector(out);
    }

    Vector Callback::h(const Vector &x, double t) const
    {
        const std::vector<double> xin = Flat(x);
        std::vector<double> out(NMeasure());
        Check(fMeasure(fUser, xin.data(), t, out.data()), "h");
        return ToVector(out);
    }

    Matrix Callback::Jacobian(const Vector &x, const Vector &u, double t) const
    {
        if (!fStepJacobian) return Extended::Jacobian(x, u, t);
        const std::vector<double> xin = Flat(x), uin = Flat(u);
        std::vector<double> out(NState() * NState());
        Check(fStepJacobian(fUser, xin.data(), uin.size() == NControl() && NControl() > 0 ? uin.data() : nullptr, t, out.data()),
              "Jacobian of f");
        return ToMatrix(out, NState(), NState());
    }

    void Callback::SetNoise(ProcessNoiseFn processNoise, MeasurementNoiseFn measurementNoise)
    {
        fProcessNoise = processNoise;
        fMeasurementNoise = measurementNoise;
    }

    Matrix Callback::Q(double t) const
    {
        if (!fProcessNoise) return Extended::Q(t);
        const Vector x = fLastX.Size() == NState() ? fLastX : X0();
        const std::vector<double> xin = Flat(x), uin = Flat(fLastU);
        std::vector<double> out(NState() * NState());
        Check(fProcessNoise(fUser, xin.data(), uin.size() == NControl() && NControl() > 0 ? uin.data() : nullptr, t, out.data()),
              "process noise");
        return ToMatrix(out, NState(), NState());
    }

    Matrix Callback::R(double t) const
    {
        if (!fMeasurementNoise) return Extended::R(t);
        std::vector<double> out(NMeasure() * NMeasure());
        Check(fMeasurementNoise(fUser, t, out.data()), "measurement noise");
        return ToMatrix(out, NMeasure(), NMeasure());
    }

    Matrix Callback::L(double t) const
    {
        return fProcessNoise ? Matrix() : Extended::L(t);
    }

    Matrix Callback::JacobianH(const Vector &x, double t) const
    {
        if (!fMeasureJacobian) return Extended::JacobianH(x, t);
        const std::vector<double> xin = Flat(x);
        std::vector<double> out(NMeasure() * NState());
        Check(fMeasureJacobian(fUser, xin.data(), t, out.data()), "Jacobian of h");
        return ToMatrix(out, NMeasure(), NState());
    }
}
