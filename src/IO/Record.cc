/**
 * *********************************************
 *
 * \file Record.cc
 * \brief Implementation of the Kalman::Record class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/IO/Record.h"

#include <cmath>
#include <limits>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace Kalman::IO
{
    namespace
    {
        void CheckStep(std::size_t k, std::size_t n, const char *what)
        {
            if (k >= n)
            {
                std::ostringstream os;
                os << "Kalman::Record: " << what << " index " << k
                   << " is out of range [0, " << n << ").";
                throw std::out_of_range(os.str());
            }
        }
    }

    Record::Record(std::string name, StateFlavor flavor)
        : fName(std::move(name)), fFlavor(flavor) {}

    void Record::Add(double t, const Vector &x, const Matrix &P, const Vector &z)
    {
        fTime.push_back(t);
        fState.push_back(x);
        fCov.push_back(P);
        fMeasure.push_back(z);
    }

    void Record::Reserve(std::size_t n)
    {
        fTime.reserve(n);
        fState.reserve(n);
        fCov.reserve(n);
        fMeasure.reserve(n);
    }

    void Record::Clear()
    {
        fTime.clear();
        fState.clear();
        fCov.clear();
        fMeasure.clear();
    }

    double Record::Time(std::size_t k) const
    {
        CheckStep(k, fTime.size(), "time");
        return fTime[k];
    }

    const Vector &Record::State(std::size_t k) const
    {
        CheckStep(k, fState.size(), "state");
        return fState[k];
    }

    const Matrix &Record::Covariance(std::size_t k) const
    {
        CheckStep(k, fCov.size(), "covariance");
        return fCov[k];
    }

    const Vector &Record::Measurement(std::size_t k) const
    {
        CheckStep(k, fMeasure.size(), "measurement");
        return fMeasure[k];
    }

    std::vector<double> Record::Series(std::size_t i) const
    {
        std::vector<double> out(NSteps());
        for (std::size_t k = 0; k < NSteps(); ++k) out[k] = fState[k](i);
        return out;
    }

    std::vector<double> Record::Sigma(std::size_t i) const
    {
        std::vector<double> out(NSteps());
        for (std::size_t k = 0; k < NSteps(); ++k)
        {
            const double var = fCov[k](i, i);
            // A covariance that has gone very slightly negative through round-off
            // must not produce a NaN error bar; clamp instead.
            out[k] = var > 0. ? std::sqrt(var) : 0.;
        }
        return out;
    }

    std::vector<double> Record::MeasurementSeries(std::size_t j) const
    {
        std::vector<double> out(NSteps());
        for (std::size_t k = 0; k < NSteps(); ++k)
            out[k] = (j < fMeasure[k].Size()) ? fMeasure[k](j)
                                              : std::numeric_limits<double>::quiet_NaN();
        return out;
    }

    double Record::RMSE(std::size_t i, const std::vector<Vector> &truth) const
    {
        if (truth.size() != NSteps())
            throw std::runtime_error("Kalman::Record::RMSE: truth must have one state per step.");

        double sum = 0.;
        for (std::size_t k = 0; k < NSteps(); ++k)
        {
            const double d = fState[k](i) - truth[k](i);
            sum += d * d;
        }
        return NSteps() ? std::sqrt(sum / static_cast<double>(NSteps())) : 0.;
    }

    void Record::Print(std::ostream &os) const
    {
        os << fName << " — " << ToString(fFlavor) << ", " << NSteps()
           << " steps, n = " << NState() << ", m = " << NMeasure();
        if (!fTime.empty()) os << ", t in [" << fTime.front() << ", " << fTime.back() << "] s";
    }

    std::ostream &operator<<(std::ostream &os, const Record &record)
    {
        record.Print(os);
        return os;
    }
}
