/**
 * *********************************************
 *
 * \file Record.h
 * \brief Header of the Kalman::Record class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::IO::Record
 * \brief One filtered (or smoothed) pass: time, state, covariance, measurement.
 *
 * \details The output type of Kalman::Filter::Forward() / Backward(), and the
 * counterpart of `KalmanRecord` on the Python side.
 *
 * It is a **plain value type** and it draws nothing. Series() and Sigma() hand
 * back the arrays a plot needs; turning those into a TGraph, a matplotlib axis or
 * a CSV is the job of whoever integrates the library, because that is where the
 * plotting dependency belongs. The +/- K sigma band is available here because it
 * is the whole point of running a Kalman filter rather than a smoothing spline,
 * and it is the first thing lost when estimates are shipped around as bare arrays.
 *
 * *********************************************
 */

#ifndef KALMAN_RECORD_H
#define KALMAN_RECORD_H

#include "Kalman/Matrix.h"
#include "Kalman/Constants.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace Kalman::IO
{
    class Record
    {
        private:
            std::string fName = "record";
            std::vector<double> fTime;      // (N) time tag of each step [s]
            std::vector<Vector> fState;     // (N) each (n) — the estimate
            std::vector<Matrix> fCov;       // (N) each (n,n) — its covariance
            std::vector<Vector> fMeasure;   // (N) each (m) — the measurement fed in
            StateFlavor fFlavor = StateFlavor::Posterior;

        public:
            Record() = default;
            explicit Record(std::string name, StateFlavor flavor = StateFlavor::Posterior);

            /// Append one step. \p z may be empty when no measurement was recorded.
            void Add(double t, const Vector &x, const Matrix &P, const Vector &z);

            void Reserve(std::size_t n);
            void Clear();

            const std::string &Name() const { return fName; }
            void SetName(std::string name) { fName = std::move(name); }

            std::size_t NSteps()   const { return fState.size(); }
            std::size_t NState()   const { return fState.empty()   ? 0 : fState.front().Size(); }
            std::size_t NMeasure() const { return fMeasure.empty() ? 0 : fMeasure.front().Size(); }

            StateFlavor Flavor() const { return fFlavor; }
            void SetFlavor(StateFlavor flavor) { fFlavor = flavor; }

            double Time(std::size_t k) const;
            const Vector &State(std::size_t k) const;
            const Matrix &Covariance(std::size_t k) const;
            const Vector &Measurement(std::size_t k) const;

            const std::vector<double> &Times() const { return fTime; }

            /// Component \p i of the state across the whole run.
            std::vector<double> Series(std::size_t i) const;
            /// Standard deviation `sqrt(P_ii)` of component \p i across the run.
            std::vector<double> Sigma(std::size_t i) const;
            /// Channel \p j of the measurement across the run (NaN where it was missing).
            std::vector<double> MeasurementSeries(std::size_t j) const;

            /// Root-mean-square difference between component \p i and \p truth.
            double RMSE(std::size_t i, const std::vector<Vector> &truth) const;

            void Print(std::ostream &os) const;
    };

    std::ostream &operator<<(std::ostream &os, const Record &record);
}

namespace Kalman
{
    /// `Kalman::Record` is the short spelling of Kalman::IO::Record, which lives in
    /// `IO/` to mirror `modules/framework/io/record.py`.
    using Record = IO::Record;
}

#endif // KALMAN_RECORD_H
