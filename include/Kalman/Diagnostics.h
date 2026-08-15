/**
 * *********************************************
 *
 * \file Diagnostics.h
 * \brief Header of the Kalman::Diagnostics class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Diagnostics
 * \brief Is the filter *consistent*? — computed post-hoc from a finished run.
 *
 * \details Separated from Kalman::Filter for the same reason as
 * `modules/framework/diagnostics.py`: these are questions asked *about* a run,
 * not steps *of* one. Nothing here feeds back into the estimate, and everything
 * is derived from the stored prior/posterior/innovation logs plus the model's
 * H and R, so a diagnostic can be added without touching an engine.
 *
 * The two that matter:
 *
 * - **NIS** — normalised innovation squared. Needs no truth, so it works on real
 *   data. Averages the number of measurement channels for a consistent filter;
 *   persistently larger means the filter trusts the model too much (Q too small),
 *   persistently smaller means Q is too large.
 * - **NEES** — normalised estimation error squared. Needs truth, so it is a
 *   simulation tool. Averages the state dimension for a consistent filter.
 *
 * A filter whose estimates look right on a plot but whose NIS sits at 5 is
 * reporting error bars that are a factor of two too small, and that is invisible
 * in the trajectory alone. Kalman::Filter forwards to this class, so
 * `filter.MeanNIS()` is the short way in.
 *
 * *********************************************
 */

#ifndef KALMAN_DIAGNOSTICS_H
#define KALMAN_DIAGNOSTICS_H

#include "Kalman/Matrix.h"

#include <vector>

namespace Kalman
{
    class Filter;

    class Diagnostics
    {
        private:
            const Filter &fFilter;

        public:
            explicit Diagnostics(const Filter &filter) : fFilter(filter) {}

            /**
             * \brief Per-step `nu^T S^-1 nu`.
             *
             * Steps at which nothing was measured are reported as NaN, so they stay
             * out of averages instead of biasing them toward zero.
             */
            std::vector<double> NIS() const;
            /// Mean of NIS() over the run, ignoring the unmeasured (NaN) steps.
            double MeanNIS() const;

            /**
             * \brief Per-step `e^T P^-1 e` against a known truth.
             * \param truth one true state (n) per step.
             */
            std::vector<double> NEES(const std::vector<Vector> &truth) const;
            /// Mean of NEES() over the run.
            double MeanNEES(const std::vector<Vector> &truth) const;

            /// Per-step innovation covariance `S = H P H^T + R`.
            std::vector<Matrix> InnovationCovariance() const;

            /// Per-step Kalman gain `K = P H^T S^-1`.
            std::vector<Matrix> Gains() const;
    };
}

#endif // KALMAN_DIAGNOSTICS_H
