/**
 * *********************************************
 *
 * \file Diagnostics.cc
 * \brief Implementation of the Kalman::Diagnostics class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Diagnostics.h"
#include "Kalman/Filter.h"
#include "Kalman/Utils.h"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace Kalman
{
    namespace
    {
        /// Innovation covariance at step \p k, restricted to the channels present.
        Matrix InnovationCovarianceAt(const Filter &filter, std::size_t k,
                                      const std::vector<std::size_t> &keep)
        {
            const Model &model = *filter.GetModel();
            const double t = filter.Times()[k];

            Vector zpred;
            Matrix H;
            filter.GetEngine()->Measure(model, filter.PriorStates()[k], t, zpred, H);

            const Matrix Hc = Utils::CompressRows(H, keep);
            const Matrix Rc = Utils::CompressBoth(model.R(t), keep);

            Matrix S = Utils::Similarity(Hc, filter.PriorCovariances()[k]);
            S += Rc;
            return S;
        }
    }

    std::vector<double> Diagnostics::NIS() const
    {
        const std::size_t N = fFilter.NSteps();
        std::vector<double> out(N);

        for (std::size_t k = 0; k < N; ++k)
        {
            const std::vector<std::size_t> keep = Utils::ValidChannels(fFilter.Innovations()[k]);
            if (keep.empty())
            {
                // Nothing was measured at this step, so there is no innovation to
                // normalise; NaN keeps it out of averages instead of biasing them to 0.
                out[k] = std::numeric_limits<double>::quiet_NaN();
                continue;
            }

            const Matrix S = InnovationCovarianceAt(fFilter, k, keep);
            const Vector nu = Utils::Compress(fFilter.Innovations()[k], keep);
            out[k] = Utils::Mahalanobis(nu, Utils::Inverse(S));
        }
        return out;
    }

    double Diagnostics::MeanNIS() const
    {
        const std::vector<double> nis = NIS();
        double sum = 0.;
        std::size_t used = 0;
        for (double v : nis)
            if (!std::isnan(v)) { sum += v; ++used; }
        return used ? sum / static_cast<double>(used)
                    : std::numeric_limits<double>::quiet_NaN();
    }

    std::vector<double> Diagnostics::NEES(const std::vector<Vector> &truth) const
    {
        const std::size_t N = fFilter.NSteps();
        if (truth.size() != N)
            throw std::runtime_error("Kalman::Diagnostics::NEES: truth must have one state per step.");

        std::vector<double> out(N);
        for (std::size_t k = 0; k < N; ++k)
        {
            Vector e(fFilter.PosteriorStates()[k]);
            e -= truth[k];
            out[k] = Utils::Mahalanobis(e, Utils::Inverse(fFilter.PosteriorCovariances()[k]));
        }
        return out;
    }

    double Diagnostics::MeanNEES(const std::vector<Vector> &truth) const
    {
        const std::vector<double> nees = NEES(truth);
        double sum = 0.;
        for (double v : nees) sum += v;
        return nees.empty() ? 0. : sum / static_cast<double>(nees.size());
    }

    std::vector<Matrix> Diagnostics::InnovationCovariance() const
    {
        const std::size_t N = fFilter.NSteps();
        std::vector<Matrix> out;
        out.reserve(N);

        for (std::size_t k = 0; k < N; ++k)
        {
            // Reported over *all* channels here, not just the present ones: this is
            // the model's predicted spread, which exists whether or not the sensor
            // reported. NIS() above is the one that has to compress.
            const Model &model = *fFilter.GetModel();
            const double t = fFilter.Times()[k];

            Vector zpred;
            Matrix H;
            fFilter.GetEngine()->Measure(model, fFilter.PriorStates()[k], t, zpred, H);

            Matrix S = Utils::Similarity(H, fFilter.PriorCovariances()[k]);
            S += model.R(t);
            out.push_back(S);
        }
        return out;
    }

    std::vector<Matrix> Diagnostics::Gains() const
    {
        const std::size_t N = fFilter.NSteps();
        std::vector<Matrix> out;
        out.reserve(N);

        for (std::size_t k = 0; k < N; ++k)
        {
            const Model &model = *fFilter.GetModel();
            const double t = fFilter.Times()[k];

            Vector zpred;
            Matrix H;
            fFilter.GetEngine()->Measure(model, fFilter.PriorStates()[k], t, zpred, H);

            const Matrix &P = fFilter.PriorCovariances()[k];
            Matrix S = Utils::Similarity(H, P);
            S += model.R(t);

            out.push_back(P * H.Transpose() * Utils::Inverse(S));
        }
        return out;
    }
}
