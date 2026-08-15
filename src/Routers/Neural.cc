/**
 * *********************************************
 *
 * \file Neural.cc
 * \brief Implementation of the Kalman::Routers::Neural class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * Note what is *not* included here: any inference runtime. This file owns the
 * features; the model owns the forward pass. That is what keeps the optional
 * dependency confined to Kalman::Routers::NN::Onnx.
 */

#include "Kalman/Routers/Neural.h"
#include "Kalman/Routers/Bayesian.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace Kalman::Routers
{
    Neural::Neural(const NN::Model *model, std::size_t window)
        : fModel(model), fWindow(window)
    {
        if (window == 0) throw std::runtime_error("Kalman::Routers::Neural: window must be positive.");
    }

    void Neural::SetWindow(std::size_t window)
    {
        if (window == 0) throw std::runtime_error("Kalman::Routers::Neural: window must be positive.");
        fWindow = window;
    }

    void Neural::SetNormalisation(const Vector &mean, const Vector &scale)
    {
        if (mean.Size() != scale.Size())
            throw std::runtime_error("Kalman::Routers::Neural: mean and scale differ in length.");

        fMean = mean;
        fScale = scale;
        for (std::size_t i = 0; i < fScale.Size(); ++i)
            if (!(std::abs(fScale(i)) > 1e-8)) fScale(i) = 1.;   // a constant feature
    }

    Vector Neural::Features(const Matrix &nis) const
    {
        const Matrix block = NisWindow(nis, fWindow);

        // Row-major flatten: [step0_model0, step0_model1, ..., step1_model0, ...].
        // Must match numpy's .ravel() in routers/neural.py exactly — get it wrong
        // and the policy still returns probabilities, just meaningless ones.
        Vector feat(block.Rows() * block.Cols());
        std::size_t k = 0;
        for (std::size_t r = 0; r < block.Rows(); ++r)
            for (std::size_t j = 0; j < block.Cols(); ++j)
                feat(k++) = block(r, j);

        if (fMean.Size() == feat.Size())
            for (std::size_t i = 0; i < feat.Size(); ++i)
                feat(i) = (feat(i) - fMean(i)) / fScale(i);

        return feat;
    }

    Vector Neural::Route(const Matrix &nis, const Vector &c, const Vector &likelihoods) const
    {
        if (!IsReady())
            return Bayesian().Route(nis, c, likelihoods);   // documented fallback

        const Vector feat = Features(nis);

        if (fModel->NInputs() && feat.Size() != fModel->NInputs())
        {
            std::ostringstream os;
            os << "Kalman::Routers::Neural: the policy expects " << fModel->NInputs()
               << " features but the NIS window produced " << feat.Size()
               << " (window=" << fWindow << ", models=" << nis.Cols()
               << "). The window must match the one used at training.";
            throw std::runtime_error(os.str());
        }

        const Vector mu = (*fModel)(feat);
        if (mu.Size() != c.Size())
        {
            std::ostringstream os;
            os << "Kalman::Routers::Neural: the policy returned " << mu.Size()
               << " probabilities for " << c.Size() << " models.";
            throw std::runtime_error(os.str());
        }
        return Normalise(mu);
    }
}
