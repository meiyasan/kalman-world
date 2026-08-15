/**
 * *********************************************
 *
 * \file MLP.cc
 * \brief Implementation of the Kalman::Routers::NN::MLP class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Routers/NN/MLP.h"

#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Kalman::Routers::NN
{
    void MLP::AddLayer(const Matrix &weight, const Vector &bias)
    {
        if (weight.Rows() != bias.Size())
        {
            std::ostringstream os;
            os << "Kalman::Routers::NN::MLP: layer " << fWeights.size() << " has a ("
               << weight.Rows() << ", " << weight.Cols() << ") weight but a "
               << bias.Size() << "-element bias.";
            throw std::runtime_error(os.str());
        }
        if (!fWeights.empty() && weight.Cols() != fWeights.back().Rows())
        {
            std::ostringstream os;
            os << "Kalman::Routers::NN::MLP: layer " << fWeights.size() << " takes "
               << weight.Cols() << " inputs but the previous layer produces "
               << fWeights.back().Rows() << ".";
            throw std::runtime_error(os.str());
        }

        fWeights.push_back(weight);
        fBiases.push_back(bias);
    }

    void MLP::Clear()
    {
        fWeights.clear();
        fBiases.clear();
    }

    Vector MLP::operator()(const Vector &features) const
    {
        if (fWeights.empty())
            throw std::runtime_error("Kalman::Routers::NN::MLP: no layers have been added.");
        if (features.Size() != fWeights.front().Cols())
        {
            std::ostringstream os;
            os << "Kalman::Routers::NN::MLP: expected " << fWeights.front().Cols()
               << " features, got " << features.Size() << ".";
            throw std::runtime_error(os.str());
        }

        Vector x = features;
        for (std::size_t l = 0; l < fWeights.size(); ++l)
        {
            x = fWeights[l] * x;
            x += fBiases[l];

            // tanh between layers, nothing after the last: the head is the softmax
            // below. Same arrangement as nn/mlp.py.
            if (l + 1 < fWeights.size())
                for (std::size_t i = 0; i < x.Size(); ++i) x(i) = std::tanh(x(i));
        }

        // Softmax, shifted by the maximum. The shift is not cosmetic: logits of a
        // few hundred are ordinary for a confident policy, and exp() of those
        // overflows to inf, which then divides to NaN.
        double hi = x(0);
        for (std::size_t i = 1; i < x.Size(); ++i) hi = std::max(hi, x(i));

        double sum = 0.;
        for (std::size_t i = 0; i < x.Size(); ++i)
        {
            x(i) = std::exp(x(i) - hi);
            sum += x(i);
        }
        for (std::size_t i = 0; i < x.Size(); ++i) x(i) /= sum;
        return x;
    }

    void MLP::Load(const std::string &path)
    {
        std::ifstream in(path);
        if (!in)
            throw std::runtime_error("Kalman::Routers::NN::MLP: cannot open '" + path + "'.");

        Clear();

        std::size_t nLayers = 0;
        if (!(in >> nLayers))
            throw std::runtime_error("Kalman::Routers::NN::MLP: '" + path +
                                     "' does not start with a layer count.");

        for (std::size_t l = 0; l < nLayers; ++l)
        {
            std::size_t rows = 0, cols = 0;
            if (!(in >> rows >> cols))
                throw std::runtime_error("Kalman::Routers::NN::MLP: truncated shape for layer " +
                                         std::to_string(l) + " in '" + path + "'.");

            Matrix W(rows, cols);
            for (std::size_t r = 0; r < rows; ++r)
                for (std::size_t c = 0; c < cols; ++c)
                    if (!(in >> W(r, c)))
                        throw std::runtime_error(
                            "Kalman::Routers::NN::MLP: truncated weights for layer " +
                            std::to_string(l) + " in '" + path + "'.");

            Vector b(rows);
            for (std::size_t r = 0; r < rows; ++r)
                if (!(in >> b(r)))
                    throw std::runtime_error(
                        "Kalman::Routers::NN::MLP: truncated bias for layer " +
                        std::to_string(l) + " in '" + path + "'.");

            AddLayer(W, b);
        }
    }
}
