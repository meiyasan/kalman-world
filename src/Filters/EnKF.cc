/**
 * *********************************************
 *
 * \file EnKF.cc
 * \brief Implementation of the Kalman::Filters::EnKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Filters/EnKF.h"

#include <sstream>
#include <stdexcept>

namespace Kalman::Filters
{
    EnKF::Cycles EnKF::Cycle(const std::vector<Vector> &zk, bool analyse)
    {
        Model &model = *GetModel();
        if (zk.empty()) throw std::runtime_error("Kalman::Filters::EnKF::Cycle: no measurements given.");
        const std::size_t m = model.NMeasure();
        Reset();

        Engines::Ensemble &engine = Members();
        const double dt = model.DeltaT();
        double t = model.T0();

        Cycles out;
        out.forecast.reserve(zk.size());
        out.analysis.reserve(zk.size());
        out.spread.reserve(zk.size());
        for (std::size_t k = 0; k < zk.size(); ++k)
        {
            if (zk[k].Size() != m)
            {
                std::ostringstream os;
                os << "Kalman::Filters::EnKF::Cycle: measurement " << k << " has " << zk[k].Size()
                   << " channels, expected " << m << ".";
                throw std::runtime_error(os.str());
            }
            out.forecast.push_back(engine.Forecast(model, nullptr, t));
            out.analysis.push_back(analyse ? engine.Analyze(model, zk[k], t) : out.forecast.back());
            out.spread.push_back(engine.Spread());
            t += dt;
        }
        return out;
    }
}
