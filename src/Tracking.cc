/**
 * *********************************************
 *
 * \file Tracking.cc
 * \brief Implementation of the tracking value types
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Tracking.h"
#include "Kalman/Utils.h"

namespace Kalman::Tracking
{
    Instrument Detector::Layers(const Sensor &reading) const
    {
        Instrument detector;
        detector.name = reading.name;
        detector.sequential = true;
        const std::vector<double> radii = Radii();
        for (std::size_t i = 0; i < radii.size(); ++i)
        {
            Sensor layer = reading;
            layer.name = "layer " + std::to_string(i);
            layer.instrument = detector.name;
            layer.station = radii[i];
            detector.sensors.push_back(std::move(layer));
        }
        return detector;
    }

    Record Candidate::ToRecord(const Model &model, StateFlavor flavor) const
    {
        Record record(model.Name(), flavor);
        record.Reserve(NHits());
        for (std::size_t k = 0; k < NHits(); ++k) record.Add(radii[k], xk[k], Pk[k], zk[k]);
        return record;
    }

    double GateContext::Score(const Vector &x, const Hit &z, Vector &xPost) const
    {
        Vector y = z;
        y -= zpred;
        xPost = x;
        xPost += K * y;
        return Utils::Mahalanobis(y, Sinv);
    }
}
