/**
 * *********************************************
 *
 * \file CKF.cc
 * \brief Implementation of the Kalman::Filters::CKF class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Filters/CKF.h"

#include <stdexcept>

namespace Kalman::Filters
{
    CKF::CKF(Model *model, double chi2Cut, std::size_t maxCandidates, std::size_t maxHoles,
             std::size_t minHits, double sharedHitFrac, double holePenalty)
        : fModel(model), fEngine(chi2Cut, maxCandidates, maxHoles, minHits, sharedHitFrac, holePenalty)
    {
        if (!fModel) throw std::runtime_error("Kalman::Filters::CKF: a model is required.");
        fModel->CheckShapes();
    }

    RecordCollection CKF::Find(const Tracking::Event &hits, const std::vector<double> &radii)
    {
        fTracks = fEngine.Find(*fModel, hits, radii);
        fSmoothed = false;
        return Records(StateFlavor::Posterior);
    }

    RecordCollection CKF::Smooth()
    {
        if (fTracks.empty() && !fSmoothed)
            throw std::runtime_error("Kalman::Filters::CKF::Smooth: run Find() first (or it found no track).");
        fTracks = fEngine.SmoothTracks(*fModel, fTracks);
        fSmoothed = true;
        return Records(StateFlavor::Smoothed);
    }

    RecordCollection CKF::Records(StateFlavor flavor) const
    {
        RecordCollection out(fModel->Name() + " tracks");
        out.Reserve(fTracks.size());
        for (const Tracking::Candidate &c : fTracks) out.Add(c.ToRecord(*fModel, flavor));
        return out;
    }
}
