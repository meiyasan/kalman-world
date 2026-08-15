/**
 * *********************************************
 *
 * \file Sampler.cc
 * \brief Implementation of the Kalman::Sampler class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Sampler.h"
#include "Kalman/Utils.h"

#include <limits>
#include <stdexcept>

namespace Kalman
{
    Sampler::Sampler(Model *model, std::uint64_t seed) : fModel(model), fRandom(seed)
    {
        if (!fModel) throw std::runtime_error("Kalman::Sampler: a model is required.");
        fModel->CheckShapes();
    }

    Vector Sampler::Draw(const Matrix &factor)
    {
        if (factor.Empty()) return Vector();       // degenerate: draw no noise at all

        std::normal_distribution<double> gauss(0., 1.);
        Vector w(factor.Cols());
        for (std::size_t i = 0; i < w.Size(); ++i) w(i) = gauss(fRandom);
        return factor * w;
    }

    void Sampler::Generate(std::size_t nsteps, const Vector &x0)
    {
        if (nsteps == 0) throw std::runtime_error("Kalman::Sampler::Generate: nsteps must be positive.");

        fTruth.clear();
        fMeasure.clear();
        fTime.clear();
        fTruth.reserve(nsteps);
        fMeasure.reserve(nsteps);
        fTime.reserve(nsteps);

        Vector x = (x0.Size() == fModel->NState()) ? x0 : fModel->X0();
        const Vector u(fModel->NControl(), 0.);

        const double dt = fModel->DeltaT();
        double t = fModel->T0();

        // Q and R are evaluated per step so a time-varying model is sampled
        // correctly; the factorisation is redone only when the matrix changed.
        Matrix Qfac, Rfac;
        bool haveQ = false, haveR = false;
        double factoredAt = 0.;

        for (std::size_t k = 0; k < nsteps; ++k)
        {
            Matrix Q = fModel->Q(t);
            const Matrix L = fModel->L(t);
            if (!L.Empty()) Q = Utils::Similarity(L, Q);

            if (!haveQ || t != factoredAt) { Qfac = Utils::CholeskyFactor(Q); haveQ = true; }
            if (!haveR || t != factoredAt) { Rfac = Utils::CholeskyFactor(fModel->R(t)); haveR = true; }
            factoredAt = t;

            x = fModel->f(x, u, t);
            const Vector qw = Draw(Qfac);
            if (!qw.Empty()) x += qw;

            Vector z = fModel->h(x, t);
            const Vector rw = Draw(Rfac);
            if (!rw.Empty()) z += rw;

            fTruth.push_back(x);
            fMeasure.push_back(z);
            fTime.push_back(t);

            t += dt;
        }
    }

    void Sampler::DropChannel(std::size_t channel, double probability)
    {
        if (channel >= fModel->NMeasure())
            throw std::runtime_error("Kalman::Sampler::DropChannel: channel out of range.");

        std::uniform_real_distribution<double> uniform(0., 1.);
        for (Vector &z : fMeasure)
            if (uniform(fRandom) < probability)
                z(channel) = std::numeric_limits<double>::quiet_NaN();
    }
}
