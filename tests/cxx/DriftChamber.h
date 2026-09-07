/**
 * \file DriftChamber.h
 * \brief A transverse charged-particle tracker on concentric layers — the CKF test fixture.
 *
 * A pedagogical Belle II–style chamber: a charged particle bends in a uniform
 * axial field `Bz` and is sampled on concentric cylindrical layers, seen in the
 * transverse x–y plane. The trajectory is a circular arc; a hit is the (x, y)
 * crossing of a layer. Each layer carries several hits (signal + background), so
 * track finding is combinatorial.
 *
 *   state  x = [x, y, px, py]     (|p| is conserved, curvature q Bz / |p| is constant)
 *   f(x, u, t = r)                propagates the arc (RK4 in arc length) to radius r
 *   h(x) = [x, y]                 the crossing position, R = sigma^2 I
 *
 * The model is also a Kalman::Tracking::Detector (radii, seeding) and generates
 * its own synthetic events with the same propagator the filter uses. Mirrors
 * `modules/apps/tracking/driftchamber.py`; it is a fixture, not a catalogue —
 * physics lives with the application.
 */
#ifndef KALMAN_TESTS_DRIFTCHAMBER_H
#define KALMAN_TESTS_DRIFTCHAMBER_H

#include "Kalman/Models/Extended.h"
#include "Kalman/Tracking.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace Fixture
{
    struct TrackingEvent
    {
        Kalman::Tracking::Event hits;                 ///< per layer, the (x, y) hits
        std::vector<std::vector<long>> truth;         ///< per hit, the signal track id or -1
        std::vector<std::vector<Kalman::Vector>> tracks;   ///< per signal track, the true states per crossed layer
        std::vector<double> radii;

        std::size_t NTracks() const { return tracks.size(); }
    };

    class DriftChamber : public Kalman::Models::Extended, public Kalman::Tracking::Detector
    {
        private:
            double fBz, fQ, fSigmaHit, fPSeed, fStep;
            std::vector<double> fRadii;

        public:
            DriftChamber(std::size_t nLayers = 8, double Bz = 0.05, double rInner = 1., double rStep = 1.,
                         double q = 1., double sigmaHit = 0.05, double pSeed = 1.,
                         double qPos = 1e-6, double qMom = 1e-4, double step = 0.02)
                : Kalman::Models::Extended("CDC", 4, 2), fBz(Bz), fQ(q), fSigmaHit(sigmaHit),
                  fPSeed(pSeed), fStep(step)
            {
                for (std::size_t l = 0; l < nLayers; ++l) fRadii.push_back(rInner + rStep * static_cast<double>(l));
                SetDefaultEngine(Kalman::EngineKind::Combinatorial);
                SetSampleRate(1.);

                Kalman::Matrix Q(4, 4, 0.);
                Q(0, 0) = qPos; Q(1, 1) = qPos; Q(2, 2) = qMom; Q(3, 3) = qMom;
                SetQ(Q);
                Kalman::Matrix R = Kalman::Matrix::Identity(2);
                R *= sigmaHit * sigmaHit;
                SetR(R);
                Kalman::Matrix H(2, 4, 0.);
                H(0, 0) = 1.; H(1, 1) = 1.;
                SetH(H);
            }

            const char *Title() const override { return "Drift Chamber (transverse tracker)"; }
            double SigmaHit() const { return fSigmaHit; }

            // --- Tracking::Detector ------------------------------------------------
            std::vector<double> Radii() const override { return fRadii; }

            void SeedState(const Kalman::Vector &hit, Kalman::Vector &x0, Kalman::Matrix &P0) const override
            {
                double r = std::hypot(hit(0), hit(1));
                if (r == 0.) r = 1.;
                x0 = Kalman::Vector(4);
                x0(0) = hit(0); x0(1) = hit(1);
                x0(2) = fPSeed * hit(0) / r; x0(3) = fPSeed * hit(1) / r;
                P0 = Kalman::Matrix(4, 4, 0.);
                P0(0, 0) = P0(1, 1) = fSigmaHit * fSigmaHit;
                P0(2, 2) = P0(3, 3) = fPSeed * fPSeed;
            }

            // --- dynamics: a circular arc, propagated by radius ------------------------
            Kalman::Vector Rhs(const Kalman::Vector &s) const
            {
                const double p = std::hypot(s(2), s(3));
                Kalman::Vector d(4, 0.);
                if (p == 0.) return d;
                const double c = fQ * fBz / p;
                d(0) = s(2) / p; d(1) = s(3) / p; d(2) = c * s(3); d(3) = -c * s(2);
                return d;
            }

            Kalman::Vector RK4(const Kalman::Vector &s, double dl) const
            {
                const Kalman::Vector k1 = Rhs(s);
                Kalman::Vector s2 = s; s2 += 0.5 * dl * k1;
                const Kalman::Vector k2 = Rhs(s2);
                Kalman::Vector s3 = s; s3 += 0.5 * dl * k2;
                const Kalman::Vector k3 = Rhs(s3);
                Kalman::Vector s4 = s; s4 += dl * k3;
                const Kalman::Vector k4 = Rhs(s4);
                Kalman::Vector out = s;
                out += (dl / 6.) * (k1 + 2. * k2 + 2. * k3 + k4);
                return out;
            }

            /// The state at the first outward crossing of \p rTarget, or false if the track curls back.
            bool PropagateToRadius(Kalman::Vector s, double rTarget, Kalman::Vector &out) const
            {
                const double r0 = std::hypot(s(0), s(1));
                if (rTarget <= r0 + 1e-12) { out = s; return true; }
                const double budget = 4. * (rTarget - r0) + 20.;
                double travelled = 0.;
                while (travelled < budget)
                {
                    const Kalman::Vector next = RK4(s, fStep);
                    if (std::hypot(next(0), next(1)) >= rTarget)
                    {
                        double lo = 0., hi = fStep;               // bisect the last sub-step onto rTarget
                        for (int i = 0; i < 40; ++i)
                        {
                            const double mid = 0.5 * (lo + hi);
                            const Kalman::Vector sm = RK4(s, mid);
                            if (std::hypot(sm(0), sm(1)) >= rTarget) hi = mid; else lo = mid;
                        }
                        out = RK4(s, hi);
                        return true;
                    }
                    s = next;
                    travelled += fStep;
                }
                return false;
            }

            Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &, double t = 0.) const override
            {
                Kalman::Vector out;
                if (PropagateToRadius(x, t, out)) return out;
                return Kalman::Vector(4, std::numeric_limits<double>::quiet_NaN());
            }

            Kalman::Vector h(const Kalman::Vector &x, double = 0.) const override
            {
                Kalman::Vector z(2);
                z(0) = x(0); z(1) = x(1);
                return z;
            }

            // --- synthetic events, with this model's own propagator -------------------
            TrackingEvent GenerateEvent(std::size_t nTracks = 3, std::size_t nBackground = 4,
                                        double efficiency = 1., std::uint64_t seed = 0,
                                        double ptMin = 0.7, double ptMax = 1.5,
                                        double phiMin = 0.2, double phiMax = 1.3,
                                        double originSigma = 0.05) const
            {
                std::mt19937_64 rng(seed);
                std::uniform_real_distribution<double> uniform(0., 1.);
                std::normal_distribution<double> gauss(0., 1.);

                const std::size_t L = fRadii.size();
                TrackingEvent ev;
                ev.hits.assign(L, {});
                ev.truth.assign(L, {});
                ev.radii = fRadii;

                const double spread = (phiMax - phiMin) / static_cast<double>(std::max<std::size_t>(1, nTracks));
                for (std::size_t tid = 0; tid < nTracks; ++tid)
                {
                    const double base = nTracks > 1
                        ? phiMin + (phiMax - phiMin) * static_cast<double>(tid) / static_cast<double>(nTracks - 1)
                        : 0.5 * (phiMin + phiMax);
                    const double phi = base + (uniform(rng) * 0.3 - 0.15) * spread;
                    const double pt = ptMin + (ptMax - ptMin) * uniform(rng);
                    Kalman::Vector s(4);
                    s(0) = originSigma * gauss(rng); s(1) = originSigma * gauss(rng);
                    s(2) = pt * std::cos(phi); s(3) = pt * std::sin(phi);

                    std::vector<Kalman::Vector> states;
                    for (std::size_t l = 0; l < L; ++l)
                    {
                        Kalman::Vector next;
                        if (!PropagateToRadius(s, fRadii[l], next)) break;
                        s = next;
                        states.push_back(s);
                        if (uniform(rng) <= efficiency)
                        {
                            Kalman::Vector pos(2);
                            pos(0) = s(0) + fSigmaHit * gauss(rng);
                            pos(1) = s(1) + fSigmaHit * gauss(rng);
                            ev.hits[l].push_back(pos);
                            ev.truth[l].push_back(static_cast<long>(tid));
                        }
                    }
                    if (!states.empty()) ev.tracks.push_back(states);
                }

                for (std::size_t l = 0; l < L; ++l)
                    for (std::size_t b = 0; b < nBackground; ++b)
                    {
                        const double ang = (phiMin - 0.3) + (phiMax - phiMin + 0.6) * uniform(rng);
                        Kalman::Vector pos(2);
                        pos(0) = fRadii[l] * std::cos(ang) + fSigmaHit * gauss(rng);
                        pos(1) = fRadii[l] * std::sin(ang) + fSigmaHit * gauss(rng);
                        ev.hits[l].push_back(pos);
                        ev.truth[l].push_back(-1L);
                    }
                return ev;
            }
    };
}

#endif // KALMAN_TESTS_DRIFTCHAMBER_H
