/**
 * *********************************************
 *
 * \file DriftChamber.cc
 * \brief Implementation of the Kalman::Apps::Tracking::DriftChamber class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Tracking/DriftChamber.h"

#include "Kalman/Factory.h"

#include <cmath>
#include <limits>

namespace Kalman::Apps::Tracking
{
    DriftChamber::DriftChamber() : Kalman::Apps::Model("drift-chamber", 4, 2)
    {
        SetDefaultEngine(EngineKind::Combinatorial);
        SetSampleRate(1.);
        DeclareState({{"x", "cm"}, {"y", "cm"}, {"px", "GeV/c"}, {"py", "GeV/c"}});
        DeclareMeasurement({{"x", "cm"}, {"y", "cm"}});

        DeclareParameter("Bz", 1.5);
        DeclareParameter("q", 1.);
        DeclareParameter("sigma_hit", 0.013);
        DeclareParameter("p_seed", 1.);
        DeclareParameter("step", 0.5);
        std::vector<double> radii;
        for (int layer = 0; layer < 8; ++layer) radii.push_back(20. + 12. * layer);   // 20 .. 104 cm
        DeclareArray("radii", radii);
        ParametersChanged();

        // The Python class's defaults; a Python instance pushes its own.
        Matrix Q(4, 4, 0.);
        Q(0, 0) = Q(1, 1) = 1e-6;
        Q(2, 2) = Q(3, 3) = 1e-4;
        SetQ(Q);
        Matrix H(2, 4, 0.);
        H(0, 0) = H(1, 1) = 1.;
        SetH(H);
    }

    void DriftChamber::ParametersChanged()
    {
        fBz = Parameter("Bz");
        fCharge = Parameter("q");
        fSigmaHit = Parameter("sigma_hit");
        fPSeed = Parameter("p_seed");
        fStep = Parameter("step");
        Matrix R(2, 2, 0.);
        R(0, 0) = R(1, 1) = fSigmaHit * fSigmaHit;
        SetR(R);
    }

    void DriftChamber::SeedState(const Kalman::Tracking::Hit &hit, Vector &x0, Matrix &P0) const
    {
        double r = std::hypot(hit(0), hit(1));
        if (r == 0.) r = 1.;
        x0 = Vector(4);
        x0(0) = hit(0);
        x0(1) = hit(1);
        x0(2) = fPSeed * (hit(0) / r);
        x0(3) = fPSeed * (hit(1) / r);
        P0 = Matrix(4, 4, 0.);
        P0(0, 0) = P0(1, 1) = fSigmaHit * fSigmaHit;
        P0(2, 2) = P0(3, 3) = fPSeed * fPSeed;
    }

    // d(state)/d(arc length): unit speed, the momentum turning at kKappa q Bz / |p|.
    void DriftChamber::Rhs(const double *s, double *d) const
    {
        const double p = std::hypot(s[2], s[3]);
        if (p == 0.)
        {
            d[0] = d[1] = d[2] = d[3] = 0.;
            return;
        }
        const double c = kKappa * fCharge * fBz / p;
        d[0] = s[2] / p;
        d[1] = s[3] / p;
        d[2] = c * s[3];
        d[3] = -c * s[2];
    }

    // Written in the order the Python model sums it.
    void DriftChamber::RK4(const double *s, double dl, double *out) const
    {
        double k1[4], k2[4], k3[4], k4[4], y[4];
        Rhs(s, k1);
        for (int i = 0; i < 4; ++i) y[i] = s[i] + 0.5 * dl * k1[i];
        Rhs(y, k2);
        for (int i = 0; i < 4; ++i) y[i] = s[i] + 0.5 * dl * k2[i];
        Rhs(y, k3);
        for (int i = 0; i < 4; ++i) y[i] = s[i] + dl * k3[i];
        Rhs(y, k4);
        for (int i = 0; i < 4; ++i) out[i] = s[i] + dl / 6.0 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    }

    bool DriftChamber::PropagateToRadius(const Vector &state, double rTarget, Vector &out) const
    {
        double s[4] = {state(0), state(1), state(2), state(3)}, next[4];
        const double r0 = std::hypot(s[0], s[1]);
        out = Vector(4);
        if (rTarget <= r0 + 1e-12)
        {
            for (std::size_t i = 0; i < 4; ++i) out(i) = s[i];
            return true;
        }
        const double budget = 4.0 * (rTarget - r0) + 20.0;
        double travelled = 0.;
        while (travelled < budget)
        {
            RK4(s, fStep, next);
            if (std::hypot(next[0], next[1]) >= rTarget)
            {
                double lo = 0., hi = fStep, mid[4];           // bisect the last sub-step onto rTarget
                for (int i = 0; i < 40; ++i)
                {
                    const double half = 0.5 * (lo + hi);
                    RK4(s, half, mid);
                    if (std::hypot(mid[0], mid[1]) >= rTarget) hi = half;
                    else lo = half;
                }
                RK4(s, hi, next);
                for (std::size_t i = 0; i < 4; ++i) out(i) = next[i];
                return true;
            }
            for (int i = 0; i < 4; ++i) s[i] = next[i];
            travelled += fStep;
        }
        return false;
    }

    Vector DriftChamber::f(const Vector &x, const Vector &, double t) const
    {
        Vector out;
        if (PropagateToRadius(x, t, out)) return out;
        return Vector(4, std::numeric_limits<double>::quiet_NaN());
    }

    Vector DriftChamber::h(const Vector &x, double) const
    {
        Vector z(2);
        z(0) = x(0);
        z(1) = x(1);
        return z;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Tracking::DriftChamber, "drift-chamber");
