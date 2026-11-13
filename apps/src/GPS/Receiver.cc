/**
 * *********************************************
 *
 * \file Receiver.cc
 * \brief Implementation of the Kalman::Apps::GPS::Receiver class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/GPS/Receiver.h"

#include "Kalman/Factory.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

namespace Kalman::Apps::GPS
{
    namespace
    {
        const double kPi = 3.14159265358979323846;
        const char *const kElements[] = {"semi_major_axis", "eccentricity", "inclination", "raan", "arg_perigee",
                                         "mean_anomaly"};

        // a (3x3) = b (3x3) * c (3x3), row-major.
        std::array<double, 9> Multiply(const std::array<double, 9> &b, const std::array<double, 9> &c)
        {
            std::array<double, 9> a{};
            for (std::size_t i = 0; i < 3; ++i)
                for (std::size_t j = 0; j < 3; ++j)
                {
                    double s = 0.;
                    for (std::size_t k = 0; k < 3; ++k) s += b[3 * i + k] * c[3 * k + j];
                    a[3 * i + j] = s;
                }
            return a;
        }

        std::array<double, 9> Rz(double angle)
        {
            const double c = std::cos(angle), s = std::sin(angle);
            return {c, -s, 0., s, c, 0., 0., 0., 1.};
        }

        std::array<double, 9> Rx(double angle)
        {
            const double c = std::cos(angle), s = std::sin(angle);
            return {1., 0., 0., 0., c, -s, 0., s, c};
        }
    }

    Receiver::Receiver() : Kalman::Apps::Model("gps", 5, 8)
    {
        SetSampleRate(1.);
        DeclareState({{"x", "m"}, {"y", "m"}, {"z", "m"}, {"clock_bias", "m"}, {"clock_drift", "m/s"}});

        DeclareParameter("pseudorange_uncertainty", 1.);
        DeclareParameter("process_uncertainty", 0.2);
        DeclareParameter("clock_drift_uncertainty", 1e-3);

        // The receiver: 45 degrees north, 10 degrees east, on a spherical Earth.
        const double lat = 45. * kPi / 180., lon = 10. * kPi / 180.;
        const double up[3] = {std::cos(lat) * std::cos(lon), std::cos(lat) * std::sin(lon), std::sin(lat)};
        Vector x0(5, 0.);
        for (std::size_t i = 0; i < 3; ++i) x0(i) = kEarthRadius * up[i];
        SetInitialState(x0);
        Matrix P0(5, 5, 0.);
        for (std::size_t i = 0; i < 4; ++i) P0(i, i) = 100. * 100.;
        P0(4, 4) = 1.;
        SetInitialCovariance(P0);

        // The default constellation: 6 planes of 4 satellites at 55 degrees, as
        // the Python class lays them out, with one eccentricity for all (Python
        // draws them), of which the 8 highest above the receiver at t = 0 are
        // kept. A Python instance pushes its own.
        std::vector<Satellite> pool;
        std::vector<std::array<double, 6>> elements;
        for (int plane = 0; plane < 6; ++plane)
            for (int slot = 0; slot < 4; ++slot)
                elements.push_back({kOrbitRadius, 0.01, 55. * kPi / 180., 2. * kPi * plane / 6., 0.,
                                    2. * kPi * slot / 4. + 0.5 * plane});
        for (std::size_t k = 0; k < 6; ++k)
        {
            std::vector<double> column;
            for (const auto &e : elements) column.push_back(e[k]);
            DeclareArray(kElements[k], column);
        }
        ParametersChanged();                      // the whole pool, to place it

        std::vector<double> elevation(fSatellites.size());
        for (std::size_t i = 0; i < fSatellites.size(); ++i)
        {
            double s[3];
            Position(i, 0., s);
            double dot = 0., norm = 0.;
            for (std::size_t k = 0; k < 3; ++k)
            {
                const double los = s[k] - x0(k);
                dot += los * up[k];
                norm += los * los;
            }
            elevation[i] = dot / std::sqrt(norm);
        }
        std::vector<std::size_t> order(fSatellites.size());
        std::iota(order.begin(), order.end(), std::size_t(0));
        std::stable_sort(order.begin(), order.end(),
                         [&elevation](std::size_t a, std::size_t b) { return elevation[a] > elevation[b]; });
        for (std::size_t k = 0; k < 6; ++k)
        {
            std::vector<double> column;
            for (std::size_t i = 0; i < 8; ++i) column.push_back(elements[order[i]][k]);
            fArrays[kElements[k]] = column;
        }
        ParametersChanged();

        // db/dt = bdot is the only coupling; the position and the drift rate are driven by white noise.
        Matrix A(5, 5, 0.);
        A(3, 4) = 1.;
        SetGenerator(A);
        SetDifferentiator(DifferentiatorKind::Analytic);   // Jacobian() and JacobianH() are formulas
        SetStructure(StructureKind::Maps);        // h is code: a file refers to the model
    }

    void Receiver::ParametersChanged()
    {
        const std::size_t count = Array(kElements[0]).size();
        fConsistent = true;
        for (const char *name : kElements) fConsistent = fConsistent && Array(name).size() == count;
        if (!fConsistent) return;                 // said by CheckShapes(), once all six are in

        const auto &a = Array("semi_major_axis"), &e = Array("eccentricity"), &inc = Array("inclination"),
                   &raan = Array("raan"), &argp = Array("arg_perigee"), &m0 = Array("mean_anomaly");
        fSatellites.resize(count);
        for (std::size_t i = 0; i < count; ++i)
        {
            Satellite &s = fSatellites[i];
            s.a = a[i];
            s.e = e[i];
            s.m0 = m0[i];
            s.meanMotion = std::sqrt(kMuEarth / (a[i] * a[i] * a[i]));
            s.rotation = Multiply(Multiply(Rz(raan[i]), Rx(inc[i])), Rz(argp[i]));
        }

        if (count != NMeasure() || MeasurementComponents().empty() || MeasurementComponents()[0].name != "pseudorange_0")
        {
            ResizeMeasurement(count);
            std::vector<Component> channels;
            for (std::size_t i = 0; i < count; ++i) channels.push_back({"pseudorange_" + std::to_string(i), "m"});
            if (count > 0) DeclareMeasurement(channels);
        }

        const double sigma = Parameter("pseudorange_uncertainty");
        Matrix R(count, count, 0.);
        for (std::size_t i = 0; i < count; ++i) R(i, i) = sigma * sigma;
        SetR(R);

        const double qp = Parameter("process_uncertainty") * Parameter("process_uncertainty");
        const double qd = Parameter("clock_drift_uncertainty") * Parameter("clock_drift_uncertainty");
        Matrix Qc(4, 4, 0.), Lc(5, 4, 0.);
        Qc(0, 0) = Qc(1, 1) = Qc(2, 2) = qp;
        Qc(3, 3) = qd;
        Lc(0, 0) = Lc(1, 1) = Lc(2, 2) = 1.;
        Lc(4, 3) = 1.;
        SetNoiseIntensity(Qc, Lc);
    }

    void Receiver::CheckShapes() const
    {
        if (!fConsistent)
        {
            std::string sizes;
            for (const char *name : kElements)
                sizes += std::string(sizes.empty() ? "" : ", ") + name + " " + std::to_string(Array(name).size());
            throw std::runtime_error("Kalman::Apps::GPS::Receiver: the six orbit arrays need one entry per satellite "
                                     "each (" + sizes + ").");
        }
        Kalman::Apps::Model::CheckShapes();
    }

    void Receiver::Position(std::size_t i, double t, double *position) const
    {
        const Satellite &s = fSatellites[i];
        // Kepler's equation E - e sin E = M, by Newton from E = M, M wrapped to [-pi, pi).
        double M = std::fmod(s.m0 + s.meanMotion * t + kPi, 2. * kPi);
        if (M < 0.) M += 2. * kPi;
        M -= kPi;
        double E = M;
        for (int k = 0; k < 12; ++k) E = E - (E - s.e * std::sin(E) - M) / (1. - s.e * std::cos(E));

        const double nu = 2. * std::atan2(std::sqrt(1. + s.e) * std::sin(E / 2.), std::sqrt(1. - s.e) * std::cos(E / 2.));
        const double r = s.a * (1. - s.e * std::cos(E));
        const double px = r * std::cos(nu), py = r * std::sin(nu);
        for (std::size_t k = 0; k < 3; ++k) position[k] = s.rotation[3 * k] * px + s.rotation[3 * k + 1] * py;
    }

    Vector Receiver::h(const Vector &x, double t) const
    {
        Vector z(NMeasure());
        for (std::size_t i = 0; i < fSatellites.size(); ++i)
        {
            double s[3];
            Position(i, t, s);
            double range = 0.;
            for (std::size_t k = 0; k < 3; ++k) range += (s[k] - x(k)) * (s[k] - x(k));
            z(i) = std::sqrt(range) + x(3);
        }
        return z;
    }

    Matrix Receiver::Jacobian(const Vector &, const Vector &, double t) const { return F(t); }

    Matrix Receiver::JacobianH(const Vector &x, double t) const
    {
        Matrix H(NMeasure(), 5, 0.);
        for (std::size_t i = 0; i < fSatellites.size(); ++i)
        {
            double s[3], range = 0.;
            Position(i, t, s);
            for (std::size_t k = 0; k < 3; ++k) range += (x(k) - s[k]) * (x(k) - s[k]);
            range = std::sqrt(range);
            for (std::size_t k = 0; k < 3; ++k) H(i, k) = (x(k) - s[k]) / range;
            H(i, 3) = 1.;
        }
        return H;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::GPS::Receiver, "gps");
