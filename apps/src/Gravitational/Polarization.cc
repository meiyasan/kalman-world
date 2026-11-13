/**
 * *********************************************
 *
 * \file Polarization.cc
 * \brief Implementation of the Kalman::Apps::Gravitational::Polarization class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Gravitational/Polarization.h"

#include <Kalman/Factory.h>

#include <cmath>
#include <complex>

namespace Kalman::Apps::Gravitational
{
    namespace
    {
        using Complex = std::complex<double>;

        // The carrier a exp(i psi) and the two angles' factors, as the Python class forms them.
        struct Jones
        {
            Complex carrier, eip, eif;
            double a, c2, s2;

            explicit Jones(const Vector &x)
                : eip(std::cos(x(2)), std::sin(x(2))), eif(std::cos(x(6)), std::sin(x(6))),
                  a(std::exp(x(0))), c2(std::cos(x(4) / 2.)), s2(std::sin(x(4) / 2.))
            {
                carrier = a * eip;
            }
        };
    }

    Polarization::Polarization() : Kalman::Apps::Model("polarization", 8, 4)
    {
        DeclareParameter("delta", 0.5);
        DeclareParameter("sigma_a", 0.1);
        DeclareParameter("sigma_psi", 0.5);
        DeclareParameter("sigma_theta", 0.1);
        DeclareParameter("sigma_phi", 0.1);
        DeclareParameter("sigma_n", 1.);
        DeclareState({{"log_amplitude", ""}, {"log_amp_vel", "1/s"}, {"phase_pos", "rad"}, {"phase_vel", "rad/s"},
                      {"theta_pos", "rad"}, {"theta_vel", "rad/s"}, {"phi_pos", "rad"}, {"phi_vel", "rad/s"}});
        DeclareMeasurement({{"re1", ""}, {"im1", ""}, {"re2", ""}, {"im2", ""}});
        SetSampleRate(100.);
        ParametersChanged();
        SetStructure(StructureKind::Maps);        // h is code: a file refers to the model
    }

    void Polarization::ParametersChanged()
    {
        const double d = Parameter("delta");
        const char *const sigmas[] = {"sigma_a", "sigma_psi", "sigma_theta", "sigma_phi"};
        Matrix Q(8, 8, 0.);
        for (std::size_t i = 0; i < 4; ++i)
        {
            const double s2 = Parameter(sigmas[i]) * Parameter(sigmas[i]);
            Q(2 * i, 2 * i) = s2 * (d * d * d / 3.);
            Q(2 * i, 2 * i + 1) = Q(2 * i + 1, 2 * i) = s2 * (d * d / 2.);
            Q(2 * i + 1, 2 * i + 1) = s2 * d;
        }
        SetQ(Q);
        const double n = Parameter("sigma_n");
        Matrix R(4, 4, 0.);
        for (std::size_t i = 0; i < 4; ++i) R(i, i) = n * n;
        SetR(R);
    }

    Matrix Polarization::F(double) const
    {
        Matrix F(8, 8, 0.);
        for (std::size_t i = 0; i < 8; ++i) F(i, i) = 1.;
        for (std::size_t i = 0; i < 4; ++i) F(2 * i, 2 * i + 1) = DeltaT();
        return F;
    }

    Vector Polarization::f(const Vector &x, const Vector &, double) const
    {
        // [[1, dt], [0, 1]] on each pair, written as the matrix product the Python class takes.
        const double dt = DeltaT();
        Vector out(8);
        for (std::size_t i = 0; i < 4; ++i)
        {
            out(2 * i) = x(2 * i) + dt * x(2 * i + 1);
            out(2 * i + 1) = x(2 * i + 1);
        }
        return out;
    }

    Vector Polarization::h(const Vector &x, double) const
    {
        const Jones j(x);
        const Complex z1 = j.carrier * j.c2;
        const Complex z2 = j.carrier * j.s2 * j.eif;
        Vector z(4);
        z(0) = z1.real();
        z(1) = z1.imag();
        z(2) = z2.real();
        z(3) = z2.imag();
        return z;
    }

    Matrix Polarization::JacobianH(const Vector &x, double) const
    {
        const Jones j(x);
        const Complex i(0., 1.);
        const Complex aeip = j.a * j.eip;
        const Complex ia = i * j.a;
        const Complex columns[4][2] = {
            {aeip * j.c2, aeip * j.s2 * j.eif},                                       // log a
            {ia * j.eip * j.c2, ia * j.eip * j.s2 * j.eif},                           // psi
            {-0.5 * j.a * j.eip * j.s2, 0.5 * j.a * j.eip * j.c2 * j.eif},            // theta
            {Complex(0., 0.), ia * j.eip * j.s2 * j.eif}};                            // phi
        Matrix H(4, 8, 0.);
        for (std::size_t k = 0; k < 4; ++k)
        {
            const std::size_t col = 2 * k;
            H(0, col) = columns[k][0].real();
            H(1, col) = columns[k][0].imag();
            H(2, col) = columns[k][1].real();
            H(3, col) = columns[k][1].imag();
        }
        return H;
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Gravitational::Polarization, "polarization");
