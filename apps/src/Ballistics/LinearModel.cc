/**
 * *********************************************
 *
 * \file LinearModel.cc
 * \brief Implementation of the Kalman::Apps::Ballistics::LinearModel class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Ballistics/LinearModel.h"

#include <Kalman/Factory.h>

#include <stdexcept>

namespace Kalman::Apps::Ballistics
{
    LinearModel::LinearModel(double fs, double measurement_uncertainty, double process_uncertainty)
        : Kalman::Models::Linear("ballistic-linear", 3, 1), fSigma(measurement_uncertainty), fQ(process_uncertainty)
    {
        if (!(fs > 0.)) throw std::invalid_argument("Kalman::Apps::Ballistics::LinearModel: fs must be positive.");
        if (!(fSigma > 0.) || !(fQ > 0.))
            throw std::invalid_argument("Kalman::Apps::Ballistics::LinearModel: the measurement and the process "
                                        "uncertainty must be positive.");
        DeclareState({{"position", "m"}, {"speed", "m/s"}, {"acceleration", "m/s^2"}});
        DeclareMeasurement({{"position", "m"}});
        SetSampleRate(fs);
        Build();

        Matrix P0(3, 3, 0.);
        P0(0, 0) = fSigma * fSigma;
        P0(1, 1) = fSigma * fSigma / 100.;
        P0(2, 2) = fQ * fQ;
        SetInitialCovariance(P0);
    }

    void LinearModel::Build()
    {
        const double dt = DeltaT(), q2 = fQ * fQ;

        // exp(A dt) for A = [0 1 0; 0 0 1; 0 0 0]
        Matrix F = Matrix::Identity(3);
        F(0, 1) = dt;
        F(0, 2) = dt * dt / 2.;
        F(1, 2) = dt;
        SetF(F);

        Matrix H(1, 3, 0.);
        H(0, 0) = 1.;
        SetH(H);

        // The integral over one sample of exp(A s) L q^2 L^T exp(A^T s), with L = (0, 1, 0):
        // exp(A s) L = (s, 1, 0), so the acceleration row and column are zero.
        Matrix Q(3, 3, 0.);
        Q(0, 0) = q2 * dt * dt * dt / 3.;
        Q(0, 1) = Q(1, 0) = q2 * dt * dt / 2.;
        Q(1, 1) = q2 * dt;
        SetQ(Q);

        Matrix R(1, 1, 0.);
        R(0, 0) = fSigma * fSigma;
        SetR(R);
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Ballistics::LinearModel, "ballistic-linear");
