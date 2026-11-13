/**
 * *********************************************
 *
 * \file BarotropicFlow.cc
 * \brief Implementation of the Kalman::Apps::Weather::BarotropicFlow class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Apps/Weather/BarotropicFlow.h"

#include <Kalman/Factory.h>

#include <cmath>

namespace Kalman::Apps::Weather
{
    BarotropicFlow::BarotropicFlow() : KolmogorovFlow("barotropic-flow")
    {
        // barotropic.py: beta_at(50) and box_size(20, 20, 50).
        constexpr double kDeg = 3.14159265358979323846 / 180.;
        const double latitude = 50. * kDeg;
        const double beta = 2. * kOmega * std::cos(latitude) / kEarthRadius;
        const double ly = 20. * kDeg * kEarthRadius;
        const double lx = 20. * kDeg * kEarthRadius * std::cos(latitude);
        fUnit = "1/s";
        Configure(2.0e4, beta, 60., 60, ly, lx, false);
    }
}

KALMAN_REGISTER_MODEL(Kalman::Apps::Weather::BarotropicFlow, "barotropic-flow");
