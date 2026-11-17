/**
 * *********************************************
 *
 * \file Apps.h
 * \brief Umbrella header for libKalmanApps
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \page libkalmanapps libKalmanApps — the application models
 *
 * libKalman ships the contracts and no physics. libKalmanApps is where the
 * physics lives in C++: each model here is the counterpart of one Python class in
 * `kalman.apps`, and gives the same answer.
 *
 * | C++ class | Registered name | Python class |
 * |---|---|---|
 * | Kalman::Apps::Weather::Storm | `storm`, `storm-track-only` | `StormModel` |
 * | Kalman::Apps::Weather::Tornado | `tornado-environment` | `TornadoEnvironmentModel` |
 * | Kalman::Apps::Ballistics::ExtendedModel | `ballistic`, `ballistic-order2` | `BallisticModel` |
 * | Kalman::Apps::Ballistics::LinearModel | `ballistic-linear` | `BallisticLinearModel` |
 * | Kalman::Apps::Vehicles::CarSpeed | `car-speed` | `CarSpeedModel` |
 * | Kalman::Apps::Vehicles::CarLane | `car-lane` | `CarLaneModel` |
 * | Kalman::Apps::Aerospace::RocketLiftoff | `rocket-liftoff` | `RocketLiftoffModel` |
 * | Kalman::Apps::Aerospace::RocketAscent | `rocket-ascent` | `RocketAscentModel` |
 * | Kalman::Apps::Aerospace::RocketDescent | `rocket-descent`, `rocket-descent-control` | `RocketDescentModel` |
 * | Kalman::Apps::Aerospace::RocketLanding | `rocket-landing` | `RocketLandingModel` |
 * | Kalman::Apps::Gravitational::Polarization | `polarization` | `PolarizationModel` |
 * | Kalman::Apps::Weather::KolmogorovFlow | `kolmogorov-flow` | `KolmogorovFlowModel` (ensemble engine) |
 * | Kalman::Apps::Weather::BarotropicFlow | `barotropic-flow` | `BarotropicFlowModel` (ensemble engine) |
 *
 * From C++, include the header and use the class, or ask the factory by name:
 *
 * \code{.cpp}
 * #include <Kalman/Kalman.h>
 * #include <Kalman/Apps.h>                       // or one domain: <Kalman/Apps/Weather.h>
 *
 * Kalman::Apps::Weather::Storm storm;
 * storm.SetParameter("omega", -0.022);            // a recurving storm
 * Kalman::Filters::EKF ekf(&storm);
 * auto record = ekf.Forward(fixes);
 *
 * auto model = Kalman::Factory::Instance().Create("ballistic");   // by name
 * \endcode
 *
 * From Python nothing changes: `StormModel("TC").filter(fixes)` runs its whole
 * pass on the counterpart when this library is installed, and in Python otherwise.
 *
 * Every model derives from Kalman::Apps::Model and is configured through the
 * named parameters of Kalman::Model. Link `libKalmanApps` as well as
 * `libKalman`. The guide page \ref md_docs_216___cxx-_models-from-_python
 * shows how to write a model of your own.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_H
#define KALMAN_APPS_H

#include "Kalman/Apps/Model.h"
#include "Kalman/Apps/Aerospace.h"
#include "Kalman/Apps/Astrophysics.h"
#include "Kalman/Apps/Ballistics.h"
#include "Kalman/Apps/GPS.h"
#include "Kalman/Apps/Gravitational.h"
#include "Kalman/Apps/Tracking.h"
#include "Kalman/Apps/Vehicles.h"
#include "Kalman/Apps/Weather.h"

#endif // KALMAN_APPS_H
