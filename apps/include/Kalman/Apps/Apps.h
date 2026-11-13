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
 * physics lives in C++: each model here is the twin of one Python class in
 * `kalman.apps`, and gives the same answer.
 *
 * | C++ class | Registered name | Python class |
 * |---|---|---|
 * | Kalman::Apps::Weather::Storm | `storm`, `storm-track-only` | `StormModel` |
 * | Kalman::Apps::Weather::Tornado | `tornado-environment` | `TornadoEnvironmentModel` |
 * | Kalman::Apps::Ballistics::Ballistic | `ballistic`, `ballistic-order2` | `BallisticModel` |
 * | Kalman::Apps::Vehicles::CarSpeed | `car-speed` | `CarSpeedModel` |
 * | Kalman::Apps::Vehicles::CarLane | `car-lane` | `CarLaneModel` |
 * | Kalman::Apps::Astro::RocketLiftoff | `rocket-liftoff` | `RocketLiftoffModel` |
 * | Kalman::Apps::Astro::RocketAscent | `rocket-ascent` | `RocketAscentModel` |
 * | Kalman::Apps::Astro::RocketDescent | `rocket-descent`, `rocket-descent-control` | `RocketDescentModel` |
 * | Kalman::Apps::Astro::RocketLanding | `rocket-landing` | `RocketLandingModel` |
 *
 * From C++, include the header and use the class, or ask the factory by name:
 *
 * \code{.cpp}
 * #include <Kalman/Kalman.h>
 * #include <Kalman/Apps/Apps.h>
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
 * pass on the twin when this library is installed, and in Python otherwise.
 *
 * Every model derives from Kalman::Apps::Model and is configured through the
 * named parameters of Kalman::Model. Link `libKalmanApps` as well as
 * `libKalman`. The guide page \ref md_docs_216___cxx-_models-from-_python
 * shows how to write a model of your own.
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_APPS_H
#define KALMAN_APPS_APPS_H

#include "Kalman/Apps/Model.h"
#include "Kalman/Apps/Astro/RocketAscent.h"
#include "Kalman/Apps/Astro/RocketDescent.h"
#include "Kalman/Apps/Astro/RocketLanding.h"
#include "Kalman/Apps/Astro/RocketLiftoff.h"
#include "Kalman/Apps/Astro/Interplanetary.h"
#include "Kalman/Apps/Astro/TransLunar.h"
#include "Kalman/Apps/Ballistics/Ballistic.h"
#include "Kalman/Apps/GPS/Receiver.h"
#include "Kalman/Apps/Tracking/DriftChamber.h"
#include "Kalman/Apps/Vehicles/CarLane.h"
#include "Kalman/Apps/Vehicles/CarSpeed.h"
#include "Kalman/Apps/Weather/Storm.h"
#include "Kalman/Apps/Weather/Tornado.h"

#endif // KALMAN_APPS_APPS_H
