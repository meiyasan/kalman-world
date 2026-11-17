/**
 * *********************************************
 *
 * \file Models.h
 * \brief Every model class of the library, the operations' among them.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Every directory `Kalman/X/` has the header `Kalman/X.h` beside it, which
 * includes all of it: `#include <Kalman/X.h>` for the whole, `#include <Kalman/X/Name.h>`
 * for one.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_H
#define KALMAN_MODELS_H

#include "Kalman/Models/Linear.h"
#include "Kalman/Models/Extended.h"
#include "Kalman/Models/Callback.h"
#include "Kalman/Models/Tabulated.h"
#include "Kalman/Models/Joint.h"
#include "Kalman/Models/Fusion.h"
#include "Kalman/Models/Lag.h"
#include "Kalman/Models/Composite.h"
#include "Kalman/Models/Router.h"

#endif // KALMAN_MODELS_H
