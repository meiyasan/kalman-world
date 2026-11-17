/**
 * *********************************************
 *
 * \file Routers.h
 * \brief Every routing decision of the library.
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

#ifndef KALMAN_ROUTERS_H
#define KALMAN_ROUTERS_H

#include "Kalman/Routers/Base.h"
#include "Kalman/Routers/Bayesian.h"
#include "Kalman/Routers/Threshold.h"
#include "Kalman/Routers/Hybrid.h"
#include "Kalman/Routers/NN.h"
#include "Kalman/Routers/Neural.h"

#endif // KALMAN_ROUTERS_H
