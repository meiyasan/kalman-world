/**
 * *********************************************
 *
 * \file Controllers.h
 * \brief Every controller of the library.
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

#ifndef KALMAN_CONTROLLERS_H
#define KALMAN_CONTROLLERS_H

#include "Kalman/Controllers/Proportional.h"
#include "Kalman/Controllers/PID.h"
#include "Kalman/Controllers/LQR.h"
#include "Kalman/Controllers/MPC.h"
#include "Kalman/Controllers/MPPI.h"

#endif // KALMAN_CONTROLLERS_H
