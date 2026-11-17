/**
 * *********************************************
 *
 * \file Filters.h
 * \brief Every filter of the library.
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

#ifndef KALMAN_FILTERS_H
#define KALMAN_FILTERS_H

#include "Kalman/Filters/KF.h"
#include "Kalman/Filters/EKF.h"
#include "Kalman/Filters/UKF.h"
#include "Kalman/Filters/PF.h"
#include "Kalman/Filters/EnKF.h"
#include "Kalman/Filters/CKF.h"

#endif // KALMAN_FILTERS_H
