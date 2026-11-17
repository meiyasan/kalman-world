/**
 * *********************************************
 *
 * \file Engines.h
 * \brief Every engine of the library.
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

#ifndef KALMAN_ENGINES_H
#define KALMAN_ENGINES_H

#include "Kalman/Engines/Linear.h"
#include "Kalman/Engines/Extended.h"
#include "Kalman/Engines/Unscented.h"
#include "Kalman/Engines/Particle.h"
#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Engines/Combinatorial.h"
#include "Kalman/Engines/Interacting.h"

#endif // KALMAN_ENGINES_H
