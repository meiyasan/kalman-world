/**
 * *********************************************
 *
 * \file NN.h
 * \brief The policies a neural router can hold that need no runtime.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Every directory `Kalman/X/` has the header `Kalman/X.h` beside it, which
 * includes all of it: `#include <Kalman/X.h>` for the whole, `#include <Kalman/X/Name.h>`
 * for one.
 *
 * `Kalman/Routers/NN/Onnx.h` and `Torch.h` need ONNX Runtime and libTorch: they are
 * libKalmanLearn's, and `Kalman/Learn.h` includes them.
 *
 * *********************************************
 */

#ifndef KALMAN_ROUTERS_NN_H
#define KALMAN_ROUTERS_NN_H

#include "Kalman/Routers/NN/Model.h"
#include "Kalman/Routers/NN/MLP.h"

#endif // KALMAN_ROUTERS_NN_H
