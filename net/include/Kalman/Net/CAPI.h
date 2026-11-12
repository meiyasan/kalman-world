/**
 * *********************************************
 *
 * \file CAPI.h
 * \brief The C interface of libKalmanNet
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * What the Python package (modules/framework/native/net.py) calls in
 * libKalmanNet: one verb of Kalman::Net::Api, JSON in and JSON out. Same
 * conventions as Kalman/CAPI.h: a pointer is NULL on failure, and
 * kalman_net_last_error() / kalman_net_last_status() say why. Returned strings
 * live in thread-local storage until the next call on the same thread.
 *
 * *********************************************
 */

#ifndef KALMAN_NET_CAPI_H
#define KALMAN_NET_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

/// The last error on this thread, or "".
const char *kalman_net_last_error(void);

/// The HTTP status of the last failure on this thread (0 after a success).
int kalman_net_last_status(void);

/// KALMAN_VERSION of the libKalman this library was built with.
const char *kalman_net_version(void);

/// Run `verb` with `arguments` (a JSON object as text; NULL = {}) and `token`
/// (NULL = none) on the process-wide Kalman::Net::Api. The answer is JSON text,
/// or NULL on failure.
const char *kalman_net_call(const char *verb, const char *arguments, const char *token);

#ifdef __cplusplus
}
#endif

#endif // KALMAN_NET_CAPI_H
