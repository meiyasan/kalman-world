/**
 * \file CAPI.cc
 * \brief Implementation of the C interface of libKalmanNet
 */
#include "Kalman/Net/CAPI.h"
#include "Kalman/Net/Api.h"

#include "Kalman/Config.h"

#include <exception>
#include <memory>
#include <mutex>
#include <string>

namespace
{
    thread_local std::string tError;
    thread_local std::string tAnswer;
    thread_local int tStatus = 0;

    Kalman::Net::Api &Shared()
    {
        static std::once_flag once;
        static std::unique_ptr<Kalman::Net::Api> api;
        std::call_once(once, [] { api = std::make_unique<Kalman::Net::Api>(); });
        return *api;
    }
}

extern "C" {

const char *kalman_net_last_error(void) { return tError.c_str(); }

int kalman_net_last_status(void) { return tStatus; }

const char *kalman_net_version(void) { return KALMAN_VERSION; }

const char *kalman_net_call(const char *verb, const char *arguments, const char *token)
{
    try
    {
        tError.clear();
        tStatus = 0;
        if (!verb) throw Kalman::Net::Error(400, "null verb");
        tAnswer = Shared().Call(verb, arguments ? arguments : "{}", token ? token : "");
        return tAnswer.c_str();
    }
    catch (const Kalman::Net::Error &e) { tError = e.what(); tStatus = e.Status(); }
    catch (const std::exception &e) { tError = e.what(); tStatus = 500; }
    catch (...) { tError = "unknown C++ exception"; tStatus = 500; }
    return nullptr;
}

}
