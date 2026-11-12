/**
 * *********************************************
 *
 * \file Error.h
 * \brief Header of the Kalman::Net::Error class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Net::Error
 * \brief A refused request, with the HTTP status that says why.
 *
 * \details Every verb of Kalman::Net::Api reports a failure by throwing this,
 * and both servers turn it into their reply: the REST server into the status
 * and `{"detail": message}`, the MCP server into a tool result marked as an
 * error. The statuses are the ones the Python REST API uses
 * (modules/rest/app.py):
 *
 *  - 400 a bad request (malformed spec, wrong shapes, unknown model base);
 *  - 401 no token, or a forged, expired, malformed or revoked one;
 *  - 403 a good token without the scope;
 *  - 404 an unknown name or run;
 *  - 409 a name taken by a different definition;
 *  - 501 a verb this server does not serve (it stays in the Python server).
 *
 * *********************************************
 */

#ifndef KALMAN_NET_ERROR_H
#define KALMAN_NET_ERROR_H

#include <stdexcept>
#include <string>

namespace Kalman::Net
{
    class Error : public std::runtime_error
    {
        private:
            int fStatus;

        public:
            Error(int status, const std::string &what) : std::runtime_error(what), fStatus(status) {}

            /// The HTTP status of the refusal.
            int Status() const { return fStatus; }
    };
}

#endif // KALMAN_NET_ERROR_H
