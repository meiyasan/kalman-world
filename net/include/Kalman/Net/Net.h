/**
 * *********************************************
 *
 * \file Net.h
 * \brief libKalmanNet in one include
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \namespace Kalman::Net
 * \brief The JSON API, the REST server and the MCP server of the framework, in C++.
 *
 * \details Kalman::Net::Api answers the verbs of the Python facade `kalman.api`
 * for models that live in C++; Kalman::Net::RestServer and Kalman::Net::McpServer
 * carry them over HTTP and JSON-RPC; Kalman::Net::Auth verifies the scoped JWTs
 * the Python side issues. The program `kalman-serve` starts either server. See
 * docs/18_Net.md.
 *
 * The library's dependencies (nlohmann_json, cpp-httplib, OpenSSL, SQLite) are
 * private: these headers use the standard library only.
 *
 * *********************************************
 */

#ifndef KALMAN_NET_NET_H
#define KALMAN_NET_NET_H

#include "Kalman/Net/Error.h"
#include "Kalman/Net/Auth.h"
#include "Kalman/Net/Api.h"
#include "Kalman/Net/RestServer.h"
#include "Kalman/Net/McpServer.h"

#endif // KALMAN_NET_NET_H
