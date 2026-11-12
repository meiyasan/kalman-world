# libKalmanNet -- the JSON API, the REST server and the MCP server, in C++.
#
# libKalman knows models, engines and filters; this library answers for them over
# a wire. Kalman::Net::Api takes a verb and a JSON payload and returns JSON, for
# every model Kalman::Factory knows (libKalmanApps and loaded plugins) and for the
# authored "matrices" specs of the shared registry. Kalman::Net::RestServer and
# Kalman::Net::McpServer put it on HTTP and on JSON-RPC (stdio or HTTP), and
# `kalman-serve` starts either.
#
# Its dependencies are its own and stay PRIVATE: nlohmann_json (JSON),
# cpp-httplib (HTTP), OpenSSL (HMAC-SHA256 for the JWTs) and, when found, SQLite3
# (the shared registry: authored models, plugins, sources, the token revocation
# list). A consumer includes <Kalman/Net/...> and sees std::string only. When a
# required one is missing the library is skipped with a notice; nothing else in
# the build depends on it.
cmake_minimum_required(VERSION 3.20)

if(NOT TARGET ${LIBRARY})
    return()
endif()

set(NET_LIBRARY "KalmanNet")

# The conda environment is looked in first, as libKalmanLearn does for libTorch
# (this directory's scope only: the core's search is not changed).
if(DEFINED ENV{CONDA_PREFIX} AND EXISTS "$ENV{CONDA_PREFIX}")
    list(PREPEND CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
    if(NOT OPENSSL_ROOT_DIR)
        set(OPENSSL_ROOT_DIR "$ENV{CONDA_PREFIX}")
    endif()
endif()

find_package(nlohmann_json 3.2 QUIET)
find_package(httplib QUIET)
find_package(OpenSSL QUIET COMPONENTS Crypto)
find_package(SQLite3 QUIET)

set(_kalman_net_missing "")
if(NOT nlohmann_json_FOUND)
    list(APPEND _kalman_net_missing "nlohmann_json")
endif()
if(NOT httplib_FOUND)
    list(APPEND _kalman_net_missing "cpp-httplib")
endif()
if(NOT OpenSSL_FOUND)
    list(APPEND _kalman_net_missing "OpenSSL")
endif()
if(_kalman_net_missing)
    string(REPLACE ";" ", " _kalman_net_missing "${_kalman_net_missing}")
    message_color(STATUS YELLOW "   libKalmanNet skipped -- not found: ${_kalman_net_missing} "
        "(conda install -c conda-forge nlohmann_json cpp-httplib openssl)")
    unset(_kalman_net_missing)
    return()
endif()

file(GLOB_RECURSE KALMAN_NET_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

add_library(${NET_LIBRARY} SHARED ${KALMAN_NET_SOURCES})
target_include_directories(${NET_LIBRARY} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(${NET_LIBRARY} PUBLIC ${LIBRARY})
target_link_libraries(${NET_LIBRARY} PRIVATE
    $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
    $<BUILD_INTERFACE:httplib::httplib>
    OpenSSL::Crypto
    ${CMAKE_DL_LIBS})

set(KALMAN_NET_WITH_SQLITE 0)
if(SQLite3_FOUND)
    set(KALMAN_NET_WITH_SQLITE 1)
    if(TARGET SQLite3::SQLite3)
        target_link_libraries(${NET_LIBRARY} PRIVATE SQLite3::SQLite3)
    else()
        target_link_libraries(${NET_LIBRARY} PRIVATE SQLite::SQLite3)
    endif()
    message_color(STATUS GREEN "   libKalmanNet: JSON API, REST and MCP servers (registry: SQLite ${SQLite3_VERSION})")
else()
    message_color(STATUS YELLOW "   libKalmanNet: SQLite3 not found -- no shared registry, and tokens are refused when auth is on")
endif()

# The plugin directories allowed by default, as the Python side's
# modules/framework/native/core.py:plugin_dirs() has them for this checkout:
# the install and build trees. KALMAN_PLUGIN_PATH replaces the list at run time.
target_compile_definitions(${NET_LIBRARY} PRIVATE
    KALMAN_NET_WITH_SQLITE=${KALMAN_NET_WITH_SQLITE}
    KALMAN_NET_INSTALL_LIBDIR="${CMAKE_INSTALL_PREFIX}/lib"
    KALMAN_NET_BUILD_LIBDIR="${CMAKE_BINARY_DIR}/lib")

# OpenSSL usually comes from the conda environment, outside the loader path.
foreach(_kalman_net_dep ${OPENSSL_CRYPTO_LIBRARY} ${SQLite3_LIBRARY})
    if(_kalman_net_dep AND EXISTS "${_kalman_net_dep}")
        get_filename_component(_kalman_net_dir "${_kalman_net_dep}" DIRECTORY)
        list(APPEND _kalman_net_rpath "${_kalman_net_dir}")
    endif()
endforeach()
if(_kalman_net_rpath)
    list(REMOVE_DUPLICATES _kalman_net_rpath)
endif()

set_target_properties(${NET_LIBRARY} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${_kalman_net_rpath}"
    INSTALL_RPATH "@loader_path;$ORIGIN;${_kalman_net_rpath}")

# kalman-serve: the REST server (--rest --port N) or the MCP server on stdio (--mcp).
add_executable(kalman-serve "${CMAKE_CURRENT_SOURCE_DIR}/app/kalman-serve.cc")
target_link_libraries(kalman-serve PRIVATE ${NET_LIBRARY})
set_target_properties(kalman-serve PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;${_kalman_net_rpath}"
    INSTALL_RPATH "@loader_path/../lib;$ORIGIN/../lib;${_kalman_net_rpath}")
unset(_kalman_net_rpath)
unset(_kalman_net_dir)
unset(_kalman_net_dep)

set_target_properties(${NET_LIBRARY} PROPERTIES EXPORT_NAME Net)
install(TARGETS ${NET_LIBRARY} EXPORT KalmanTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(TARGETS kalman-serve RUNTIME DESTINATION bin)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
        FILES_MATCHING PATTERN "*.h")
