
# libKalmanApps — the application models of `modules/apps/` in C++.
#
# libKalman ships the contracts and no physics; this library is where
# the physics lives. It registers every model with Kalman::Factory, so it is
# loaded like any plugin: by the Python package (which then runs a Python
# model's filtering pass on its C++ twin), by the MCP server, or linked directly
# by a C++ or ROOT+ program that includes <Kalman/Apps/...>.
cmake_minimum_required(VERSION 3.20)

if(NOT TARGET ${LIBRARY})
    return()
endif()

set(APPS_LIBRARY "KalmanApps")
file(GLOB_RECURSE KALMAN_APPS_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

add_library(${APPS_LIBRARY} SHARED ${KALMAN_APPS_SOURCES})
target_include_directories(${APPS_LIBRARY} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(${APPS_LIBRARY} PUBLIC ${LIBRARY})
set_target_properties(${APPS_LIBRARY} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "@loader_path;$ORIGIN")

set_target_properties(${APPS_LIBRARY} PROPERTIES EXPORT_NAME Apps)
install(TARGETS ${APPS_LIBRARY} EXPORT KalmanTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
        FILES_MATCHING PATTERN "*.h")
