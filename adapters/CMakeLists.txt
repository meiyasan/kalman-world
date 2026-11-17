# The adapters: specific hardware, as instruments of the library's generic sensors
# (`modules/adapters/` in Python). Headers only so far: <Kalman/Adapters.h> and one
# <Kalman/Adapters/<Device>.h> per device.
cmake_minimum_required(VERSION 3.20)

if(NOT TARGET ${LIBRARY})
    return()
endif()

target_include_directories(${LIBRARY} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
        FILES_MATCHING PATTERN "*.h")
