# FindONNXRuntime.cmake — locate an ONNX Runtime installation.
#
# libKalmanFramework uses ONNX Runtime for one job: executing a decision or
# routing policy that was trained in the Python framework and exported to ONNX.
# It is entirely optional, so this module never errors — it sets
# ONNXRuntime_FOUND and the caller decides.
#
# Sets:
#   ONNXRuntime_FOUND
#   ONNXRuntime_INCLUDE_DIRS
#   ONNXRuntime_LIBRARIES
#   ONNXRuntime::ONNXRuntime   (imported target)
#
# Hints, in order: ONNXRuntime_ROOT / ONNXRUNTIME / ORT (variable or environment),
# then the usual local layouts. The glob picks up versioned prefixes such as
# ~/Software/onnxruntime/1.16.3.

file(GLOB _ort_hints
    "$ENV{HOME}/Software/onnxruntime/*"
    "$ENV{HOME}/.local"
    "/usr/local"
    "/opt/onnxruntime"
    "/opt/homebrew/opt/onnxruntime"
    "/usr/local/opt/onnxruntime"
)

find_path(ONNXRuntime_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRuntime_ROOT} $ENV{ONNXRuntime_ROOT}
          ${ONNXRUNTIME} $ENV{ONNXRUNTIME}
          ${ORT} $ENV{ORT}
          ${_ort_hints}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
)

find_library(ONNXRuntime_LIBRARY
    NAMES onnxruntime
    HINTS ${ONNXRuntime_ROOT} $ENV{ONNXRuntime_ROOT}
          ${ONNXRUNTIME} $ENV{ONNXRUNTIME}
          ${ORT} $ENV{ORT}
          ${_ort_hints}
    PATH_SUFFIXES lib lib64
)
unset(_ort_hints)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    REQUIRED_VARS ONNXRuntime_LIBRARY ONNXRuntime_INCLUDE_DIR)

if(ONNXRuntime_FOUND)
    set(ONNXRuntime_INCLUDE_DIRS ${ONNXRuntime_INCLUDE_DIR})
    set(ONNXRuntime_LIBRARIES    ${ONNXRuntime_LIBRARY})

    if(NOT TARGET ONNXRuntime::ONNXRuntime)
        add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
        set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
    endif()
endif()

mark_as_advanced(ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)
