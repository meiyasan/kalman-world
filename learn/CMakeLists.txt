
# libKalmanLearn — learned components, and the AI runtimes they run on.
#
# libKalman defines what a learned policy is (Kalman::Routers::NN::Model) and
# evaluates a small network with no dependency (NN::MLP). This library adds the
# two backends that need a runtime: NN::Onnx (ONNX Runtime) and NN::Torch
# (TorchScript). Training stays in Python; what crosses is a trained policy as
# one file. A world-model component, when there is one, belongs here too.
#
# Neither runtime is required. The library builds without them, and a backend
# whose runtime is absent reports HasRuntime() == false and refuses to load a
# file, so a program written against the API compiles and links everywhere.
cmake_minimum_required(VERSION 3.20)

if(NOT TARGET ${LIBRARY})
    return()
endif()

set(LEARN_LIBRARY "KalmanLearn")
file(GLOB_RECURSE KALMAN_LEARN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")

add_library(${LEARN_LIBRARY} SHARED ${KALMAN_LEARN_SOURCES})
target_include_directories(${LEARN_LIBRARY} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(${LEARN_LIBRARY} PUBLIC ${LIBRARY})
set_target_properties(${LEARN_LIBRARY} PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH FALSE
    BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    INSTALL_RPATH "@loader_path;$ORIGIN")

# --- optional: ONNX Runtime ---------------------------------------------------
# The seam between the two implementations: train a router or controller in the
# Python framework, export it to ONNX, execute it here. Reached only from
# src/Routers/NN/Onnx.cc, behind a pImpl, so it stays PRIVATE: a consumer never
# needs ONNX Runtime on its include path.
option(KALMAN_USE_ONNXRUNTIME "Let libKalmanLearn run ONNX policies when ONNX Runtime is found" ON)
set(KALMAN_WITH_ONNXRUNTIME 0)
if(KALMAN_USE_ONNXRUNTIME)
    find_package(ONNXRuntime QUIET)
endif()
if(ONNXRuntime_FOUND)
    set(KALMAN_WITH_ONNXRUNTIME 1)
    message_color(STATUS GREEN "   ONNX Runtime found — ONNX policies available")
    target_link_libraries(${LEARN_LIBRARY} PRIVATE ONNXRuntime::ONNXRuntime)

    # ONNX Runtime usually lives outside the loader path, and an installed
    # library keeps only its INSTALL_RPATH: without this a consumer of the
    # install tree fails at load with "Library not loaded: @rpath/libonnxruntime".
    get_filename_component(_ort_libdir "${ONNXRuntime_LIBRARY}" DIRECTORY)
    set_property(TARGET ${LEARN_LIBRARY} APPEND PROPERTY BUILD_RPATH "${_ort_libdir}")
    set_property(TARGET ${LEARN_LIBRARY} APPEND PROPERTY INSTALL_RPATH "${_ort_libdir}")
    unset(_ort_libdir)
else()
    message_color(STATUS YELLOW "   ONNX Runtime not found — ONNX policies unavailable")
endif()

# --- optional: libTorch -------------------------------------------------------
# The C++ library, for TorchScript policies and for the batched field solver
# (Kalman::Learn::Solvers::Vorticity). It is found like any C++ dependency:
# Torch_DIR when given, else the conda environment's libtorch package
# ($CONDA_PREFIX/share/cmake/Torch), else wherever CMake finds one.
#
# A process holds one libTorch. Where Python's torch is also imported, this
# must be the same installation; in a conda environment `libtorch` and
# `pytorch` are, which is why the environment is looked in first.
option(KALMAN_USE_TORCH "Build libKalmanLearn with libTorch (policies, field solver)" ON)
set(KALMAN_WITH_TORCH 0)
if(KALMAN_USE_TORCH)
    if(NOT Torch_DIR AND DEFINED ENV{CONDA_PREFIX}
       AND EXISTS "$ENV{CONDA_PREFIX}/share/cmake/Torch/TorchConfig.cmake")
        set(Torch_DIR "$ENV{CONDA_PREFIX}/share/cmake/Torch" CACHE PATH "libTorch of the conda environment")
    endif()
    get_cmake_property(_kalman_cache CACHE_VARIABLES)
    foreach(_kalman_torch_cached ${_kalman_cache})
        if(_kalman_torch_cached MATCHES "^(KALMAN_)?(TORCH|torch|C10|c10|kineto)[A-Za-z0-9_]*_LIBRARY$"
           OR _kalman_torch_cached STREQUAL "Caffe2_DIR")
            unset(${_kalman_torch_cached} CACHE)
        endif()
    endforeach()
    unset(_kalman_cache)
    # ... and pin them to Torch_DIR's own tree: TorchConfig searches
    # CMAKE_PREFIX_PATH before its own prefix, where another libTorch may sit.
    if(Torch_DIR)
        get_filename_component(_kalman_torch_root "${Torch_DIR}/../../.." ABSOLUTE)
        if(EXISTS "${Torch_DIR}/../Caffe2/Caffe2Config.cmake")
            get_filename_component(_kalman_caffe2 "${Torch_DIR}/../Caffe2" ABSOLUTE)
            set(Caffe2_DIR "${_kalman_caffe2}" CACHE PATH "Caffe2 of the chosen Torch" FORCE)
            unset(_kalman_caffe2)
        endif()
        find_library(TORCH_LIBRARY torch PATHS "${_kalman_torch_root}/lib" NO_DEFAULT_PATH)
        foreach(_kalman_torch_cached c10 torch_cpu kineto)
            find_library(${_kalman_torch_cached}_LIBRARY ${_kalman_torch_cached}
                         PATHS "${_kalman_torch_root}/lib" NO_DEFAULT_PATH)
        endforeach()
        unset(_kalman_torch_root)
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(KALMAN_WITH_TORCH 1)
        message_color(STATUS GREEN "   libTorch found (${Torch_DIR}) — TorchScript policies and the field solver available")
        # The three libraries themselves, by path, from Torch_DIR's own tree. Not
        # ${TORCH_LIBRARIES}: that list carries whatever protobuf and abseil
        # CMake found on this machine, which need not be the ones this Torch was
        # built with, and two protobufs in one process crash at load.
        get_filename_component(_torch_dir "${TORCH_LIBRARY}" DIRECTORY)
        set(_kalman_torch_libs "${TORCH_LIBRARY}")
        foreach(_torch_lib torch_cpu c10)
            find_library(KALMAN_${_torch_lib}_LIBRARY ${_torch_lib} PATHS "${_torch_dir}" NO_DEFAULT_PATH)
            if(KALMAN_${_torch_lib}_LIBRARY)
                list(APPEND _kalman_torch_libs "${KALMAN_${_torch_lib}_LIBRARY}")
            endif()
        endforeach()
        # GPU builds register their kernels from these; keep them linked although
        # no symbol of theirs is referenced.
        foreach(_torch_lib torch_cuda c10_cuda)
            find_library(KALMAN_${_torch_lib}_LIBRARY ${_torch_lib} PATHS "${_torch_dir}" NO_DEFAULT_PATH)
            if(KALMAN_${_torch_lib}_LIBRARY)
                list(APPEND _kalman_torch_libs "-Wl,--no-as-needed" "${KALMAN_${_torch_lib}_LIBRARY}" "-Wl,--as-needed")
            endif()
        endforeach()
        target_link_libraries(${LEARN_LIBRARY} PRIVATE ${_kalman_torch_libs})
        target_include_directories(${LEARN_LIBRARY} SYSTEM PRIVATE ${TORCH_INCLUDE_DIRS})
        set_property(TARGET ${LEARN_LIBRARY} APPEND PROPERTY BUILD_RPATH "${_torch_dir}")
        set_property(TARGET ${LEARN_LIBRARY} APPEND PROPERTY INSTALL_RPATH "${_torch_dir}")
        unset(_kalman_torch_libs)
        unset(_torch_lib)
        unset(_torch_dir)
    else()
        message_color(STATUS YELLOW "   libTorch not found — TorchScript policies unavailable")
    endif()
else()
    message_color(STATUS YELLOW "   libTorch switched off (-DKALMAN_USE_TORCH=OFF) — no TorchScript policies, no field solver")
endif()

target_compile_definitions(${LEARN_LIBRARY} PRIVATE
    KALMAN_WITH_ONNXRUNTIME=${KALMAN_WITH_ONNXRUNTIME}
    KALMAN_WITH_TORCH=${KALMAN_WITH_TORCH})

set_target_properties(${LEARN_LIBRARY} PROPERTIES EXPORT_NAME Learn)
install(TARGETS ${LEARN_LIBRARY} EXPORT KalmanTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
        FILES_MATCHING PATTERN "*.h")
