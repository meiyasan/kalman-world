
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
# For one thing: executing a TorchScript policy. Off unless asked for. A libTorch
# loaded into a Python process that has its own Torch breaks `import torch`, and
# this library is what Python would load to run a learned component.
option(KALMAN_USE_TORCH "Let libKalmanLearn run TorchScript policies (links libTorch)" OFF)
set(KALMAN_WITH_TORCH 0)
if(KALMAN_USE_TORCH)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(KALMAN_WITH_TORCH 1)
        message_color(STATUS GREEN "   libTorch found — TorchScript policies available")
        # TORCH_LIBRARIES, not the imported target: the Torch config sets flags
        # (notably -D_GLIBCXX_USE_CXX11_ABI) that must not leak to consumers.
        target_link_libraries(${LEARN_LIBRARY} PRIVATE ${TORCH_LIBRARIES})
        target_include_directories(${LEARN_LIBRARY} PRIVATE ${TORCH_INCLUDE_DIRS})
        foreach(_torch_lib ${TORCH_LIBRARIES})
            if(EXISTS "${_torch_lib}")
                get_filename_component(_torch_dir "${_torch_lib}" DIRECTORY)
                set_property(TARGET ${LEARN_LIBRARY} APPEND PROPERTY BUILD_RPATH "${_torch_dir}")
                set_property(TARGET ${LEARN_LIBRARY} APPEND PROPERTY INSTALL_RPATH "${_torch_dir}")
            endif()
        endforeach()
        unset(_torch_lib)
        unset(_torch_dir)
    else()
        message_color(STATUS YELLOW "   libTorch not found — TorchScript policies unavailable")
    endif()
else()
    message_color(STATUS YELLOW "   libTorch not requested (-DKALMAN_USE_TORCH=ON) — TorchScript policies unavailable")
endif()

target_compile_definitions(${LEARN_LIBRARY} PRIVATE
    KALMAN_WITH_ONNXRUNTIME=${KALMAN_WITH_ONNXRUNTIME}
    KALMAN_WITH_TORCH=${KALMAN_WITH_TORCH})

set_target_properties(${LEARN_LIBRARY} PROPERTIES EXPORT_NAME Learn)
install(TARGETS ${LEARN_LIBRARY} EXPORT KalmanTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
        FILES_MATCHING PATTERN "*.h")
