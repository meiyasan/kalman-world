# Version used during development
cmake_minimum_required(VERSION 3.20)

# Project information
set(PROJECT_TITLE "Kalman Framework")
project(kalman
    VERSION 0.1
    DESCRIPTION "Composable Kalman-family estimation 🛰️📈"
    HOMEPAGE_URL "https://max-planck.sv.yc.tcu.ac.jp/kagra/sus/kalman-framework"
)

set(LIBRARY ${PROJECT_NAME})
set(LIBRARY_TITLE ${PROJECT_TITLE})

# C++20 for the library. (Do NOT set CMAKE_CXX_STANDARD_REQUIRED here:
# cmake/Modules/StandardCompilerCXX.cmake repurposes it as a *minimum standard
# number*, so the usual ON would be read as "require C++23" and configuration
# would fail.)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)

# Include custom CMake modules and utilities
file(GLOB_RECURSE CMAKE_MODULE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/*/*")
list(TRANSFORM CMAKE_MODULE_PATHS REPLACE "(.*)/[^/]*$" "\\1")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATHS})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/Message.cmake")
    message(FATAL_ERROR
        "cmake/Modules is empty — initialise the submodules first:\n"
        "    git submodule update --init --recursive")
endif()

#
# Prepare project architecture
include(ProjectArchitecture)
include(BuildOptions)
include(DisplayMotd)
include(Message)

DISPLAY_MOTD()
MESSAGE_TITLE("${PROJECT_TITLE}")

update_build_options()

# CMake has two jobs here. The first is the Doxygen + doxygen-awesome
# documentation. The second is `libKalmanFramework` (include/Kalman, src/), the framework
# core in C++ — a standalone library with no required dependency, which a ROOT+
# component or any other C++ analysis links against.
#
# The Python package is *not* built here: it is installed with pip (`make
# setup`) and its optional C fast path is built by CFFI (`make accel`). The two
# implementations are peers — libKalmanFramework does not wrap the Python code and does
# not link against CPython.
set(BUILD_TOOLS OFF CACHE BOOL "kalman has no CMake tool targets" FORCE)
# pyproject.toml is maintained by hand here: it carries the optional-dependency
# extras, console scripts, ruff/pytest/coverage configuration and the cffi build
# requirement, none of which pyproject.py knows about. Regenerating it would
# silently drop them.
set(BUILD_PYPROJECT OFF CACHE BOOL "pyproject.toml is hand-maintained" FORCE)

check_build_option(BUILD_DOCUMENTATION "Library documentation won't be built.")
check_build_option(BUILD_LIBRARY "The C++ library `libKalmanFramework` will not be built.")
check_build_option(BUILD_TESTS "C++ tests for `libKalmanFramework` won't be running (`make tests` is Python).")
check_build_option(BUILD_EXAMPLES "C++ examples won't be compiled.")

#
# Additional restriction/warning feature
include(DisableInSourceBuild)
include(GitBranchWarning)

# Project overview
DUMP_PROJECT()

#
# Create the C++ library — libKalmanFramework
#
# It is a **standalone** library: no ROOT, no Eigen, no Torch required. The
# dependency runs the other way — a ROOT/ROOT+ component links libKalmanFramework,
# not the reverse — so this build must never reach for ROOT. `Kalman::Matrix` and
# `Kalman::Vector` are why: small dense types, sized for the matrices a filter
# actually multiplies.
#
# Three optional backends, all detected quietly and all inessential:
#   BLAS         — faster matrix products above KALMAN_BLAS_MIN_DIM
#   libTorch     — tensor-shaped work (ensembles, fields, learned components)
#   ONNXRuntime  — executing decision/routing policies trained in Python
#
# The target is `KalmanFramework`, so the artefact is `libKalmanFramework.so`:
# it names the *framework*, and cannot be mistaken for root-plus' own much
# smaller `ROOT::IOPlus::TKalmanFilter`. Headers stay under `Kalman/` and the
# namespace stays `Kalman`.
if(BUILD_LIBRARY)

    include(StandardCompilerC)
    include(StandardCompilerCXX)

    message_title("Library")

    set(LIBRARY "KalmanFramework")
    set(KALMAN_VERSION       ${PROJECT_VERSION})
    set(KALMAN_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
    set(KALMAN_VERSION_MINOR ${PROJECT_VERSION_MINOR})

    # --- optional: BLAS -------------------------------------------------------
    # Measured on Apple Silicon (double, square, hot cache): the hand-rolled
    # kernel wins below n≈6 because a BLAS call costs ~20 ns of argument
    # handling; BLAS wins by 2x at n=12, 4x at n=16 and 17x at n=32. So the
    # threshold is a measurement, not a preference, and both paths stay compiled.
    option(KALMAN_USE_BLAS "Use CBLAS for matrix products above the crossover size" ON)
    set(KALMAN_BLAS_MIN_DIM 6 CACHE STRING "Matrix dimension at or above which BLAS is used")
    set(KALMAN_WITH_BLAS 0)
    if(KALMAN_USE_BLAS)
        find_package(BLAS QUIET)
        if(BLAS_FOUND)
            set(KALMAN_WITH_BLAS 1)
            message_color(STATUS GREEN "   BLAS found — matrix products >= ${KALMAN_BLAS_MIN_DIM} use cblas_dgemm")
        else()
            message_color(STATUS YELLOW "   BLAS not found — using the built-in kernel (correct, just slower at large n)")
        endif()
    endif()

    # --- optional: libTorch ---------------------------------------------------
    # Used for one thing: executing a TorchScript routing policy, the direct
    # counterpart of `routers/nn/`, whose architectures are all torch.nn modules.
    # Never for Kalman::Matrix — a Torch op carries a ~230 ns dispatch floor,
    # which is 10-25x an entire small matrix product.
    set(KALMAN_WITH_TORCH 0)
    find_package(Torch QUIET)
    if(Torch_FOUND)
        set(KALMAN_WITH_TORCH 1)
        message_color(STATUS GREEN "   libTorch found — TorchScript policies available")
    else()
        message_color(STATUS YELLOW "   libTorch not found — TorchScript policies unavailable")
    endif()

    # --- optional: ONNX Runtime ----------------------------------------------
    # The seam between the two implementations: train a router/controller in the
    # Python framework, export to ONNX, execute it here.
    set(KALMAN_WITH_ONNXRUNTIME 0)
    find_package(ONNXRuntime QUIET)
    if(ONNXRuntime_FOUND)
        set(KALMAN_WITH_ONNXRUNTIME 1)
        message_color(STATUS GREEN "   ONNX Runtime found — ONNX policies available")
    else()
        message_color(STATUS YELLOW "   ONNX Runtime not found — ONNX policies unavailable")
    endif()

    # Neither is required: Kalman::Routers::NN::MLP evaluates a policy with no
    # dependency at all, so the routing layer works in any build.

    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman/Config.h.in"
                   "${CMAKE_BINARY_DIR}/include/Kalman/Config.h" @ONLY)

    file(GLOB_RECURSE KALMAN_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc")
    add_library(${LIBRARY} SHARED ${KALMAN_SOURCES})

    target_include_directories(${LIBRARY} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
        $<INSTALL_INTERFACE:include>)

    target_compile_features(${LIBRARY} PUBLIC cxx_std_20)
    set_target_properties(${LIBRARY} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR}
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")

    if(KALMAN_WITH_BLAS)
        target_link_libraries(${LIBRARY} PRIVATE ${BLAS_LIBRARIES})
    endif()

    # ONNX Runtime is reached only from src/Routers/Neural.cc, and only behind a
    # pImpl, so it stays PRIVATE: a consumer of libKalmanFramework never needs ORT
    # on its include path even when the learned router is compiled in.
    if(KALMAN_WITH_ONNXRUNTIME)
        target_link_libraries(${LIBRARY} PRIVATE ONNXRuntime::ONNXRuntime)

        # ONNX Runtime usually lives outside the system loader path, and the
        # *installed* library keeps only its INSTALL_RPATH -- so without this a
        # consumer of the install tree fails at load with
        # "Library not loaded: @rpath/libonnxruntime...", even though the build
        # tree worked (CMake adds link-time paths to the build rpath).
        get_filename_component(_ort_libdir "${ONNXRuntime_LIBRARY}" DIRECTORY)
        set_property(TARGET ${LIBRARY} APPEND PROPERTY BUILD_RPATH "${_ort_libdir}")
        set_property(TARGET ${LIBRARY} APPEND PROPERTY INSTALL_RPATH "${_ort_libdir}")
        unset(_ort_libdir)
    endif()

    if(KALMAN_WITH_TORCH)
        # TORCH_LIBRARIES rather than the imported target: the Torch config sets
        # flags (notably -D_GLIBCXX_USE_CXX11_ABI) that must reach only the one
        # translation unit that includes torch/script.h.
        target_link_libraries(${LIBRARY} PRIVATE ${TORCH_LIBRARIES})

        # Same install-rpath reasoning as ONNX Runtime above: libTorch lives
        # outside the loader path, and the installed library keeps only its
        # INSTALL_RPATH.
        foreach(_torch_lib ${TORCH_LIBRARIES})
            if(EXISTS "${_torch_lib}")
                get_filename_component(_torch_dir "${_torch_lib}" DIRECTORY)
                set_property(TARGET ${LIBRARY} APPEND PROPERTY BUILD_RPATH "${_torch_dir}")
                set_property(TARGET ${LIBRARY} APPEND PROPERTY INSTALL_RPATH "${_torch_dir}")
            endif()
        endforeach()
        unset(_torch_lib)
        unset(_torch_dir)
    endif()

    # An environment script, so a consumer can `source` one file and then compile
    # or run against the install tree.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/thislib.sh.in"
                   "${CMAKE_BINARY_DIR}/bin/thislib.${LIBRARY}.sh" @ONLY)

    install(TARGETS ${LIBRARY} LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)
    install(FILES "${CMAKE_BINARY_DIR}/bin/thislib.${LIBRARY}.sh" DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
            FILES_MATCHING PATTERN "*.h")
    install(FILES "${CMAKE_BINARY_DIR}/include/Kalman/Config.h" DESTINATION include/Kalman)

    if(CMAKE_MAINPROJECT)

        add_custom_target(lib DEPENDS ${LIBRARY} COMMENT "Compiling library")

        # The C++ tests live in tests/cxx so that `tests/` stays the pytest suite;
        # they are driven by ctest (`make ctests`), not by `make tests`.
        #
        # enable_testing() has to be called *here*, in the top-level list file:
        # calling it only in the subdirectory builds the binaries but leaves the
        # root CTestTestfile.cmake empty, so `ctest` from the build directory
        # reports "No tests were found".
        # Examples first: `06_factory` loads the example plugin, so the test
        # list file needs that target to exist when it is configured.
        if(BUILD_EXAMPLES AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/cxx)
            add_subdirectory("examples/cxx")
        endif()

        if(BUILD_TESTS AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/cxx)
            enable_testing()
            add_subdirectory("tests/cxx")
        endif()
    endif()
endif()

#
# Prepare assets for documentation
#
# DoxygenVariables collects the conventional inputs (README.md, LICENSE,
# CONTRIBUTING.md, docs/, examples/, tests/); add the Python package itself,
# whose docstrings Doxygen reads (PYTHON_DOCSTRING = YES), and the contributor
# guide. README.md stays the landing page.
set(DOXYGEN_INPUT
    "${CMAKE_SOURCE_DIR}/modules"
    "${CMAKE_SOURCE_DIR}/AGENTS.md"
)
set(DOXYGEN_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")

if(BUILD_DOCUMENTATION AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/share/doxygen/CMakeLists.txt)
    add_subdirectory("share/doxygen")
elseif(BUILD_DOCUMENTATION)
    message(FATAL_ERROR
        "share/doxygen is empty — initialise the submodules first:\n"
        "    git submodule update --init --recursive")
endif()

#
# Hook Scripts
if(CMAKE_MAINPROJECT)

    # Make sure cmake includes are available
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake
            DESTINATION share
            PATTERN "*.in" EXCLUDE
            PATTERN ".*" EXCLUDE
            PATTERN "README*" EXCLUDE
    )

    # Post install message
    find_file(CMAKE_POST_CONFIG PostInstallConfig.cmake PATHS ${CMAKE_MODULE_PATHS})
    if(CMAKE_POST_CONFIG)
        install(SCRIPT "${CMAKE_POST_CONFIG}")
    endif()

    # Post cmake message
    include(PostCMakeConfig)

endif()
