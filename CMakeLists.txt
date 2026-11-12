# Version used during development
cmake_minimum_required(VERSION 3.20)

# Project information
set(PROJECT_TITLE "Kalman World")
project(kalman
    VERSION 1.0.2
    DESCRIPTION "State estimation, dynamical systems and predictive learning 🛰️📈"
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
# documentation. The second is `libKalman` (include/Kalman, src/), the framework
# core in C++ — a standalone library with no required dependency, which a ROOT+
# component or any other C++ analysis links against.
#
# The Python package is *not* built here: it is installed with pip (`make
# setup`) and its optional C fast path is built by CFFI (`make accel`). The two
# implementations are peers — libKalman does not wrap the Python code and does
# not link against CPython.
set(BUILD_TOOLS OFF CACHE BOOL "kalman has no CMake tool targets" FORCE)
# pyproject.toml is maintained by hand here: it carries the optional-dependency
# extras, console scripts, ruff/pytest/coverage configuration and the cffi build
# requirement, none of which pyproject.py knows about. Regenerating it would
# silently drop them.
set(BUILD_PYPROJECT OFF CACHE BOOL "pyproject.toml is hand-maintained" FORCE)

check_build_option(BUILD_DOCUMENTATION "Library documentation won't be built.")
check_build_option(BUILD_LIBRARY "The C++ library `libKalman` will not be built.")
check_build_option(BUILD_TESTS "C++ tests for `libKalman` won't be running (`make tests` is Python).")
check_build_option(BUILD_EXAMPLES "C++ examples won't be compiled.")

#
# Additional restriction/warning feature
include(DisableInSourceBuild)
include(GitBranchWarning)

# Project overview
DUMP_PROJECT()

#
# Create the C++ libraries
#
#   libKalman       the core: models, engines, filters, integrators, solvers,
#                   controllers, routers. No required dependency.
#   libKalmanApps   the application models (apps/).
#   libKalmanLearn  learned components and the AI runtimes they need (learn/).
#   libKalmanNet    the JSON API, the REST and MCP servers (net/).
#
# libKalman is a **standalone** library: no ROOT, no Eigen, no Torch, no ONNX
# Runtime. The dependency runs the other way — a ROOT/ROOT+ component links
# libKalman, not the reverse — so this build must never reach for ROOT.
# `Kalman::Matrix` and `Kalman::Vector` are why: small dense types, sized for
# the matrices a filter actually multiplies. Its one optional backend is BLAS,
# for matrix products above KALMAN_BLAS_MIN_DIM. The AI runtimes are found and
# linked by libKalmanLearn alone, so what happens to be installed on the build
# machine never decides what the core depends on.
#
# Up to release 1.0 the core was `libKalmanFramework`. That name is still
# installed, as a link to libKalman, so what linked it keeps linking.
if(BUILD_LIBRARY)

    include(StandardCompilerC)
    include(StandardCompilerCXX)

    message_title("Library")

    set(LIBRARY "Kalman")
    set(LIBRARY_FORMER "KalmanFramework")
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

    # Kalman::Factory::LoadPlugin dlopens; on glibc < 2.34 that lives in libdl.
    target_link_libraries(${LIBRARY} PRIVATE ${CMAKE_DL_LIBS})

    if(KALMAN_WITH_BLAS)
        if(APPLE)
            # src/Matrix.cc takes cblas_dgemm from Accelerate's headers (new
            # LAPACK interface), so link Accelerate itself: whatever BLAS
            # find_package picked (conda's libblas, say) lacks those symbols.
            target_link_libraries(${LIBRARY} PRIVATE "-framework Accelerate")
        else()
            target_link_libraries(${LIBRARY} PRIVATE ${BLAS_LIBRARIES})
        endif()
    endif()

    # An environment script, so a consumer can `source` one file and then compile
    # or run against the install tree.
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/thislib.sh.in"
                   "${CMAKE_BINARY_DIR}/bin/thislib.${LIBRARY}.sh" @ONLY)

    set_target_properties(${LIBRARY} PROPERTIES EXPORT_NAME Kalman)
    install(TARGETS ${LIBRARY} EXPORT KalmanTargets LIBRARY DESTINATION lib ARCHIVE DESTINATION lib)

    # The former name, as a link beside the library in the build and install trees.
    set(_kalman_file "${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    set(_kalman_former "${CMAKE_SHARED_LIBRARY_PREFIX}${LIBRARY_FORMER}${CMAKE_SHARED_LIBRARY_SUFFIX}")
    add_custom_command(TARGET ${LIBRARY} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink "${_kalman_file}" "${CMAKE_BINARY_DIR}/lib/${_kalman_former}"
        VERBATIM)
    install(CODE "execute_process(COMMAND \${CMAKE_COMMAND} -E create_symlink
        \"${_kalman_file}\" \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}/lib/${_kalman_former}\")")
    install(FILES "${CMAKE_BINARY_DIR}/bin/thislib.${LIBRARY}.sh" DESTINATION bin
            PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                        GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
    install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/Kalman" DESTINATION include
            FILES_MATCHING PATTERN "*.h")
    install(FILES "${CMAKE_BINARY_DIR}/include/Kalman/Config.h" DESTINATION include/Kalman)

    # `kalman`: what this installation can run (the models and what each
    # declares). It links the core only and loads libKalmanApps as a plugin.
    add_executable(kalman-cli "${CMAKE_CURRENT_SOURCE_DIR}/app/kalman.cc")
    target_link_libraries(kalman-cli PRIVATE ${LIBRARY} ${CMAKE_DL_LIBS})
    set_target_properties(kalman-cli PROPERTIES
        OUTPUT_NAME kalman
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
        INSTALL_RPATH "@loader_path/../lib;$ORIGIN/../lib")
    install(TARGETS kalman-cli RUNTIME DESTINATION bin)

    # libKalmanApps: the application models, beside the library that defines
    # their contracts. Before the tests, which link it.
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/apps/CMakeLists.txt)
        add_subdirectory("apps")
    endif()

    # libKalmanLearn: the learned components, and the only place an AI runtime
    # is looked for.
    option(KALMAN_BUILD_LEARN "Build libKalmanLearn (ONNX and TorchScript policies)" ON)
    if(KALMAN_BUILD_LEARN AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/learn/CMakeLists.txt)
        add_subdirectory("learn")
    endif()

    # libKalmanNet: the JSON API, the REST and MCP servers and `kalman-serve`.
    # Skipped, with a notice, when nlohmann_json, cpp-httplib or OpenSSL is missing.
    option(KALMAN_BUILD_NET "Build libKalmanNet (JSON API, REST and MCP servers)" ON)
    if(KALMAN_BUILD_NET AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/net/CMakeLists.txt)
        add_subdirectory("net")
    endif()

    # The CMake package: `find_package(Kalman COMPONENTS Apps Learn Net)` gives the
    # targets Kalman::Kalman, Kalman::Apps, Kalman::Learn and Kalman::Net. After the
    # subdirectories, which add their targets to the same export set.
    include(CMakePackageConfigHelpers)
    set(KALMAN_COMPONENTS "")
    foreach(_component Apps Learn Net)
        if(TARGET Kalman${_component})
            list(APPEND KALMAN_COMPONENTS ${_component})
        endif()
    endforeach()
    configure_package_config_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/KalmanConfig.cmake.in"
        "${CMAKE_BINARY_DIR}/cmake/KalmanConfig.cmake"
        INSTALL_DESTINATION lib/cmake/Kalman)
    write_basic_package_version_file("${CMAKE_BINARY_DIR}/cmake/KalmanConfigVersion.cmake"
        VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
    install(EXPORT KalmanTargets NAMESPACE Kalman:: DESTINATION lib/cmake/Kalman)
    install(FILES "${CMAKE_BINARY_DIR}/cmake/KalmanConfig.cmake"
                  "${CMAKE_BINARY_DIR}/cmake/KalmanConfigVersion.cmake"
            DESTINATION lib/cmake/Kalman)

    if(CMAKE_MAINPROJECT)

        add_custom_target(lib DEPENDS ${LIBRARY} COMMENT "Compiling library")
        if(TARGET KalmanApps)
            add_dependencies(lib KalmanApps)
        endif()
        if(TARGET KalmanLearn)
            add_dependencies(lib KalmanLearn)
        endif()
        if(TARGET KalmanNet)
            add_dependencies(lib KalmanNet kalman-serve)
        endif()

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
    # The C++ side: the library's headers, the application models and the examples,
    # so the site documents and searches both languages.
    "${CMAKE_SOURCE_DIR}/include/Kalman"
    "${CMAKE_SOURCE_DIR}/apps/include"
    "${CMAKE_SOURCE_DIR}/learn/include"
    "${CMAKE_SOURCE_DIR}/net/include"
    "${CMAKE_SOURCE_DIR}/examples"
    # The grouping of the guide pages in the side navigation.
    "${CMAKE_SOURCE_DIR}/share/site"
)
set(DOXYGEN_MAINPAGE "${CMAKE_SOURCE_DIR}/README.md")

if(BUILD_DOCUMENTATION AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/share/doxygen/CMakeLists.txt)
    add_subdirectory("share/doxygen")

    # The theme's layout file lists the navigation only. Doxygen 1.12 filled in
    # the rest; later versions leave class, namespace and file pages empty. Use
    # the complete layout kept in cmake/ (see the comment at its top). A later
    # assignment wins in a Doxyfile, so this is appended to the generated one.
    # The other settings appended here:
    # - STRIP_FROM_PATH: without it every page name carries the absolute path of
    #   the checkout (md__2_users_2<you>_2...), so links change from machine to
    #   machine and a local path is published.
    # - IMAGE_PATH: the figures and animations the guide pages show.
    # - DOT_GRAPH_MAX_NODES: Kalman.h includes 54 headers, over the default 50.
    if(EXISTS "${CMAKE_BINARY_DIR}/Doxyfile")
        file(APPEND "${CMAKE_BINARY_DIR}/Doxyfile"
             "\nLAYOUT_FILE = ${CMAKE_SOURCE_DIR}/cmake/DoxygenLayout.xml\n"
             "STRIP_FROM_PATH = ${CMAKE_SOURCE_DIR}\n"
             "MARKDOWN_ID_STYLE = GITHUB\n"
             "IMAGE_PATH += ${CMAKE_SOURCE_DIR}/docs/images ${CMAKE_SOURCE_DIR}/docs/apps/images\n"
             "DOT_GRAPH_MAX_NODES = 80\n")
    endif()

    # Full-text search. Doxygen's search box finds names; tools/site_search.py
    # adds the text of every page to it, after Doxygen has written the site and
    # before `pages` publishes it.
    find_program(KALMAN_SITE_PYTHON NAMES python3 python)
    if(TARGET doxygen AND KALMAN_SITE_PYTHON)
        # The notebooks as web pages, where their interactive players run (a
        # notebook viewed on GitLab is shown without JavaScript).
        add_custom_target(sitesearch
            COMMAND ${KALMAN_SITE_PYTHON} "${CMAKE_SOURCE_DIR}/tools/site_search.py"
                    --newest "${CMAKE_BINARY_DIR}/public/docs"
            COMMAND ${KALMAN_SITE_PYTHON} "${CMAKE_SOURCE_DIR}/tools/site_notebooks.py"
                    --newest "${CMAKE_BINARY_DIR}/public/docs"
            COMMAND ${KALMAN_SITE_PYTHON} "${CMAKE_SOURCE_DIR}/tools/site_versions.py"
                    "${CMAKE_BINARY_DIR}/public/docs"
            DEPENDS doxygen
            COMMENT "Indexing the documentation and publishing the notebooks"
            VERBATIM)
        if(TARGET metadata)
            add_dependencies(metadata sitesearch)
        endif()
    endif()
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
