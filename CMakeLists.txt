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

# This project is pure Python (installed with pip — `make setup`); the C fast
# path is built by CFFI through `make accel`, not by CMake. So CMake is here for
# one job: the Doxygen + doxygen-awesome documentation.
set(BUILD_LIBRARY OFF CACHE BOOL "kalman has no CMake library target" FORCE)
set(BUILD_TOOLS OFF CACHE BOOL "kalman has no CMake tool targets" FORCE)
set(BUILD_EXAMPLES OFF CACHE BOOL "examples are notebooks, not CMake targets" FORCE)
set(BUILD_TESTS OFF CACHE BOOL "tests run under pytest — `make tests`" FORCE)
# pyproject.toml is maintained by hand here: it carries the optional-dependency
# extras, console scripts, ruff/pytest/coverage configuration and the cffi build
# requirement, none of which pyproject.py knows about. Regenerating it would
# silently drop them.
set(BUILD_PYPROJECT OFF CACHE BOOL "pyproject.toml is hand-maintained" FORCE)

check_build_option(BUILD_DOCUMENTATION "Library documentation won't be built.")

#
# Additional restriction/warning feature
include(DisableInSourceBuild)
include(GitBranchWarning)

# Project overview
DUMP_PROJECT()

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
