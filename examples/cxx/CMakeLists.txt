
# Version used during development
cmake_minimum_required(VERSION 3.20)
project(${PROJECT_NAME}/Examples)

# Include custom CMake modules and utilities
file(GLOB_RECURSE CMAKE_MODULE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/*/*")
list(TRANSFORM CMAKE_MODULE_PATHS REPLACE "(.*)/[^/]*$" "\\1")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_MODULE_PATHS})
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

include(Message)
include(ProjectArchitecture)
include(ParentCMakeOnly)

if(NOT TARGET ${LIBRARY})
    return()
endif()

message_title("Examples")

# Each example defines its own model, because that is where a model belongs:
# libKalmanFramework ships the contracts, applications ship the physics.
#
# `*_root.cc` examples demonstrate the integration with ROOT and are built only
# when ROOT is present. This is the one place in the project allowed to mention
# ROOT — and note the direction: the example links both ROOT and
# libKalmanFramework, while the library itself knows about neither.
file(GLOB EXAMPLE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")

# `*_plugin.cc` examples are built as *shared libraries*: they define models and
# register them with Kalman::Factory, and exist to be dlopen'ed — by the Python
# package, the MCP server, a ROOT+ analysis or the C++ tests — not to be run.
set(ROOT_EXAMPLES)
set(PLUGIN_EXAMPLES)
foreach(SRC ${EXAMPLE_SOURCES})
    if(${SRC} MATCHES "_root\\.cc$")
        list(APPEND ROOT_EXAMPLES ${SRC})
        list(REMOVE_ITEM EXAMPLE_SOURCES ${SRC})
    elseif(${SRC} MATCHES "_plugin\\.cc$")
        list(APPEND PLUGIN_EXAMPLES ${SRC})
        list(REMOVE_ITEM EXAMPLE_SOURCES ${SRC})
    endif()
endforeach()

if(ROOT_EXAMPLES)
    find_package(ROOT QUIET COMPONENTS Graf Gpad Hist)
    if(NOT ROOT_FOUND)
        message_color(STATUS YELLOW "   ROOT not found — skipping the *_root demonstrators")
        set(ROOT_EXAMPLES)
    endif()
endif()

if(NOT TARGET examples)
    add_custom_target(examples COMMENT "Compiling the examples")
endif()

# example_plugin.cc -> libKalmanPlugin_example, next to libKalmanFramework so a
# consumer of either tree (build or install) finds both with one search path.
foreach(SRC ${PLUGIN_EXAMPLES})
    get_filename_component(PLUGIN_STEM ${SRC} NAME_WE)
    string(REGEX REPLACE "_plugin$" "" PLUGIN_STEM ${PLUGIN_STEM})
    set(PLUGIN_NAME "KalmanPlugin_${PLUGIN_STEM}")
    message_color("   ./cxx/${PLUGIN_NAME} (plugin)")

    add_library(${PLUGIN_NAME} SHARED ${SRC})
    target_link_libraries(${PLUGIN_NAME} PRIVATE ${LIBRARY})
    set_target_properties(${PLUGIN_NAME} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    )
    install(TARGETS ${PLUGIN_NAME} LIBRARY DESTINATION lib)
    add_dependencies(examples ${PLUGIN_NAME})
endforeach()

foreach(SRC ${EXAMPLE_SOURCES} ${ROOT_EXAMPLES})
    get_filename_component(EXAMPLE_NAME ${SRC} NAME_WE)
    message_color("   ./cxx/${EXAMPLE_NAME}")

    add_executable(${EXAMPLE_NAME} ${SRC})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE ${LIBRARY})

    if(${SRC} IN_LIST ROOT_EXAMPLES)
        target_include_directories(${EXAMPLE_NAME} PRIVATE ${ROOT_INCLUDE_DIRS})
        target_link_libraries(${EXAMPLE_NAME} PRIVATE ${ROOT_LIBRARIES})
    endif()

    set_target_properties(${EXAMPLE_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    )

    # toy_sensor loads the toy plugin by path; give it the built library as the
    # default so `./toy_sensor` runs from anywhere in the build tree.
    if(EXAMPLE_NAME STREQUAL "toy_sensor" AND TARGET KalmanPlugin_toy)
        target_compile_definitions(${EXAMPLE_NAME} PRIVATE
            KALMAN_TOY_PLUGIN="$<TARGET_FILE:KalmanPlugin_toy>")
        add_dependencies(${EXAMPLE_NAME} KalmanPlugin_toy)
    endif()

    add_dependencies(examples ${EXAMPLE_NAME})
endforeach()
