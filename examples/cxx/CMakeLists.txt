
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

set(ROOT_EXAMPLES)
foreach(SRC ${EXAMPLE_SOURCES})
    if(${SRC} MATCHES "_root\\.cc$")
        list(APPEND ROOT_EXAMPLES ${SRC})
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

    add_dependencies(examples ${EXAMPLE_NAME})
endforeach()
