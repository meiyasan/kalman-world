
# Version used during development
cmake_minimum_required(VERSION 3.20)
project(${PROJECT_NAME}/Tests)

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

message_title("Tests")

# These are plain executables rather than gtest cases: the project's test
# dependency is pytest (`make tests`), and libKalman itself has no required
# dependency at all — adding GTest just to assert on a few matrices would make
# the C++ build harder to satisfy than the library it tests. Each binary returns
# the number of failures, which is all ctest needs.
if(NOT TARGET ctests)
    add_custom_target(ctests
        COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure $(ARGS)
        COMMENT "Running the libKalman C++ tests"
    )
endif()

file(GLOB TEST_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cc")
list(SORT TEST_SOURCES)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${SRC} NAME_WE)
    message_color("   ./cxx/${TEST_NAME}")

    add_executable(${TEST_NAME} ${SRC})
    target_include_directories(${TEST_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(${TEST_NAME} PRIVATE ${LIBRARY})

    # A Release build inherits CMAKE_BUILD_WITH_INSTALL_RPATH from
    # ProjectArchitecture, which points these binaries at the install prefix -- so
    # `ctest` in a fresh build tree would fail to find libKalman until after
    # `make install`. Point the tests at the build tree instead; running the suite
    # must never require installing first.
    set_target_properties(${TEST_NAME} PROPERTIES
        SKIP_BUILD_RPATH FALSE
        BUILD_WITH_INSTALL_RPATH FALSE
        BUILD_RPATH "${CMAKE_BINARY_DIR}/lib"
    )

    # The factory test exercises LoadPlugin() on the example plugin when the
    # examples are built; without them that part of the test reports itself as
    # skipped rather than failing.
    if(TEST_NAME STREQUAL "06_factory" AND TARGET KalmanPlugin_example)
        target_compile_definitions(${TEST_NAME} PRIVATE
            KALMAN_EXAMPLE_PLUGIN="$<TARGET_FILE:KalmanPlugin_example>")
        add_dependencies(${TEST_NAME} KalmanPlugin_example)
    endif()

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    add_dependencies(ctests ${TEST_NAME})
endforeach()
