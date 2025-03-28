cmake_minimum_required(VERSION 3.14)

set(This LpOpt)

project(${This} C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(gtest_force_shared_crt on)

include(FetchContent)
FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG main)
FetchContent_MakeAvailable(googletest)

enable_testing()

set(EIGEN3_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/deps/eigen-3.4.0")
include_directories(${EIGEN3_INCLUDE_DIR})

set(Headers
    src/lp_model.h
    src/utils.h
)

set(Sources
    src/lp_model.cpp
    src/main.cpp
    src/utils.cpp
)

add_library(${This} STATIC ${Sources} ${Headers})

add_subdirectory(tests)
