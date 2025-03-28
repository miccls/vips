cmake_minimum_required(VERSION 3.14)

set(This Tests)

set(Sources test_lp_model.cpp)

add_executable(${This} ${Sources})
target_link_libraries(${This} PUBLIC
    gtest_main
    LpOpt
)

add_test(
    NAME ${This}
    COMMAND ${This}
)