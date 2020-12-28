cmake_minimum_required(VERSION 2.8.8)

add_executable(test_surf surf_test.cpp)
target_link_libraries(test_surf ${thirdparty_libraries} myCV_lib)