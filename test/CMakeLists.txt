cmake_minimum_required(VERSION 2.8.8)

add_executable(testPnP testPnP.cpp)
target_link_libraries(testPnP ${thirdparty_libraries})