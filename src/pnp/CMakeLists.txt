cmake_minimum_required(VERSION 2.8.8)

add_library(myCV_lib mcvPnP.cpp)

target_link_libraries(myCV_lib ${thirdparty_libraries})