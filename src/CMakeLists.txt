cmake_minimum_required(VERSION 2.8.8)
add_library(myCV_lib pnp/mcvPnP.cpp pnp/mcvPnP.h
        triangulation/single_triangulation.h triangulation/single_triangulation.cpp
        features/Surf.cpp features/Surf.h features/Sift.cpp features/Sift.h)

target_link_libraries(myCV_lib ${thirdparty_libraries})
