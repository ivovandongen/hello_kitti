cmake_minimum_required(VERSION 3.15)
project(hello-kitti)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
include(deps/onnxruntime.cmake)

add_subdirectory(src)


