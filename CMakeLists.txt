cmake_minimum_required(VERSION 3.15)
project(hello-kitti)
set(CMAKE_CXX_STANDARD 17)

option(ENABLE_TESTS "Enable tests" ON)

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
include(deps/onnxruntime.cmake)

add_subdirectory(models)
add_subdirectory(modules)

# Add tests
if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
