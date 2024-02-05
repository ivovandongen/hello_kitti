cmake_minimum_required(VERSION 3.15)
project(hello-kitti)
set(CMAKE_CXX_STANDARD 17)

option(ENABLE_TESTS "Enable tests" ON)
option(UPDATE_PYTHON_DEPS "Update python dependencies" ON)
option(ML_BACKEND_VERBOSE_LOGS "Verbose ML Backend logging" OFF)

# Use a virtual env for our python dependencies
include(${PROJECT_SOURCE_DIR}/cmake/python.cmake)
setup_python()

include(FetchContent)
set(FETCHCONTENT_QUIET FALSE)
include(deps/onnxruntime.cmake)
include(deps/kitty_parser.cmake)
include(deps/cxxopts.cmake)
include(deps/libnpy.cmake)
include(deps/colormap.cmake)
include(deps/capnproto.cmake)
include(deps/cereal.cmake)

add_subdirectory(data)
add_subdirectory(models)
add_subdirectory(modules)

# Add tests
if (ENABLE_TESTS)
    enable_testing()
    add_subdirectory(test)
endif ()
