include_guard(GLOBAL)

add_library(cxxopts INTERFACE)
target_include_directories(cxxopts SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/deps/cxxopts/include")