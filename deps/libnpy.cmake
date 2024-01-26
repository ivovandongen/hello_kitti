include_guard(GLOBAL)

add_library(libnpy INTERFACE)
target_include_directories(libnpy SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/deps/libnpy/include")