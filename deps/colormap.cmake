include_guard(GLOBAL)

add_library(colormap INTERFACE)
target_include_directories(colormap SYSTEM INTERFACE "${PROJECT_SOURCE_DIR}/deps/colormap/include")