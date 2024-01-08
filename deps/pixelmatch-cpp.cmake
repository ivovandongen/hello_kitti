include_guard()

add_library(pixelmatch-cpp INTERFACE)
target_include_directories(pixelmatch-cpp SYSTEM INTERFACE ${CMAKE_SOURCE_DIR}/deps/pixelmatch-cpp/include)

# Organize our build
set_target_properties(pixelmatch-cpp PROPERTIES FOLDER deps)