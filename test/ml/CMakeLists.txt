add_test_module()

include(${PROJECT_SOURCE_DIR}/deps/pixelmatch-cpp.cmake)
target_link_libraries(ml_tests PRIVATE pixelmatch-cpp)