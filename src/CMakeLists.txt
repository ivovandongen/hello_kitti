find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(yolo_test
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(yolo_test PRIVATE onnxruntime opencv_core)