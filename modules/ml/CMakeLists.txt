find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

add_module(ml)
target_link_libraries(ml PUBLIC onnxruntime opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn)
add_dependencies(ml yolo_generate_headers)