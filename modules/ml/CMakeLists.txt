find_package(onnxruntime REQUIRED)
find_package(OpenCV REQUIRED)

add_module(ml)
target_link_libraries(ml PUBLIC common onnxruntime opencv_core opencv_imgproc opencv_imgcodecs opencv_dnn)
add_dependencies(ml yolo_generate_headers)
if (${ML_BACKEND_VERBOSE_LOGS})
    target_compile_definitions(ml PRIVATE ML_BACKEND_VERBOSE_LOGS=1)
endif ()