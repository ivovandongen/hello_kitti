
find_package (Python COMPONENTS Interpreter)

add_custom_target(
    yolo_generate_headers ALL
    COMMAND ${PYTHON_EXECUTABLE} yolo_generate_classnames_header.py > ${PROJECT_SOURCE_DIR}/modules/ml/include/ml/yolo/yolo_classes.hpp
    BYPRODUCTS ${PROJECT_SOURCE_DIR}/modules/ml/include/ml/yolo/yolo_classes.hpp
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating coco class header for Yolo"
)

add_custom_target(
    yolo_export_model ALL
    COMMAND ${PYTHON_EXECUTABLE} yolo_export_model.py
    BYPRODUCTS ${PROJECT_SOURCE_DIR}/models/yolo/yolov8n.onnx ${PROJECT_SOURCE_DIR}/models/yolo/yolov8n.mlpackage
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Exporting Yolo model"
)