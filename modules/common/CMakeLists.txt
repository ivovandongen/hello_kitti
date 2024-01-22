add_module(common)
find_package(OpenCV REQUIRED COMPONENTS core calib3d)
target_link_libraries(common PUBLIC opencv_core opencv_calib3d)