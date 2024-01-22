add_module(kitti)
find_package(OpenCV REQUIRED COMPONENTS core)
target_link_libraries(kitti PUBLIC common yaml-cpp opencv_core)