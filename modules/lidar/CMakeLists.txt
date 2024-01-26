add_module(lidar)
find_package(OpenCV REQUIRED COMPONENTS core)
target_link_libraries(lidar PUBLIC common colormap opencv_core)