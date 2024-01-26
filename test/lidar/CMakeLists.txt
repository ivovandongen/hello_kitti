add_test_module()
find_package(OpenCV REQUIRED COMPONENTS imgproc imgcodecs)
target_link_libraries(lidar_tests PUBLIC common kitti colormap libnpy opencv_imgproc opencv_imgcodecs)