add_module(kitti_lidar_depth)
find_package(OpenCV REQUIRED COMPONENTS highgui)
target_link_libraries(kitti_lidar_depth PUBLIC ml lidar kitti kitti_parser opencv_highgui cxxopts)