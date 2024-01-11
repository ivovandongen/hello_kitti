add_module(kitti_player)
find_package(OpenCV REQUIRED COMPONENTS highgui)
target_link_libraries(kitti_player PUBLIC ml kitti_parser opencv_highgui cxxopts)