add_module(comma_player)
find_package(OpenCV REQUIRED COMPONENTS highgui)
target_link_libraries(comma_player PUBLIC common comma opencv_highgui cxxopts)