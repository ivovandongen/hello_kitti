add_module(stereo)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs stereo)
target_link_libraries(stereo PUBLIC common opencv_core opencv_imgproc opencv_imgcodecs opencv_stereo)