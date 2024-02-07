add_module(comma)
find_package(BZip2 REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS videoio)
target_link_libraries(comma PUBLIC common cereal CapnProto::capnp-json BZip2::BZip2 opencv_videoio)