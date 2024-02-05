add_module(comma)
find_package(BZip2 REQUIRED)
target_link_libraries(comma PUBLIC common cereal BZip2::BZip2)