include_guard(GLOBAL)

set(CAPNPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/cereal)
set(CAPNPC_SRC_PREFIX ${CMAKE_SOURCE_DIR}/deps)
capnp_generate_cpp(cereal_cpp_files cereal_h_files
        ${CMAKE_SOURCE_DIR}/deps/cereal/log.capnp
        ${CMAKE_SOURCE_DIR}/deps/cereal/car.capnp
        ${CMAKE_SOURCE_DIR}/deps/cereal/legacy.capnp
        ${CMAKE_SOURCE_DIR}/deps/cereal/custom.capnp
)
add_library(cereal STATIC ${cereal_cpp_files})
target_link_libraries(cereal PUBLIC CapnProto::capnp)
target_include_directories(cereal PUBLIC ${CAPNPC_OUTPUT_DIR})