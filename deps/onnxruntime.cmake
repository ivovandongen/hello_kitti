include_guard(GLOBAL)

set(onnxruntime_GENERATE_TEST_REPORTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_COREML ON CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_SHARED_LIB ON CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_FOR_NATIVE_MACHINE ON CACHE BOOL "" FORCE)

# https://github.com/protocolbuffers/protobuf/issues/12292#issuecomment-1529680040
find_package(Protobuf REQUIRED CONFIG)

add_subdirectory(${PROJECT_SOURCE_DIR}/deps/onnxruntime/cmake ${CMAKE_BINARY_DIR}/deps/onnxruntime EXCLUDE_FROM_ALL SYSTEM)

# Copy public headers to the binary dir (not part of the include directories..)
get_target_property(ONNXRUNTIME_PUBLIC_HEADERS onnxruntime PUBLIC_HEADER)
get_target_property(ONXXRUNTIME_BINARY_DIR onnxruntime BINARY_DIR)
foreach(h_ ${ONNXRUNTIME_PUBLIC_HEADERS})
    get_filename_component(HEADER_NAME ${h_} NAME)
    configure_file(${h_} "${ONXXRUNTIME_BINARY_DIR}/include/${HEADER_NAME}" COPYONLY)
endforeach()

target_include_directories(onnxruntime PUBLIC $<BUILD_INTERFACE:${ONXXRUNTIME_BINARY_DIR}/include/>)