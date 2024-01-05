include_guard()

set(onnxruntime_GENERATE_TEST_REPORTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(onnxruntime_USE_COREML ON CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_SHARED_LIB ON CACHE BOOL "" FORCE)
set(onnxruntime_BUILD_FOR_NATIVE_MACHINE ON CACHE BOOL "" FORCE)
#set(onnxruntime_BUILD_APPLE_FRAMEWORK ON CACHE BOOL "" FORCE)
#set(onnxruntime_USE_FULL_PROTOBUF OFF CACHE BOOL "" FORCE)
#set(onnxruntime_RUN_ONNX_TESTS ON CACHE BOOL "" FORCE)
#set(onnxruntime_DISABLE_ABSEIL ON)
#set(onnxruntime_ENABLE_PYTHON OFF)
#set(onnxruntime_ENABLE_TRAINING OFF)
#set(ONNX_USE_PROTOBUF_SHARED_LIBS OFF CACHE BOOL "" FORCE)
#set(ONNX_USE_LITE_PROTO OFF CACHE BOOL "" FORCE)

add_subdirectory(${PROJECT_SOURCE_DIR}/deps/onnxruntime/cmake ${CMAKE_BINARY_DIR}/deps/onnxruntime EXCLUDE_FROM_ALL SYSTEM)

# Copy public headers to the binary dir
get_target_property(ONNXRUNTIME_PUBLIC_HEADERS onnxruntime PUBLIC_HEADER)
get_target_property(ONXXRUNTIME_BINARY_DIR onnxruntime BINARY_DIR)
foreach(h_ ${ONNXRUNTIME_PUBLIC_HEADERS})
    get_filename_component(HEADER_NAME ${h_} NAME)
    configure_file(${h_} "${ONXXRUNTIME_BINARY_DIR}/include/${HEADER_NAME}" COPYONLY)
endforeach()

target_include_directories(onnxruntime PUBLIC $<BUILD_INTERFACE:${ONXXRUNTIME_BINARY_DIR}/include/>)