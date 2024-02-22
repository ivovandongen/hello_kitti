include_guard(GLOBAL)

find_package(CapnProto CONFIG QUIET)

FetchContent_Declare(opendbc
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/opendbc
        BINARY_DIR ${CMAKE_BINARY_DIR}/opendbc
        PATCH_COMMAND git apply --ignore-whitespace "${CMAKE_SOURCE_DIR}/deps/opendbc.patch"
        UPDATE_DISCONNECTED 1
        SYSTEM OVERRIDE_FIND_PACKAGE
)
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(opendbc)
target_link_libraries(opendbc PUBLIC CapnProto::capnp cereal)
target_include_directories(opendbc SYSTEM PUBLIC ${CMAKE_SOURCE_DIR}/deps)
target_compile_definitions(opendbc PRIVATE DBC_FILE_PATH="${OPENDBC_DIR}")
