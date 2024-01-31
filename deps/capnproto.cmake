include_guard(GLOBAL)

FetchContent_Declare(capnproto
        SOURCE_DIR ${CMAKE_SOURCE_DIR}/deps/capnproto
        BINARY_DIR ${CMAKE_BINARY_DIR}/capnproto
        SYSTEM OVERRIDE_FIND_PACKAGE
)
set(BUILD_TESTING OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(capnproto)