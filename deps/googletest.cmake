set(GTEST_HAS_ABSL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
        googletest
        GIT_REPOSITORY git@github.com:google/googletest.git
        GIT_TAG        v1.14.0
        OVERRIDE_FIND_PACKAGE
)

FetchContent_MakeAvailable(googletest)