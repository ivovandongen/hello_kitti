include(${PROJECT_SOURCE_DIR}/deps/googletest.cmake)

function(add_test_module)
    # Determine test exe name
    get_filename_component(MODULE_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
    string(REPLACE " " "_" MODULE_NAME ${MODULE_NAME})
    set(MODULE_ON_TEST ${MODULE_NAME})
    set(MODULE_NAME ${MODULE_NAME}_tests)

    message(STATUS "TESTS: Adding test module ${MODULE_NAME}")

    # Find all source files
    file(GLOB_RECURSE SRC_FILES
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")


    # Fixtures Dir
    set(FIXTURES_DIR ${CMAKE_SOURCE_DIR}/test/fixtures)

    # Common test files (driver)
    set(COMMON_TEST_DIR ${CMAKE_SOURCE_DIR}/test/_test)

    # Create the test executable
    add_executable(${MODULE_NAME}
            ${COMMON_TEST_DIR}/test.hpp
            ${COMMON_TEST_DIR}/main.cpp

            ${SRC_FILES}
    )
    target_include_directories(${MODULE_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${COMMON_TEST_DIR})
    target_link_libraries(${MODULE_NAME} PUBLIC gtest gmock ${MODULE_ON_TEST})
    target_compile_definitions(${MODULE_NAME} PRIVATE FIXTURES_DIR=\"${FIXTURES_DIR}\")
    target_compile_definitions(${MODULE_NAME} PRIVATE MODELS_DIR=\"${CMAKE_SOURCE_DIR}/models\")

    # Add a CTest entry
    add_test(NAME ${MODULE_NAME} COMMAND ${MODULE_NAME})
endfunction()

file(GLOB dirs "${CMAKE_CURRENT_SOURCE_DIR}/*")
foreach (dir ${dirs})
    if (IS_DIRECTORY ${dir})
        # Add the test module
        add_subdirectory(${dir})
    endif ()
endforeach ()