macro(ADD_MODULE MODULE_NAME)

    message(STATUS "MODULES: adding module ${MODULE_NAME}")

    file(GLOB_RECURSE HEADER_FILES
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp")

    file(GLOB_RECURSE SRC_FILES
            RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
            "${CMAKE_CURRENT_SOURCE_DIR}/src/*.mm"
    )

    set(${MODULE_NAME}_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/)

    if (NOT "${SRC_FILES}" STREQUAL "")
        add_library(${MODULE_NAME} ${HEADER_FILES} ${SRC_FILES})

        target_include_directories(${MODULE_NAME} PUBLIC
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
        )
    else ()
        add_library(${MODULE_NAME} INTERFACE)

        target_include_directories(${MODULE_NAME} INTERFACE
                $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                $<INSTALL_INTERFACE:include>
        )
    endif ()

    # Organize our build
    set_target_properties(${MODULE_NAME} PROPERTIES FOLDER modules)

endmacro(ADD_MODULE)

# Modules
add_subdirectory(ml)