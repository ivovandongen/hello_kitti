add_custom_target(
    kitty_download
    COMMAND ${PYTHON_EXECUTABLE} download_kitty.py
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Downloading KITTI data"
)