cmake_minimum_required(VERSION 3.6)

set(CMAKE_CONFIGURATION_TYPES Debug Release CACHE TYPE INTERNAL FORCE)

project(stereo_reconstruction)

# Options
set(LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../libs CACHE PATH "Path to lib folder")
set(Eigen3_INCLUDE_DIR ${LIBRARY_DIR}/Eigen/ CACHE PATH "Path to Eigen source folder")

# Set C++ flags
set(CMAKE_CXX_STANDARD 14)

# Find OpenCV

find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})


# Define header and source files
set(HEADERS
    Eigen.h
)

set(SOURCES
    main.cpp
)

set(APP_LIBS
    ${OpenCV_LIBS})

add_executable(stereo_reconstruction ${HEADERS} ${SOURCES})
target_include_directories(stereo_reconstruction PUBLIC ${Eigen3_INCLUDE_DIR} ${FreeImage_INCLUDE_DIR})
target_link_libraries(stereo_reconstruction ${APP_LIBS})

# Visual Studio properties
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT stereo_reconstruction)
set_property(TARGET stereo_reconstruction PROPERTY VS_DEBUGGER_WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/)
