cmake_minimum_required(VERSION 3.10)
project(WaferDefectX_Core)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# We will create a library for the algorithms and an executable driver
add_library(wafer_core 
    preprocess.cpp 
    defect_localization.cpp
)

add_executable(WaferDefectX_Run main.cpp)

target_link_libraries(wafer_core ${OpenCV_LIBS})
target_link_libraries(WaferDefectX_Run wafer_core ${OpenCV_LIBS})
