cmake_minimum_required(VERSION 3.10)
project(WaferDefectX_Core)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_library(wafer_core
    preprocess.cpp
    defect_localization.cpp
)
target_include_directories(wafer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wafer_core PUBLIC ${OpenCV_LIBS})

add_executable(WaferDefectX_Run main.cpp)
target_link_libraries(WaferDefectX_Run PRIVATE wafer_core)
