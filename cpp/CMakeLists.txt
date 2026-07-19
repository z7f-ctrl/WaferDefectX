cmake_minimum_required(VERSION 3.10)
project(WaferDefectX_Core)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

find_library(ONNXRUNTIME onnxruntime HINTS /opt/homebrew/lib /usr/local/lib)
find_path(ONNXRUNTIME_INCLUDE onnxruntime/onnxruntime_c_api.h
          HINTS /opt/homebrew/include /usr/local/include)

add_library(wafer_core
    preprocess.cpp
    defect_localization.cpp
    feature_extraction.cpp
)
target_include_directories(wafer_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(wafer_core PUBLIC ${OpenCV_LIBS})

add_library(onnx_classifier STATIC onnx_classifier.cpp)
target_include_directories(onnx_classifier PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ONNXRUNTIME_INCLUDE}
)
target_link_libraries(onnx_classifier PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME})

add_executable(WaferDefectX_Run main.cpp)
target_link_libraries(WaferDefectX_Run PRIVATE wafer_core onnx_classifier)
