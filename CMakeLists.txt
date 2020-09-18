cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH};/usr/local/lib/cmake/opencv4;
    ${CMAKE_SOURCE_DIR}/cmake;)
project(inference_in_trt)

find_package(CUDA REQUIRED)
find_package(TENSORRT REQUIRED)
find_package(Threads)
find_package(OpenCV REQUIRED)

# TARGETS ------------------------------------------
add_executable(inference_in_trt inference_in_trt.cpp)
target_include_directories(inference_in_trt PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS})
target_link_libraries(inference_in_trt PUBLIC ${CUDA_LIBRARIES}  ${TENSORRT_LIBRARIES})
