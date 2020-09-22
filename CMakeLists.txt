cmake_minimum_required(VERSION 3.15)


find_program(CLANG_C_BINARY clang)
find_program(CLANG_CXX_BINARY clang++)
if(CLANG_C_BINARY STREQUAL CLANG_C_BINARY-NOTFOUND OR CLANG_C_BINARY STREQUAL
                                                      CLANG_CXX_BINARY-NOTFOUND)
  message(
    STATUS
      "Clang is not installed (could not find both `clang` and `clang++`). Using GCC: You will probably get many warnings."
    )
else()
  set(CMAKE_CXX_COMPILER clang++)
  set(CMAKE_C_COMPILER clang)
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH};/usr/local/lib/cmake/opencv4;
    ${CMAKE_CURRENT_LIST_DIR}/cmake;)
project(inference_in_trt)

find_package(CUDA REQUIRED)
find_package(TENSORRT REQUIRED)
find_package(Threads)
find_package(OpenCV REQUIRED)

# TARGETS ------------------------------------------
set(TRT_INCLUDE_DIRS 
	${CMAKE_CURRENT_LIST_DIR}/src/common
)
message(STATUS "TRT_INCLUDE_DIR: ${TRT_INCLUDE_DIRS}")
set(SOURCES
	${TRT_INCLUDE_DIRS}/argsParser.h
	${TRT_INCLUDE_DIRS}/buffers.h
	${TRT_INCLUDE_DIRS}/half.h
	${TRT_INCLUDE_DIRS}/logger.h
	${TRT_INCLUDE_DIRS}/logging.h
	${CMAKE_CURRENT_LIST_DIR}/src/main.cpp 
	${CMAKE_CURRENT_LIST_DIR}/src/inference_in_trt.cpp 
	${CMAKE_CURRENT_LIST_DIR}/src/inference_in_trt.hpp 
)

add_executable(inference_in_trt ${SOURCES})
target_include_directories(inference_in_trt PRIVATE ${TRT_INCLUDE_DIRS})

target_include_directories(inference_in_trt PUBLIC ${CUDA_INCLUDE_DIRS} ${TENSORRT_INCLUDE_DIRS})
target_link_libraries(inference_in_trt PUBLIC ${CUDA_LIBRARIES}  ${TENSORRT_LIBRARIES})
