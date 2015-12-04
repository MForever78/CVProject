cmake_minimum_required(VERSION 2.8)
project(Personal)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(Display main.cpp)
target_link_libraries(Display ${OpenCV_LIBS})

