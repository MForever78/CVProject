cmake_minimum_required(VERSION 2.8)
project(Harris_Corner)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(Harris main.cpp)
target_link_libraries(Harris ${OpenCV_LIBS})

set_property(TARGET Harris PROPERTY CXX_STANDARD 11)
set_property(TARGET Harris PROPERTY CXX_STANDARD_REQUIRED ON)

