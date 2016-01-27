cmake_minimum_required(VERSION 2.8)
project(Camera)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Boost COMPONENTS system filesystem REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

add_executable(Camera main.cpp)
target_link_libraries(Camera ${OpenCV_LIBS} ${Boost_LIBRARIES})

set_property(TARGET Camera PROPERTY CXX_STANDARD 11)
set_property(TARGET Camera PROPERTY CXX_STANDARD_REQUIRED ON)

