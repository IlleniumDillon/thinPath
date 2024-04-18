cmake_minimum_required(VERSION 3.0)
project(thinPath)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(thinPath main.cpp)
target_link_libraries(thinPath ${OpenCV_LIBS})

