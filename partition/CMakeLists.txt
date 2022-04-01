cmake_minimum_required(VERSION 3.8)

project(edgepipe VERSION 0.1.0
                 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

find_package(yaml-cpp REQUIRED)

add_executable(sched-pipeline sched-pipeline.cpp)
target_link_libraries(sched-pipeline PRIVATE yaml-cpp)
