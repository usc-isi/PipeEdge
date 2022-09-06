cmake_minimum_required(VERSION 3.8)

include(GNUInstallDirs)

if(DEFINED PY_BUILD_CMAKE_MODULE_NAME AND DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION)
  set(_VERSION ${PY_BUILD_CMAKE_PACKAGE_VERSION})
  message(STATUS "Checking for py-build-cmake environment - found")
  message(STATUS "  Using PEP 427-compatible install paths")
  set(CMAKE_INSTALL_BINDIR "${PY_BUILD_CMAKE_MODULE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/scripts")
else()
  set(_VERSION 0.0.0)
  message(STATUS "Checking for py-build-cmake environment - not found")
  message(STATUS "  Using default install paths")
endif()

project(pipeedge VERSION ${_VERSION}
                 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

find_package(yaml-cpp REQUIRED)

add_executable(sched-pipeline sched-pipeline.cpp schedule.cpp)
target_link_libraries(sched-pipeline PRIVATE yaml-cpp)

install(TARGETS sched-pipeline
        COMPONENT PipeEdge_Runtime
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
