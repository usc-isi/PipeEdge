cmake_minimum_required(VERSION 3.8)

if(DEFINED PY_BUILD_CMAKE_MODULE_NAME AND DEFINED PY_BUILD_CMAKE_PACKAGE_VERSION)
  set(_IS_PY_BUILD_CMAKE ON)
  set(_VERSION ${PY_BUILD_CMAKE_PACKAGE_VERSION})
else()
  set(_IS_PY_BUILD_CMAKE OFF)
  set(_VERSION 0.0.0)
endif()

project(pipeedge VERSION ${_VERSION}
                 LANGUAGES CXX)

include(GNUInstallDirs)
if(_IS_PY_BUILD_CMAKE)
  message(STATUS "Checking for py-build-cmake environment - found")
  message(STATUS "  Using PEP 427-compatible install paths")
  set(CMAKE_INSTALL_BINDIR "${PY_BUILD_CMAKE_MODULE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/scripts")
else()
  message(STATUS "Checking for py-build-cmake environment - not found")
  message(STATUS "  Using default install paths")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-Wall -Wextra)

find_package(yaml-cpp REQUIRED)

add_executable(sched-pipeline sched-pipeline.cpp schedule.cpp)
if(TARGET yaml-cpp::yaml-cpp)
  # yaml-cpp >= 0.8.0
  target_link_libraries(sched-pipeline PRIVATE yaml-cpp::yaml-cpp)
else()
  # 0.6.0 <= yaml-cpp < 0.8.0
  target_link_libraries(sched-pipeline PRIVATE yaml-cpp)
  # yaml-cpp < 0.6.3 doesn't have INTERFACE_INCLUDE_DIRECTORIES set on the library target
  target_include_directories(sched-pipeline PRIVATE ${YAML_CPP_INCLUDE_DIR})
endif()

install(TARGETS sched-pipeline
        COMPONENT PipeEdge_Runtime
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
