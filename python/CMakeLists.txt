cmake_minimum_required(VERSION 3.16...3.26)
project(genz_icp_pybind VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PYBIND11_NEWPYTHON ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  execute_process(COMMAND ${Python_EXECUTABLE} -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR})
  find_package(pybind11 CONFIG REQUIRED)
endif()

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/genz_icp/)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/genz_icp ${CMAKE_CURRENT_BINARY_DIR}/genz_icp)
else()
  message(FATAL_ERROR "Out-of-tree build is not supported yet for genz-icp python package")
endif()

add_subdirectory(genz_icp/pybind)
