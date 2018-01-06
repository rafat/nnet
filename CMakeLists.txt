cmake_minimum_required(VERSION 2.8.0 FATAL_ERROR)

set(PROJECT_NAME nnet)
project(${PROJECT_NAME} CXX C)


# src root path
set(NNET_SRC_ROOT ${PROJECT_SOURCE_DIR} CACHE PATH "nnet source root")
# binary output by default
set(COMMON_BIN_PATH ${CMAKE_BINARY_DIR}/Bin)
set(LIBRARY_OUTPUT_PATH ${COMMON_BIN_PATH}/${CMAKE_BUILD_TYPE})
set(EXECUTABLE_OUTPUT_PATH ${COMMON_BIN_PATH}/${CMAKE_BUILD_TYPE})
# set where to find additional cmake modules if any
set(CMAKE_MODULE_PATH ${NNET_SRC_ROOT}/cmake ${CMAKE_MODULE_PATH})


set(NNET_VERSION "0.1.0" CACHE STRING "nnet version" FORCE)
message(">>> Building nnet version: ${NNET_VERSION}")
message(">>> EXECUTABLE_OUTPUT_PATH = ${EXECUTABLE_OUTPUT_PATH}")


# cleanup prefix lib for Unix-like OSes
set(CMAKE_SHARED_MODULE_PREFIX)

# install target to this folder by default
set(NNETLIB_BINARY_DIR ${NNET_SRC_ROOT}/bin)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${NNET_BINARY_DIR}" CACHE PATH "default install path" FORCE)
endif()

# make include globaly visible
set(PROJECT_WIDE_INCLUDE ${NNET_SRC_ROOT}/include)
include_directories(${PROJECT_WIDE_INCLUDE})




include_directories(${COMMON_BIN_PATH})


add_subdirectory(src)
add_subdirectory(test) 

install(DIRECTORY ${NNET_SRC_ROOT}/include/ DESTINATION include FILES_MATCHING PATTERN "*.h")