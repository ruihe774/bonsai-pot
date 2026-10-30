# Out-of-tree build of the reference dumper against a built PrismML llama.cpp fork.
#
#   cmake -S scripts/llama_dump -B target/llama_dump \
#         -DLLAMA_CPP_DIR=/path/to/prismml-llama.cpp -DLLAMA_BUILD_DIR=/path/to/prismml-llama.cpp/build-vk
#   cmake --build target/llama_dump
#
# The fork must be configured with -DBUILD_SHARED_LIBS=ON.
cmake_minimum_required(VERSION 3.16)
project(llama_dump CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(LLAMA_CPP_DIR "" CACHE PATH "PrismML llama.cpp source tree")
set(LLAMA_BUILD_DIR "${LLAMA_CPP_DIR}/build" CACHE PATH "PrismML llama.cpp build tree")
if(NOT LLAMA_CPP_DIR)
    message(FATAL_ERROR "set -DLLAMA_CPP_DIR=/path/to/prismml-llama.cpp")
endif()

find_library(LLAMA_LIB llama PATHS "${LLAMA_BUILD_DIR}/bin" "${LLAMA_BUILD_DIR}/src" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIB ggml PATHS "${LLAMA_BUILD_DIR}/bin" "${LLAMA_BUILD_DIR}/ggml/src" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIB ggml-base PATHS "${LLAMA_BUILD_DIR}/bin" "${LLAMA_BUILD_DIR}/ggml/src" NO_DEFAULT_PATH REQUIRED)

add_executable(llama_dump llama_dump.cpp)
target_include_directories(llama_dump PRIVATE "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include")
target_link_libraries(llama_dump PRIVATE ${LLAMA_LIB} ${GGML_LIB} ${GGML_BASE_LIB})
get_filename_component(LLAMA_LIB_DIR ${LLAMA_LIB} DIRECTORY)
set_target_properties(llama_dump PROPERTIES BUILD_RPATH "${LLAMA_LIB_DIR}")
