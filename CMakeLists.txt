cmake_minimum_required(VERSION 3.18)
project(CuMatMul LANGUAGES CXX CUDA)

# Set C++ and CUDA standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-lineinfo)


# Pybind11 and Python
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
add_subdirectory(pybind11)

include_directories(include)

# Create the shared library for Python
add_library(cumatmul MODULE
    src/matmul.cu
    src/bindings.cpp
)

# Properties for Python module
set_target_properties(cumatmul PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX "" SUFFIX ".pyd"
)

target_link_libraries(cumatmul PRIVATE pybind11::module)