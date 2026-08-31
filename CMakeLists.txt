# --- Project basics -----------------------------------------------------------
# 3.18+ gives first-class CUDA support and CMAKE_CUDA_ARCHITECTURES handling.
cmake_minimum_required(VERSION 3.18)

# Declaring CUDA as a project language makes CMake find nvcc and treat every
# .cu file as device code automatically. CXX is your host C++ (main.cpp).
project(inferencing LANGUAGES CXX CUDA)

# --- Language standards --------------------------------------------------------
# Equivalent to the -std=c++17 you passed by hand, for both host and device.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- GPU architecture ----------------------------------------------------------
# This is your -arch=sm_89 (RTX 4060 / Ada = compute capability 8.9).
set(CMAKE_CUDA_ARCHITECTURES 89)

# --- The executable ------------------------------------------------------------
# List both sources; CMake sends main.cpp to cl and kernel.cu to nvcc, then links.
add_executable(infer
    src/main.cpp
    src/kernel.cu
)

# --- Includes ------------------------------------------------------------------
# Your -I include (for json.hpp). Also add src/ so kernel.h is found.
target_include_directories(infer PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# --- CUDA libraries ------------------------------------------------------------
# Finds the toolkit and exposes imported targets like CUDA::cudart, CUDA::cublas.
# cudart is linked for you by the CUDA language, but cuBLAS must be requested.
find_package(CUDAToolkit REQUIRED)
target_link_libraries(infer PRIVATE CUDA::cublas)

# --- MSVC conforming preprocessor ---------------------------------------------
# This is the /Zc:preprocessor fix. For .cu files it must be forwarded through
# nvcc to cl via -Xcompiler; for .cpp files it goes straight to cl.
target_compile_options(infer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
)
