# --- Project basics -----------------------------------------------------------
# 3.18+ gives first-class CUDA support and CMAKE_CUDA_ARCHITECTURES handling.
cmake_minimum_required(VERSION 3.18)

# Declaring CUDA as a project language makes CMake find nvcc and treat every
# .cu file as device code automatically. CXX is your host C++ (main.cpp).
project(deltaforge LANGUAGES CXX CUDA)

# --- Language standards --------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# --- GPU architecture ----------------------------------------------------------
# -arch=sm_89 (RTX 4060 / Ada = compute capability 8.9).
set(CMAKE_CUDA_ARCHITECTURES 89)

# Finds the toolkit and exposes imported targets like CUDA::cudart, CUDA::cublas.
find_package(CUDAToolkit REQUIRED)

# --- Core library --------------------------------------------------------------
# Everything reusable: the loader/model headers plus the CUDA kernels. The exes
# below link this, so build settings are declared once, here.
add_library(deltaforge_core STATIC
    src/kernel.cu
)

# PUBLIC = also applied to anything that links deltaforge_core (infer, deltaforge_test),
# which is what makes `#include "model.hpp"` work in those targets without repeating paths.
target_include_directories(deltaforge_core PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
)

# cudart explicitly: the test binary is plain C++ that calls cudaMalloc through
# DeviceBuffer, so it must not rely on cublas pulling the runtime in for it.
target_link_libraries(deltaforge_core PUBLIC CUDA::cudart CUDA::cublas)

# /Zc:preprocessor — conforming MSVC preprocessor. For .cu files it must be forwarded
# through nvcc to cl via -Xcompiler; for .cpp files it goes straight to cl.
target_compile_options(deltaforge_core PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
)

# Repo root as a wide string literal, so tests/tools find model/ and dumps/ no matter
# which directory they are run from.
target_compile_definitions(deltaforge_core PUBLIC
    DELTAFORGE_ROOT=L"${CMAKE_SOURCE_DIR}"
)

# --- Executables ---------------------------------------------------------------
add_executable(infer src/main.cpp)
target_link_libraries(infer PRIVATE deltaforge_core)

add_executable(deltaforge_test tests/test_main.cpp)
target_link_libraries(deltaforge_test PRIVATE deltaforge_core)

# --- Tests ---------------------------------------------------------------------
# ctest --test-dir build -C Debug
# The test's exit code is the verdict: 0 = pass, non-zero = number of failed checks.
enable_testing()
add_test(NAME deltaforge_test COMMAND deltaforge_test)
