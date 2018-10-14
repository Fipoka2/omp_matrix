cmake_minimum_required(VERSION 3.12)
project(omp_matrix)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

add_executable(omp_matrix main.cpp Matrix.cpp Matrix.h)