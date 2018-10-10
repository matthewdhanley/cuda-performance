cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# define the name of the project
project(adding)

# find the required packages, here it's cuda
find_package(CUDA REQUIRED)

# creating a cuda executable
cuda_add_executable(
    adding
    kernel.cu
    cuda_err_check.h)

message("Done.")
