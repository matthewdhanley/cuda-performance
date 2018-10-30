#ifndef ADDING_GPU_H
#define ADDING_GPU_H

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>  // used for rand()
#include "cuda_err_check.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"


struct kernel_parameters
{
	int num_threads;
	int num_blocks;
};
__global__ void cuda_copy(float *a, float *b, float *c, int N);
__global__ void cuda_add(float *a, float *b, float *c, int N);
__global__ void init(float *x, float *y, int N);
kernel_parameters get_kernel_parameters(int device_num, int num_ops);
extern "C" void gpu_copy_driver(int device_num, int N);
extern "C" void gpu_naive_driver(int device_num, int N);


#endif