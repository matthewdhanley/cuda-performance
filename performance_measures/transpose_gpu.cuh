#ifndef TRANSPOSE_GPU_H
#define TRANSPOSE_GPU_H
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "cuda_err_check.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

__global__ void copy_matrix(float *a, float *b, int nx, int ny);
__global__ void naive_transpose(float *a, float *b, int nx, int ny);
__global__ void transpose_shared(float *a, float *b, int nx, int ny);
__global__ void transpose_no_conflict(float *a, float *b, int nx, int ny);
extern "C" void matrix_copy_driver(int deviceId, int nx, int ny);
extern "C" void naive_transpose_driver(int deviceId, int nx, int ny);
extern "C" void shared_transpose_driver(int deviceId, int nx, int ny);
extern "C" void no_bank_conflict_transpose_driver(int deviceId, int nx, int ny);

#endif