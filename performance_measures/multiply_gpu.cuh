#ifndef MULTIPLY_GPU_H
#define MULTIPLY_GPU_H
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "cuda_err_check.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

__global__ void naive_mult(float *a, float *b, float *c, int nx);
extern "C" void naive_mult_driver(int deviceId, int nx, int ny);

#endif // !MULTIPLY_GPU_H
