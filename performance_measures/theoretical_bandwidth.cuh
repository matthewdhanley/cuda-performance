#ifndef THEORETICAL_BANDWIDTH_H
#define THEORETICAL_BANDWIDTH_H
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "cuda_err_check.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "common.h"

__global__ void effective_bandwidth_kernel(float a, float *x, float *y, int N);
void theoreticalBandwidth(int deviceId);
void effectiveBandwidth(int deviceId, int N);

#endif