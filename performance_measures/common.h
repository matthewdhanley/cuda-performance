#ifndef COMMON_H
#define COMMON_H
#include <assert.h>
#include <stdio.h>
#include <math.h>

#pragma once
void postprocess(const float *reference, const float *result, int reads_writes, float ms, int n, int num_loops);
#endif