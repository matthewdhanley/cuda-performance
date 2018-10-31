#!/bin/bash
nvcc adding_cpu.cu common.cpp multiply_gpu.cu transpose_gpu.cu adding_gpu.cu theoretical_bandwidth.cu main.cu -o performance_test -arch=sm_30
