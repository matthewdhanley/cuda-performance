#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>  // used for rand()

#include "adding_gpu.cuh"
#include "adding_cpu.cuh"
#include "common.h"
#include "theoretical_bandwidth.cuh"
#include "transpose_gpu.cuh"
#include "multiply_gpu.cuh"



int main(){
	// need to "wake up" the api. This adsorbs the startup overhead that was
	// biasing my results
	cudaFree(0);

	int device_num = 0;

	// DEVICE PROPERTIES
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, device_num); // this allows us to query the device
									   // if there are multiple devices, you can
									   // loop thru them by changing the 0

	// print a couple of many properties
	printf("=============================================================\n");
	printf("GPU: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
	printf("Max Threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0],
		prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("Clock Rate: %d MHz\n", prop.clockRate);
	printf("Number of SMs: %d\n", prop.multiProcessorCount);
	printf("Shared Memory Per Block: %d\n", (int) prop.sharedMemPerBlock);
	printf("Shared Memory Per SM: %d\n", (int) prop.sharedMemPerMultiprocessor);
	printf("Registers per block: %d\n", prop.regsPerBlock);
	printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
	printf("Warp Size: %d\n", prop.warpSize);
	printf("=============================================================\n\n");


	// ------------
	// time kernels
	// ------------
	printf("%25s%25s\n", "ROUTINE", "BANDWIDTH (GB/s)");
	theoreticalBandwidth(0);

	// size of the array
    //const unsigned long long N = 1000000000;
    //const unsigned long long N = 100000000;
    //const unsigned long long N = 85710165;
    int N = 1000000;
	int nx = 1024;
	int ny = 1024;
	//      2147483647
    //const unsigned long long N = 1000000;
    //const unsigned long long N = 10000;
	//printf("Array Size: %d\n", N);
	effectiveBandwidth(0, N);
	gpu_copy_driver(device_num, N);
	gpu_naive_driver(device_num, N);
	matrix_copy_driver(device_num, nx, ny);
	naive_transpose_driver(device_num, nx, ny);
	shared_transpose_driver(device_num, nx, ny);
	no_bank_conflict_transpose_driver(device_num, nx, ny);
	naive_mult_driver(device_num, nx, ny);

	printf("=============================================================\n\n");
    return 0;
}
