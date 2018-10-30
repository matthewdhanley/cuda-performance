
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>  // used for rand()

#include "cuda_err_check.h"


//============================= GPU Kernels ====================================
/*
The "__global__" tag tells nvcc that the function will execute on the device
but will be called from the host. Notice that we must use pointers!
*/
/*
 * function: add_vec
 * purpose: add two vectors on GPU
 * PARAMETERS:
 *  a - first array
 *  b - second array
 *  c - output array
 */
__global__ void cuda_add(float *a, float *b, float *c, unsigned long long N) {
	// assign tid by using block id, block dimension, and thread id
	unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;

	// stride is for big arrays, i.e. bigger than threads we have
	unsigned long long stride = blockDim.x * gridDim.x;

	// do the operations
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += stride;
	}
}


int main() {
	// need to "wake up" the api. This adsorbs the startup overhead that was
	// biasing my results
	cudaFree(0);

	// size of the array
	//const unsigned long long N = 1000000000;
	//const unsigned long long N = 100000000;
	//const unsigned long long N = 1000000;
	const unsigned long long N = 2;
	printf("Array Size: %d\n", N);

	// how many times to run it?
	int iterations = 1;

	// DEVICE PROPERTIES
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // this allows us to query the device
									   // if there are multiple devices, you can
									   // loop thru them by changing the 0

	// print a couple of many properties
	printf("Max Threads per block: %d\n", prop.maxThreadsPerBlock);

	printf("Max Grid Size: %d x %d x %d\n", prop.maxGridSize[0],
		prop.maxGridSize[1], prop.maxGridSize[2]);

	// using max number of threads in the x dim possible
	int nThreads = prop.maxThreadsDim[0];
	printf("nThreads: %d\n", nThreads);

	// calculate number of blocks based on the number of threads
	int nBlocks = (N + nThreads - 1) / nThreads;
	printf("nBlocks: %d\n", nBlocks);

	float *a, *b, *c;
	a = 0;
	b = 0;
	c = 0;
	//// allocate memory once before the iterations
	//CudaSafeCall( cudaMallocManaged(&a, sizeof(float) * N) );
	//CudaSafeCall( cudaMallocManaged(&b, sizeof(float) * N) );
	//CudaSafeCall( cudaMallocManaged(&c, sizeof(float) * N) );
	CudaSafeCall(cudaDeviceSynchronize());

	//float *a, *b, *c;
	cudaMallocManaged(&a, N * sizeof(float));
	CudaSafeCall(cudaDeviceSynchronize());

	cudaMallocManaged(&b, N * sizeof(float));
	CudaSafeCall(cudaDeviceSynchronize());

	cudaMallocManaged(&c, N * sizeof(float));
	CudaSafeCall(cudaDeviceSynchronize());

	cudaPointerAttributes ptrAttr;
	CudaSafeCall(cudaPointerGetAttributes(&ptrAttr, a));

	//float *testp = (float *) ptrAttr.hostPointer;

	//printf("data: %f\n", testp[0]);

	// create vectors.
	for (unsigned long long i = 0; i < N; i++) {
		// actual values don't matter, as long as they're floats.
		CudaSafeCall(cudaDeviceSynchronize());
		a[i] = 1.0f;
		CudaSafeCall(cudaDeviceSynchronize());
		b[i] = 2.0f;
	}
	CudaSafeCall(cudaDeviceSynchronize());

	printf("Pointer to nThreads: 0x%p\n", &nThreads);
	printf("Pointer to nBlocks: 0x%p\n", &nBlocks);
	printf("Pointer to a: 0x%p\n", a);
	printf("Pointer to b: 0x%p\n", b);
	printf("Pointer to c: 0x%p\n", c);


	//CudaSafeCall( cudaMalloc((void**) &d_a, sizeof(float) * N) );
	//CudaSafeCall( cudaMalloc((void**) &d_b, sizeof(float) * N) );
	//CudaSafeCall( cudaMalloc((void**) &d_c, sizeof(float) * N) );

	//CudaSafeCall( cudaMemset(a, 0, N * sizeof(float)) );
	//CudaSafeCall( cudaMemset(b, 0, N * sizeof(float)) );

	printf("Running on GPU\n");
	for (int j = 0; j < iterations; j++) {

		// run the kernel
		cuda_add <<< nBlocks, nThreads >>> (a, b, c, N);
		CudaCheckError();

		// wait for device to finish
		CudaSafeCall(cudaDeviceSynchronize());

		// calculate the error.
		float maxError = 0.0f;
		printf("Error: %f\n", (c[0] - a[0] - b[0]));
		//printf("Testing for errors . . . \n");
		//for (unsigned long long i = 0; i < N; i++) {
		//	maxError = abs(c[i] - 2.0f - 1.0f);
		//}
		//if (maxError != 0.0) {
		//	printf("Max error: %f\n", maxError);
		//}
	}
	CudaSafeCall(cudaDeviceSynchronize());

	CudaSafeCall( cudaFree(a) );
	CudaSafeCall(cudaDeviceSynchronize());

	CudaSafeCall( cudaFree(b) );
	CudaSafeCall(cudaDeviceSynchronize());

	CudaSafeCall( cudaFree(c) );
	//free(a);
	//free(b);
	//free(c);
	// ============== END ==================
	return 0;
}
