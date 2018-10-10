
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>  // used for rand()

#include "cuda_err_check.h"


//============================= CPU ===========================================
/*
This function is called by the CPU
*/
/*
 * function: add_vec
 * purpose: add two vectors on CPU
 * PARAMETERS:
 *  a - first array
 *  b - second array
 *  c - output array
 *  N - size of the array
 */
void cpu_add(float *a, float *b, float *c, int N) {
	for (unsigned int i = 0; i < N; i++) {
		c[i] = a[i] + b[i];
	}
}


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

__global__ void init(float *x, float *y, unsigned long long N) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}
}


int main(){
	// need to "wake up" the api. This adsorbs the startup overhead that was
	// biasing my results
	cudaFree(0);

	// size of the array
    //const unsigned long long N = 1000000000;
    const unsigned long long N = 100000000;
    //const unsigned long long N = 1000000;
   //const unsigned long long N = 10000;
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

	// do a rough estimate of memory needed in stack
	printf("RAM needed estimate: %lu Mbytes\n", sizeof(float)*N * 6 / 1000000);

	// allocate memory. Must use malloc() when dealing with arrays this large
	// note this is allocating on the host computer.
	float *a = (float *)malloc(N * sizeof(float));
	float *b = (float *)malloc(N * sizeof(float));
	float *c = (float *)malloc(N * sizeof(float));
	float *d_a, *d_b, *d_c;

	// using max number of threads in the x dim possible
	int nThreads = prop.maxThreadsDim[0];
	printf("nThreads: %d\n", nThreads);

	// calculate number of blocks based on the number of threads
	int nBlocks = (N + nThreads - 1) / nThreads;
	printf("nBlocks: %d\n", nBlocks);

	// allocate memory once before the iterations
//	float *a, *b, *c;
//	CudaSafeCall( cudaMallocManaged(&a, sizeof(float) * N) );
//	CudaSafeCall( cudaMallocManaged(&b, sizeof(float) * N) );
//	CudaSafeCall( cudaMallocManaged(&c, sizeof(float) * N) );

	CudaSafeCall( cudaMalloc((void**) &d_a, sizeof(float) * N) );
	CudaSafeCall( cudaMalloc((void**) &d_b, sizeof(float) * N) );
	CudaSafeCall( cudaMalloc((void**) &d_c, sizeof(float) * N) );
	//cudaMalloc((float**) &d_a, sizeof(float) * N);
	//cudaMalloc((float**) &d_b, sizeof(float) * N);
	//cudaMalloc((float**) &d_c, sizeof(float) * N);
	printf("Allocated memory on the Device for a, b, and c . . .\n");
	
	//CudaSafeCall(cudaDeviceSynchronize());

	// create vectors.
	for (unsigned long long i = 0; i < N; i++) {
		// actual values don't matter, as long as they're floats.
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
	printf("Done assigning values.\n");
	//CudaSafeCall( cudaMemset(a, 0, N * sizeof(float)) );
	//CudaSafeCall( cudaMemset(b, 0, N * sizeof(float)) );

	//CudaSafeCall( cudaDeviceSynchronize() );

	// copy memory from CPU to GPU
	cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	printf("Running on GPU\n");
	for (int j = 0; j < iterations; j++) {

		// run the kernel
		cuda_add<<<nBlocks, nThreads>>>(d_a, d_b, d_c, N);
		//init<<<nBlocks, nThreads>>>(d_a, d_b, N);
		CudaCheckError();

		//cuda_add<<<nBlocks, nThreads>>>(a, b, c, N);
		//CudaCheckError();
		// copy the result from the device back to the host
		cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

		// wait for device to finish
		CudaSafeCall( cudaDeviceSynchronize() );

		// calculate the error.
		float maxError = 0.0f;
		printf("Testing for errors . . . \n");
		for (unsigned long long i = 0; i < N; i++) {
			maxError = abs(c[i] - 2.0f - 1.0f);
		}
		if (maxError != 0.0) {
			printf("Max error: %f\n", maxError);
		}
	}

	printf("Running on CPU\n");

	for (int j = 0; j < iterations; j++) {
		cpu_add(a, b, c, N); // add the vectors
	}

	printf("Done!\n");

	// ============== END ==================
    return 0;
}
