
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef N_THREADS
#define N_THREADS 1024
#endif // !N_THREADS


float cpu_inner_product(float *a, float *b, int n)
{
	// initialize the result value as a float equal to zero.
	float result = 0;
	for (int i = 0; i < n; i++) {
		result += a[i] * b[i];  // accumulate into the result.
	}
	return result;
}


__global__ void gpu_inner_product_naive(float *a, float *b, float *result, int n)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x; // get the row
	__shared__ float sharedMem[N_THREADS];
	int sharedIndex = threadIdx.x;        // for indexing into shared memory
	int stride = blockDim.x * gridDim.x;  // striding for large arrays.

	while (index < n) // check to make sure that the thread needs to compute
	{
		sharedMem[sharedIndex] += a[index] * b[index];
		index += stride;
	}

	__syncthreads();  // make sure all threads finish
	// now product holds the sum for the parts that this thread calculated. 
	// We need to combine! How? Reduction!

	for (int i = blockDim.x / 2; i > 0; i /= 2) {
		if (sharedIndex < i) {
			sharedMem[sharedIndex] += sharedMem[sharedIndex + i];
		}
		__syncthreads();  // need to let all threads catch up
	}

	// now each block has its reduction for the portion in sharedMem[0]
	// need to sum together the blocks...

	// only need one representative from each block
	if (sharedIndex == 0) {
		result[blockIdx.x] = sharedMem[0];  // set the corresponging value in result
	}
	__syncthreads();
	for (int i = gridDim.x / 2; i > 0; i /= 2) {
		if (blockIdx.x < i) {
			result[blockIdx.x] += result[blockIdx.x + i];
		}
		__syncthreads();
	}

	__syncthreads();

}


void gpu_driver(int N_max, int iterator) {
	FILE *g = fopen("gpu_inner_prod_times.txt", "w");
	// error handling for file open
	if (g == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	// get device information ====================================
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // this allows us to query the device
									   // if there are multiple devices, you can
									   // loop thru them by changing the 0

	// SET UP TIMER ===============================================
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	for (int N = 1000; N < N_max; N += iterator) {
		float millis = 0;

		// setting up memory =========================================
		// init
		float result, *d_a, *d_b, *d_result;

		// allocate host mem
		//float *a = (float *)calloc(N, sizeof(float));
		//float *b = (float *)calloc(N, sizeof(float));
		// default result to zero
		result = 0.0f;

		// allocate device mem
		cudaMallocManaged((void**)&d_a, N * sizeof(float));
		cudaMallocManaged((void**)&d_b, N * sizeof(float));
		// result is allocated after we determine the number of blocks 

		// populate a and b ==========================================
		for (int i = 0; i < N; i++) {
			a[i] = 2.0f;
			b[i] = 3.0f;
		}

		// transfer a and b to device ================================
		cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

		// CALCULATE nBlocks =========================================
		// calculate number of blocks based on the number of threads
		int nBlocks = (N + N_THREADS - 1) / N_THREADS;

		// don't want to try to launch a grid that is larger than allowed
		if (nBlocks > prop.maxGridSize[0]) {
			nBlocks = prop.maxGridSize[0];
		}

		// we will only need the first index outside of the kernel
		// in the kernel, we need the full dim N to do reduction
		// each block needs its own "bin" to store values for reduction
		cudaMalloc((void**)&d_result, nBlocks * sizeof(float));

		// start timer
		cudaEventRecord(start);

		// LAUNCH =================================================
		gpu_inner_product_naive <<< nBlocks, (int) N_THREADS >>> (d_a, d_b, d_result, N);

		// stop the timer
		cudaEventRecord(stop);
		// sync the events
		cudaEventSynchronize(stop);

		// copy the result from the device back to the host
		// only need the first index, hence only giving size of a single float
		cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

		// calc elapsed time and print to file
		cudaEventElapsedTime(&millis, start, stop);
		millis = millis / 1000;
		fprintf(g, "%d,%f\n", N, millis);

		// Clean up ===============================================
		free(a); 
		free(b);
		//free(&result);
		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_result);
	}

}


void cpu_inner_product(int N_max, int iterator) {
	FILE *g = fopen("cpu_inner_prod_times.txt", "w");
	// error handling for file open
	if (g == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	for (int N = 1000; N < N_max; N += iterator) {
		float *a = (float *)calloc(N, sizeof(float));

		float *b = (float *)calloc(N, sizeof(float));

		for (int i = 0; i < N; i++) {
			a[i] = 2.0f;
			b[i] = 3.0f;
		}
		clock_t cpu_start, cpu_stop;
		double cpu_millis;
		cpu_start = clock();

		// RUN THE INNER PRODUCT ==================
		float result = cpu_inner_product(a, b, N);
		// ========================================

		cpu_stop = clock();
		cpu_millis = ((double)(cpu_stop - cpu_start)) / CLOCKS_PER_SEC;
		fprintf(g, "%d,%f\n", N, cpu_millis);
		//printf("Time to execute inner product of size %d: %lf seconds\n", N, cpu_millis);
		free(a);
		free(b);
	}
	//printf(' \n');
	fclose(g);
}


int main()
{
	int N_max = 100000000;  // max size of the array
	int iterator = N_max / 5;

	// run on the CPU
	printf("running on cpu ...\n");
	cpu_inner_product(N_max, iterator);
	printf("running on gpu ...\n");
	gpu_driver(N_max, iterator);
	
	printf("Done!\n\n");
}
