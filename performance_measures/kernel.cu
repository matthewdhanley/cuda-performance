
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>  // used for rand()

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


//============================= GPU Kernel ====================================
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
__global__ void cuda_add(float *a, float *b, float *c, int N) {
	// assign tid by using block id, block dimension, and thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// do the operations
	if (tid < N) {
		c[tid] = a[tid] + b[tid];
	}
}


int main(){
	// size of the array
    const int N = 1000000000;
	printf("Array Size: %d\n", N);

	// how many times to run it?
	int iterations = 10;

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

	// set up timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// using max number of threads in the x dim possible
	int nThreads = prop.maxThreadsDim[0];
	printf("nThreads: %d\n", nThreads);

	// calculate number of blocks based on the number of threads
	int nBlocks = (N + nThreads - 1) / nThreads;
	printf("nBlocks: %d\n", nBlocks);


	// init the timer variable
	float millis = 0;

	// open file for keeping track of stats.
	FILE *f = fopen("gpu_add_times.txt", "a");

	// error handling for file open
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	printf("\nRunning %d iterations of add on CPU and GPU\n", iterations);
	printf("Saving execution times to gpu_add_times.txt and cpu_add_times.txt\n");

	// allocate memory once before the iterations
	cudaMalloc((void **)&d_a, sizeof(float) * N);
	cudaMalloc((void **)&d_b, sizeof(float) * N);
	cudaMalloc((void **)&d_c, sizeof(float) * N);
	printf("Allocated memory on the Device for a, b, and c . . .\n");

	float avg = 0;

	// create vectors.
	for (unsigned int i = 0; i < N; i++) {
		// actual values don't matter, as long as they're floats.
		a[i] = 1.0f;
		b[i] = 2.0f;
	}

	// copy memory from CPU to GPU
	cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

	printf("Running on GPU\n");
	for (int j = 0; j < iterations; j++) {

		// start timer
		cudaEventRecord(start);

		// run the kernel
		cuda_add<<<nBlocks, nThreads>>>(d_a, d_b, d_c, N);

		// stop the timer
		cudaEventRecord(stop);

		// copy the result from the device back to the host
		cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

		// sync the events
		cudaEventSynchronize(stop);

		// calc elapsed time and print to file
		cudaEventElapsedTime(&millis, start, stop);
		millis = millis / 1000;
		fprintf(f, "%d,%f\n", N, millis);
		avg = avg + millis;

		// calculate the error.
		float maxError = 0.0f;
		for (int i = 0; i < N; i++) {
			maxError = abs(c[i] - a[0] - b[0]);
		}
		if (maxError != 0.0) {
			printf("Max error: %f\n", maxError);
		}
	}
	

	avg = (float) avg / (float) iterations;
	printf("Average GPU Time: %fs \n", avg);

	avg = 0;
	FILE *g = fopen("cpu_add_times.txt", "a");

	if (g == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	printf("Running on CPU\n");

	clock_t cpu_start, cpu_stop;
	double cpu_millis;
	double cpu_avg = 0.0;

	for (int j = 0; j < iterations; j++) {
		cpu_start = clock();

		cpu_add(a, b, c, N); // add the vectors

		cpu_stop = clock();
		cpu_millis = ((double)(cpu_stop - cpu_start)) / CLOCKS_PER_SEC;
		fprintf(g, "%d,%f\n", N, cpu_millis); // print to file
		cpu_avg = avg + cpu_millis;
	}

	cpu_avg = cpu_avg / (double)iterations;
	printf("Average CPU Time %f: \n", cpu_avg);

	printf("Done!\n");
	fclose(g); // close the file

	// ============== END ==================
    return 0;
}
