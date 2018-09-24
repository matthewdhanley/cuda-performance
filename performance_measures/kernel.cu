
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
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
void cpu_add(int *a, int *b, int *c, unsigned int N) {
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
__global__ void cuda_add(int *a, int *b, int *c, unsigned int N) {
	// init thread id
	int tid;

	// assign tid by using block id, block dimension, and thread id
	tid = blockIdx.x * blockDim.x + threadIdx.x;

	// stride is for big arrays, i.e. bigger than threads we have
	int stride = blockDim.x * gridDim.x;

	// do the operations
	while (tid < N) {
		c[tid] = a[tid] + b[tid];
		tid += stride;
	}
}


int main(){
	// size of the array
    const int N = 1000000;
	printf("Array Size: %d\n", N);

	// max value allowed in each array.
	unsigned int max_value = 100;

	// how many times to run it?
	int iterations = 10000;

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
	printf("RAM needed estimate: %lu Mbytes\n", sizeof(int)*N * 6 / 1000000);

	// allocate memory. Must use malloc() when dealing with arrays this large
	// note this is allocating on the host computer.
	int *a = (int *)malloc(N * sizeof(int));
	int *b = (int *)malloc(N * sizeof(int));
	int *c = (int *)malloc(N * sizeof(int));
	int *d_a, *d_b, *d_c;

	// set up random number generator
	time_t tictoc;
	srand((unsigned) time(&tictoc));

	// set up timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// initialize variables for holding number of blocks and number of threads 
	// to be launched
	unsigned int numBlocks, numThreads;

	numThreads = prop.maxThreadsPerBlock;
	numBlocks = (N + numThreads - 1) / numThreads;  // this rounds up to make sure
													// enough blocks are launched

	// now need to check that we are not launching a grid larger than availible
	if (numBlocks > prop.maxGridSize[1]) {
		printf("Could not launch a grid of size %lu, exceeds max grid size dimension\n\
			    of %d. Launching with %d instead.", numBlocks, prop.maxGridSize[1], 
				prop.maxGridSize[1]);
	}

	// init the timer variable
	float millis = 0;

	// open file for keeping track of stats.
	FILE *f = fopen("gpu_add_times.txt", "w");

	// error handling for file open
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	printf("\nRunning %d iterations of add on CPU and GPU\n", iterations);
	printf("Saving execution times to gpu_add_times.txt and cpu_add_times.txt\n");

	// allocate memory once before the iterations
	cudaMalloc((void **)&d_a, sizeof(int) * N);
	cudaMalloc((void **)&d_b, sizeof(int) * N);
	cudaMalloc((void **)&d_c, sizeof(int) * N);
	printf("Allocated memory on the Device for a, b, and c . . .\n");

	float avg = 0;

	printf("Running on GPU\n");
	for (int j = 0; j < iterations; j++) {
		// create random vectors.
		for (unsigned int i = 0; i < N; i++) {
			a[i] = rand() % max_value;
			b[i] = rand() % max_value;
		}

		// copy memory from CPU to GPU
		cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice);

		// start timer
		cudaEventRecord(start);

		// run the kernel
		cuda_add<<<numBlocks, numThreads>>>(d_a, d_b, d_c, N);

		// stop the timer
		cudaEventRecord(stop);

		// copy the result from the device back to the host
		cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost);

		// calc elapsed time and print to file
		cudaEventElapsedTime(&millis, start, stop);
		fprintf(f, "%f \n", millis);
		avg = avg + millis;
	}

	avg = avg / (float) iterations;
	printf("Average GPU Time %f: \n", avg);

	avg = 0;
	FILE *g = fopen("cpu_add_times.txt", "w");

	if (g == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	printf("Running on CPU\n");

	for (int j = 0; j < iterations; j++) {
		for (unsigned int i = 0; i < N; i++) {
			a[i] = rand() % max_value;
			b[i] = rand() % max_value;
		}

		cudaEventRecord(start); // start timer
		cpu_add(a, b, c, N); // add the vectors
		cudaEventRecord(stop); // stop the timer
		cudaEventSynchronize(stop); // sync
		cudaEventElapsedTime(&millis, start, stop); // calc time
		fprintf(g, "%f\n", millis); // print to file
		avg = avg + millis;
	}

	avg = avg / (float)iterations;
	printf("Average CPU Time %f: \n", avg);

	printf("Done!\n");
	fclose(g); // close the file

	// ============== END ==================
    return 0;
}
