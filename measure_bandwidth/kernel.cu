
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

float theoreticalBandwidth(cudaDeviceProp props)
{
	/*
	This function calculates the theoretical bandwidth of the GPU.
	*/
	
	// theoretical bandwidth = clock rate (Hz) * interface width (in bytes) * 2 (because double data rate)
	int clock_rate = props.clockRate * 1000;  // clock rate in hz
	int mem_interface_width = props.memoryBusWidth / 8;
	int ddr = 2;  // most GPUs will be using double data rate. I.e. can transfer data on rising and falling edge of clock cycles
	printf("clock rate: %d Hz\n", clock_rate);
	printf("memory bus width: %d bytes\n", mem_interface_width);

	float theoretical_bandwidth = (float) clock_rate * (float) mem_interface_width * (float) ddr / (float) pow(10,9);
	printf("theoretical bandwidth: %f GB/s\n\n", theoretical_bandwidth);
	return theoretical_bandwidth;
}

__global__ void effective_bandwidth_kernel(float a, float *x, float *y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		y[i] = a * x[i] + y[i];
	}
}

float effectiveBandwidth(cudaDeviceProp props) {
	// in order to calculate effective bandwidth, we must be able to measure reads and
	// writes. The equation for effective bandwidth is the total number of bytes read
	// per kernel plus the total number of bytes written per kernel divided by the 
	// time taken to execute.

	int N = 20 * (1 << 20);
	printf("running with arrays of size %d.\n",N);

	float *x, *y, *d_x, *d_y, a;

	// allocate space for x and y arrays on the host
	x = (float*)malloc(N * sizeof(float));
	y = (float*)malloc(N * sizeof(float));

	// allocate space on the device
	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));

	for (int i = 0; i < N; i++) {
		// the values of the numbers don't matter, as long as they're floats.
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	a = 3.0f; // constant for multiplication

	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// copy the memory to the device
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// using max number of threads in the x dim possible
	int nThreads = props.maxThreadsDim[0];

	// calculate number of blocks based on the number of threads
	int nBlocks = (N + nThreads-1) / nThreads;

	// start the timer
	cudaEventRecord(start);

	// Launch the kernel
	effective_bandwidth_kernel <<< nBlocks, nThreads >>> (a, d_x, d_y, N);

	// stop the timer
	cudaEventRecord(stop);

	// copy memory back to the host
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	// sync the events
	cudaEventSynchronize(stop);
	float millis = 0;
	cudaEventElapsedTime(&millis, start, stop);

	// calculate the error.
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = max(maxError, abs(y[i] - 3.0f * 1.0f - 2.0f));
	}

	printf("Max error: %f\n", maxError);

	// number of bytes transferred per array read or write
	int numBytesTx = (int) sizeof(float);
	// number of reads/writes
	int numReadWrite = 3;

	float eff_band = (float) N * (float) numBytesTx * (float) numReadWrite / (float) millis / 1.0e6;

	printf("Effective Bandwidth (GB/s): %f\n", eff_band);
	return eff_band;
}

int main()
{
	// ========= GET DEVICE PROPERTIES =====================
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0); // this allows us to query the device
									  // if there are multiple devices, you can
									  // loop thru them by changing the 0

	// ========= CALCULATE THEORETICAL BANDWIDTH ===========
	float theoretical_bandwidth = theoreticalBandwidth(prop);

	float eff_band = effectiveBandwidth(prop);


	return 0;
}
