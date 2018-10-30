#include "theoretical_bandwidth.cuh"

int NUM_REPS = 1000;

__global__ void effective_bandwidth_kernel(float a, float *x, float *y, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;
	while (i < N) {
		y[i] = a * x[i] + y[i];
		i += stride;
	}
}


void cpu_effective_bandwidth(float a, float *x, float *y, int N) {
	for (int i = 0; i < N; i++) {
		y[i] = a * x[i] + y[i];
	}
}


void effectiveBandwidth(int deviceId, int N) {
	// in order to calculate effective bandwidth, we must be able to measure reads and
	// writes. The equation for effective bandwidth is the total number of bytes read
	// per kernel plus the total number of bytes written per kernel divided by the 
	// time taken to execute.

	// ========= GET DEVICE PROPERTIES =====================
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, deviceId); // this allows us to query the device
									  // if there are multiple devices, you can
									  // loop thru them by changing the 0

	float *x, *y, *d_x, *d_y, *d_y_ref, a, *y_ref;

	// allocate space for x and y arrays on the host
	x = (float*) malloc(N * sizeof(float));
	y = (float*) malloc(N * sizeof(float));
	y_ref = (float*) malloc(N * sizeof(float));

	// allocate space on the device
	cudaMalloc(&d_x, N * sizeof(float));
	cudaMalloc(&d_y, N * sizeof(float));
	cudaMalloc(&d_y_ref, N * sizeof(float));

	for (int i = 0; i < N; i++) {
		// the values of the numbers don't matter, as long as they're floats.
		x[i] = 1.0f;
		y[i] = 2.0f;
		y_ref[i] = 2.0f;
	}

	a = 3.0f; // constant for multiplication

	for (int rep = 0; rep < NUM_REPS + 1; rep++) {
		cpu_effective_bandwidth(a, x, y_ref, N);
	}

	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// copy the memory to the device
	cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

	// using max number of threads in the x dim possible
	int nThreads = 128;

	// calculate number of blocks based on the number of threads
	int nBlocks = (N + nThreads - 1) / nThreads;

	// warm up
	effective_bandwidth_kernel <<< nBlocks, nThreads >>> (a, d_x, d_y, N);
	cudaDeviceSynchronize();

	// start the timer
	cudaEventRecord(start,0);
	// Launch the kernel
	for (int rep = 0; rep < NUM_REPS; rep++) {
		effective_bandwidth_kernel <<< nBlocks, nThreads >>> (a, d_x, d_y, N);
	}

	// stop the timer
	cudaEventRecord(stop,0);

	// copy memory back to the host
	cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

	// sync the events
	cudaEventSynchronize(stop);
	float millis = 0;
	cudaEventElapsedTime(&millis, start, stop);
	// ==================================
	// ROUTINE: Effictive Bandwidth
	printf("%20s", "Effective Bandwidth");
	// ==================================
	int numReadWrite = 3;
	postprocess(y_ref, y, numReadWrite, millis, N, NUM_REPS);
	
	printf("%30s %2d, %5s %2d \n", "Threads/Block:", nThreads,
		"Number of Blocks:", nBlocks);

	//// number of bytes transferred per array read or write
	//int numBytesTx = (int) sizeof(float);
	//// number of reads/writes
	//int numReadWrite = 3;

	//float eff_band = (float)N * (float)numBytesTx * (float)numReadWrite / (float)millis / 1.0e6;

	//printf("Effective Bandwidth (GB/s): %f\n", eff_band);
	//return eff_band;
}

void theoreticalBandwidth(int deviceId)
{
	// ========= GET DEVICE PROPERTIES =====================
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, deviceId); // this allows us to query the device
									          // if there are multiple devices, you can
									          // loop thru them by changing the 0
	/*
	This function calculates the theoretical bandwidth of the GPU.
	*/

	// theoretical bandwidth = clock rate (Hz) * interface width (in bytes) * 2 (because double data rate)
	float clock_rate = (float) props.memoryClockRate * 1000.0f;  // clock rate in hz
	int mem_interface_width = props.memoryBusWidth / 8;
	int ddr = 2;  // most GPUs will be using double data rate. I.e. can transfer data on rising and falling edge of clock cycles
	//printf("clock rate: %d Hz\n", clock_rate);
	//printf("memory bus width: %d bytes\n", mem_interface_width);

	float theoretical_bandwidth = clock_rate * (float)mem_interface_width * (float)ddr / (float)pow(10, 9);
	//printf("theoretical bandwidth: %f GB/s\n\n", theoretical_bandwidth);
	printf("%20s%20.2f\n", "Theoretical", theoretical_bandwidth);

	//return theoretical_bandwidth;
}