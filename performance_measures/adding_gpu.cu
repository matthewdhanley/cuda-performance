#include "adding_gpu.cuh"

//============================= GPU Kernels ====================================
/*
The "__global__" tag tells nvcc that the function will execute on the device
but will be called from the host. Notice that we must use pointers!
*///============================================================================

int NUMBER_OF_REPS = 1000;


__global__ void cuda_copy(float *a, float *b, float *c, int N) {
	/*
	 * function: cuda_copy
	 * purpose: perform copy ops to get "optimal" bandwidth
	 * this function really does nothing useful other than
	 * setting a theoretical goal for adding optomizations
	 * PARAMETERS:
	 *  a - first array
	 *  b - second array
	 *  c - output array
	 */
	 // assign tid by using block id, block dimension, and thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// stride is for big arrays, i.e. bigger than threads we have
	int stride = blockDim.x * gridDim.x;

	// do the operations
	while (tid < N) {
		// two reads, one write
		c[tid] = a[tid];
		b[tid] = 2.0f;
		tid += stride;
	}
}


__global__ void cuda_add(float *a, float *b, float *c, int N) {
	/*
	 * function: cuda_add
	 * purpose: add two vectors on GPU
	 * PARAMETERS:
	 *  a - first array
	 *  b - second array
	 *  c - output array
	 */

	// assign tid by using block id, block dimension, and thread id
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	// stride is for big arrays, i.e. bigger than threads we have
	int stride = blockDim.x * gridDim.x;

	// do the operations
	while (tid < N) {
		// two reads, one write
		c[tid] = a[tid] + b[tid];
		tid += stride;
	}
}


__global__ void init(float *a, float *b, int N) {
	/*
	 * function: init
	 * purpose: init memory from the GPU instead of needing to copy memory over from the cpu
	 * PARAMETERS:
	 *  a - first array
	 *  b - second array
	 *  c - output array
	 */

	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for (int i = index; i < N; i += stride) {
		a[i] = 1.0f;
		b[i] = 2.0f;
	}
}


kernel_parameters get_kernel_parameters(int device_num, int num_ops) {
	// first query the device for the parameters
	cudaDeviceProp prop;
	CudaSafeCall(cudaGetDeviceProperties(&prop, device_num));

	// init my custom struct that is defined in adding_gpu.cuh
	kernel_parameters return_parameters = {0};

	// using max number of threads in the x dim possible
	return_parameters.num_threads = 128;

	// calculate number of blocks based on the number of threads
	return_parameters.num_blocks = (num_ops + return_parameters.num_threads - 1) / return_parameters.num_threads;

	// check to make sure the number of blocks is in bounds...
	if (return_parameters.num_blocks > prop.maxGridSize[0]) {
		printf("Number of blocks calculated exceeds maximum grid size.\nUsing maxGridSize[0] instead.");
		return_parameters.num_blocks = prop.maxGridSize[0];
	}

	return return_parameters;
}


extern "C" void gpu_copy_driver(int device_num, int N) {
	cudaSetDevice(device_num);
	cudaFree(0);
	// allocate memory. Must use malloc() when dealing with arrays this large
	// note this is allocating on the host computer.
	float *a = (float *)malloc(N * sizeof(float));
	float *b = (float *)malloc(N * sizeof(float));
	float *c = (float *)malloc(N * sizeof(float));
	float *c_ref = (float *)malloc(N * sizeof(float));
	float *d_a, *d_b, *d_c;

	// Now there are SIX total pointers. All of them currently reside on the CPU.

	CudaSafeCall(cudaMalloc((void**)&d_a, sizeof(float) * N));
	CudaSafeCall(cudaMalloc((void**)&d_b, sizeof(float) * N));
	CudaSafeCall(cudaMalloc((void**)&d_c, sizeof(float) * N));
	CudaCheckError();

	for (int i = 0; i < N; i++) {
		// actual values don't matter, as long as they're floats. Float is importatnt because
		// I am interested in FLOPS!
		a[i] = 1.0f;
		b[i] = 2.0f;
		c_ref[i] = 1.0f;
	}

	CudaSafeCall(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
	CudaCheckError();

	kernel_parameters num_blocks_threads = get_kernel_parameters(device_num, N);
	CudaCheckError();

	// events for timing
	cudaEvent_t startEvent, stopEvent;
	CudaSafeCall(cudaEventCreate(&startEvent));
	CudaSafeCall(cudaEventCreate(&stopEvent));
	float ms;

	// As Mark Harris does, "warm up"
	cuda_copy <<< num_blocks_threads.num_blocks, num_blocks_threads.num_threads >>> (d_a, d_b, d_c, N);
	cudaDeviceSynchronize();
	CudaCheckError();

	// ==================================
	// ROUTINE: Naive Add
	printf("%20s", "CUDA Copy");
	// ==================================

	// START THE TIMER
	CudaSafeCall(cudaEventRecord(startEvent, 0));

	for (int rep = 0; rep < NUMBER_OF_REPS; rep++) {
		cuda_copy <<< num_blocks_threads.num_blocks, num_blocks_threads.num_threads >>> (d_a, d_b, d_c, N);
	}
	CudaSafeCall(cudaEventRecord(stopEvent, 0));
	CudaSafeCall(cudaEventSynchronize(stopEvent));
	CudaSafeCall(cudaEventElapsedTime(&ms, startEvent, stopEvent));

	// now since c on the host is still filled with random bits, I want to copy d_c from the device
	// (where the result of the addition was stored) to the host variable c. Once again I do this
	// using cudaMemcpy()

	CudaSafeCall(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

	// And now the data is on the host and ready to be checked for errors!
	postprocess(c_ref, a, 3, ms, N, NUMBER_OF_REPS);
	printf("%30s %2d, %5s %2d \n", "Threads/Block:", num_blocks_threads.num_threads,
		"Number of Blocks:", num_blocks_threads.num_blocks);

	// and finally be a good programmer and free the variables.
	free(a); free(b); free(c);

	CudaSafeCall(cudaFree(d_a));
	CudaSafeCall(cudaFree(d_b));
	CudaSafeCall(cudaFree(d_c));
}



extern "C" void gpu_naive_driver(int device_num, int N) {
	cudaSetDevice(device_num);
	cudaFree(0);
	// allocate memory. Must use malloc() when dealing with arrays this large
	// note this is allocating on the host computer.
	float *a = (float *)malloc(N * sizeof(float));
	float *b = (float *)malloc(N * sizeof(float));
	float *c = (float *)malloc(N * sizeof(float));
	float *c_ref = (float *)malloc(N * sizeof(float));
	float *d_a, *d_b, *d_c;

	// Now there are SIX total pointers. All of them currently reside on the CPU.

	// Next step is to allocate some memory on the GPU. This is done using cudaMalloc().
	// All the device variables are currently of the type float*, but cudaMalloc takes an
	// input of void** which is essentially a pointer to the pointer, aka the address of
	// the pointer variable

	//printf("Device Global Memory Needed: %d\n", 3 * N * sizeof(float));

	CudaSafeCall( cudaMalloc((void**) &d_a, sizeof(float) * N) );
	CudaSafeCall( cudaMalloc((void**) &d_b, sizeof(float) * N) );
	CudaSafeCall( cudaMalloc((void**) &d_c, sizeof(float) * N) );
	CudaCheckError();

	// Now there are three chunks of memory allocated for d_a, d_b, and d_c ON THE DEVICE.
	// What is currently in that memory is anyone's guess, it's garbage until we set it,
	// so let's do that.

	// First, I am going to make the arrays on the CPU in the variables a and b that I have
	// allocated above. Similar to the d_a, d_b, and d_c, the values in a, b, and c are
	// currently garbage. I want to set them to known values so I can check for errors
	// later.

	for (int i = 0; i < N; i++) {
		// actual values don't matter, as long as they're floats. Float is importatnt because
		// I am interested in FLOPS!
		a[i] = 1.0f;
		b[i] = 2.0f;
		c_ref[i] = 1.0f + 2.0f;
	}

	// Now a and b are filled with 1 and 2, respectively. Let's add them!

	// I need to first fill the pointers d_a and d_b with the values in a and b. This requires
	// me to copy memory from the host to the device. Not the most efficient thing to do here,
	// but that's what I'm going for this naive example. I will do this using cudaMemcpy()

	CudaSafeCall(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));
	CudaCheckError();

	// Now the data is on the GPU and the CPU!!! It is ready to be added! But first, we want to
	// optomize the occupancy of the GPU, i.e. I want to use as many resources as possible as
	// letting cores stand idly by is not efficient!

	kernel_parameters num_blocks_threads = get_kernel_parameters(device_num, N);
	//printf("%d, %d\n", num_blocks_threads.num_blocks, num_blocks_threads.num_threads);
	CudaCheckError();

	// events for timing
	cudaEvent_t startEvent, stopEvent;
	CudaSafeCall(cudaEventCreate(&startEvent));
	CudaSafeCall(cudaEventCreate(&stopEvent));
	float ms;


	// As Mark Harris does, "warm up"
	cuda_add <<< num_blocks_threads.num_blocks, num_blocks_threads.num_threads >>> (d_a, d_b, d_c, N);
	CudaCheckError();

	// ==================================
	// ROUTINE: Naive Add
	printf("%20s", "Naive Add");
	// ==================================

	// START THE TIMER
	CudaSafeCall(cudaEventRecord(startEvent, 0));

	for (int rep = 0; rep < NUMBER_OF_REPS; rep++) {
		cuda_add <<< num_blocks_threads.num_blocks, num_blocks_threads.num_threads >>> (d_a, d_b, d_c, N);
	}
	CudaSafeCall(cudaEventRecord(stopEvent, 0));
	CudaSafeCall(cudaEventSynchronize(stopEvent));
	CudaSafeCall(cudaEventElapsedTime(&ms, startEvent, stopEvent));

	// now since c on the host is still filled with random bits, I want to copy d_c from the device
	// (where the result of the addition was stored) to the host variable c. Once again I do this
	// using cudaMemcpy()

	CudaSafeCall(cudaMemcpy(c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

	// And now the data is on the host and ready to be checked for errors!
	postprocess(c_ref, c, 3, ms, N, NUMBER_OF_REPS);
	printf("%30s %2d, %5s %2d \n", "Threads/Block:", num_blocks_threads.num_threads, 
		"Number of Blocks:", num_blocks_threads.num_blocks);

	// and finally be a good programmer and free the variables.
	free(a); free(b); free(c);

	CudaSafeCall(cudaFree(d_a));
	CudaSafeCall(cudaFree(d_b));
	CudaSafeCall(cudaFree(d_c));
}
