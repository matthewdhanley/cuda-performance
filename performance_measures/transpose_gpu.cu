#include "transpose_gpu.cuh"

const int THREADS_X = 32;
const int THREADS_Y = 8;
const int NUM_REPS = 100;

__global__ void copy_matrix(float *a, float *b, int nx, int ny) {
	/*
	* FUNCTION: naive_transpose
	* Transposes a into b
	* param float *a - input matrix, should be square
	* param float *b - location to put output matrix
	*/

	// INDEX INTO THE TILE
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.x + threadIdx.y;

	// Calculate width of grid. 
	int width = gridDim.x * blockDim.x;

	for (int j = 0; j < THREADS_X; j += THREADS_Y) {
		if (tidy + j < ny) {
			b[(tidy + j)*width + tidx] = a[(tidy + j)*width + tidx];
		}
	}
}

__global__ void naive_transpose(float *a, float *b, int nx, int ny) {
	/*
	* FUNCTION: naive_transpose
	* Transposes a into b
	* param float *a - input matrix, should be square
	* param float *b - location to put output matrix
	*/

	// INDEX INTO THE TILE
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.x + threadIdx.y;

	// Calculate width of grid. 
	int width = gridDim.x * blockDim.x;

	for (int j = 0; j < THREADS_X; j += THREADS_Y) {
		if (tidy + j < ny) {
			b[tidx*width + (tidy + j)] = a[(tidy + j)*width + tidx];
		}
	}
}

__global__ void transpose_shared(float *a, float *b, int nx, int ny) {
	/*
	* FUNCTION: naive_transpose
	* Transposes a into b
	* param float *a - input matrix, should be square
	* param float *b - location to put output matrix
	*/

	__shared__ float tile[THREADS_X][THREADS_X];

	// INDEX INTO THE TILE
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.x + threadIdx.y;

	// Calculate width of grid. 
	int width = gridDim.x * blockDim.x;

	for (int j = 0; j < THREADS_X; j += THREADS_Y) {
		if (tidy + j < ny) {
			tile[threadIdx.y+j][threadIdx.x] = a[(tidy + j)*width + tidx];
		}
	}

	__syncthreads();

	tidx = blockIdx.y * blockDim.x + threadIdx.x;
	tidy = blockIdx.x * blockDim.x + threadIdx.y;

	for (int j = 0; j < THREADS_X; j += THREADS_Y) {
		if (tidy + j < ny) {
			b[(tidy + j)*width + tidx] = tile[threadIdx.x][threadIdx.y + j];
		}
	}
}

__global__ void transpose_no_conflict(float *a, float *b, int nx, int ny) {
	/*
	* FUNCTION: naive_transpose
	* Transposes a into b
	* param float *a - input matrix, should be square
	* param float *b - location to put output matrix
	*/

	__shared__ float tile[THREADS_X][THREADS_X+1];

	// INDEX INTO THE TILE
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.x + threadIdx.y;

	// Calculate width of grid. 
	int width = gridDim.x * blockDim.x;

	for (int j = 0; j < THREADS_X; j += THREADS_Y) {
		if (tidy + j < ny) {
			tile[threadIdx.y + j][threadIdx.x] = a[(tidy + j)*width + tidx];
		}
	}

	__syncthreads();

	tidx = blockIdx.y * blockDim.x + threadIdx.x;
	tidy = blockIdx.x * blockDim.x + threadIdx.y;

	for (int j = 0; j < THREADS_X; j += THREADS_Y) {
		if (tidy + j < ny) {
			b[(tidy + j)*width + tidx] = tile[threadIdx.x][threadIdx.y + j];
		}
	}
}

extern "C" void matrix_copy_driver(int deviceId, int nx, int ny) {
	dim3 numThreads(THREADS_X, THREADS_Y);
	dim3 numBlocks(nx / THREADS_X, ny / THREADS_X);

	float *a, *b, *b_ref, *d_a, *d_b;
	a = (float*) malloc(nx * ny * sizeof(float));
	b = (float*) malloc(nx * ny * sizeof(float));
	b_ref = (float*) malloc(nx * ny * sizeof(float));

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			a[i + j * nx] = i + j * nx;
			b_ref[i + j * nx] = i + j * nx;
		}
	}

	CudaSafeCall(cudaMalloc((float**)&d_a, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMalloc((float**)&d_b, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMemcpy(d_a, a, nx * ny * sizeof(float), cudaMemcpyHostToDevice));



	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ==================================
	// ROUTINE:Matrix Copy
	printf("%20s", "Naive Matrix Copy");
	float millis = 0;

	// warm up
	copy_matrix << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	cudaDeviceSynchronize();

	// start the timer
	cudaEventRecord(start, 0);
	for (int rep = 0; rep < NUM_REPS; rep++) {
		copy_matrix <<<numBlocks, numThreads >>> (d_a, d_b, nx, ny);
		//cudaDeviceSynchronize();
	}
	// stop the timer
	cudaEventRecord(stop, 0);

	// sync the events
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	// copy memory back to the host
	cudaMemcpy(b, d_b, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

	postprocess(b_ref, b, 2, millis, nx*ny, NUM_REPS);
	printf("%30s: %d x %d\n", "Matrix Size", nx, ny);
	printf("%30s (%d,%d), %10s (%d,%d) \n", "Threads/Block:", numThreads.x, numThreads.y,
		"Number of Blocks:", numBlocks.x, numBlocks.y);
}


extern "C" void naive_transpose_driver(int deviceId, int nx, int ny) {
	dim3 numThreads(THREADS_X, THREADS_Y);
	dim3 numBlocks(nx / THREADS_X, ny / THREADS_X);

	float *a, *b, *b_ref, *d_a, *d_b;
	a = (float*)malloc(nx * ny * sizeof(float));
	b = (float*)malloc(nx * ny * sizeof(float));
	b_ref = (float*)malloc(nx * ny * sizeof(float));

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			a[i + j * nx] = i + j * nx;
		}
	}

	// correct result
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			b_ref[j*nx + i] = a[i*nx + j];

	CudaSafeCall(cudaMalloc( (float**) &d_a, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMalloc( (float**) &d_b, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMemcpy(d_a, a, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ==================================
	// ROUTINE: Naive Transpose
	printf("%20s", "Naive Transpose");
	float millis = 0;

	// warm up
	naive_transpose << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	cudaDeviceSynchronize();

	// start the timer
	cudaEventRecord(start);
	for (int rep = 0; rep < NUM_REPS; rep++) {
		naive_transpose << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	}

	// stop the timer
	cudaEventRecord(stop);

	// sync the events
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	// copy memory back to the host
	cudaMemcpy(b, d_b, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

	postprocess(b_ref, b, 2, millis, nx*ny, NUM_REPS);

	printf("%30s: %d x %d\n", "Matrix Size", nx, ny);

	printf("%30s (%d,%d), %10s (%d,%d) \n", "Threads/Block:", numThreads.x, numThreads.y,
		"Number of Blocks:", numBlocks.x, numBlocks.y);
}

extern "C" void shared_transpose_driver(int deviceId, int nx, int ny) {
	dim3 numThreads(THREADS_X, THREADS_Y);
	dim3 numBlocks(nx / THREADS_X, ny / THREADS_X);

	float *a, *b, *b_ref, *d_a, *d_b;
	a = (float*)malloc(nx * ny * sizeof(float));
	b = (float*)malloc(nx * ny * sizeof(float));
	b_ref = (float*)malloc(nx * ny * sizeof(float));

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			a[i + j * nx] = i + j * nx;
		}
	}

	// correct result
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			b_ref[j*nx + i] = a[i*nx + j];

	CudaSafeCall(cudaMalloc((float**)&d_a, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMalloc((float**)&d_b, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMemcpy(d_a, a, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ==================================
	// ROUTINE: Naive Transpose
	printf("%20s", "Transpose w/ Shared");
	float millis = 0;

	// warm up
	transpose_shared << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	cudaDeviceSynchronize();

	// start the timer
	cudaEventRecord(start);
	for (int rep = 0; rep < NUM_REPS; rep++) {
		transpose_shared << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	}

	// stop the timer
	cudaEventRecord(stop);

	// sync the events
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	// copy memory back to the host
	cudaMemcpy(b, d_b, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

	postprocess(b_ref, b, 2, millis, nx*ny, NUM_REPS);

	printf("%30s: %d x %d\n", "Matrix Size", nx, ny);

	printf("%30s (%d,%d), %10s (%d,%d) \n", "Threads/Block:", numThreads.x, numThreads.y,
		"Number of Blocks:", numBlocks.x, numBlocks.y);
}

extern "C" void no_bank_conflict_transpose_driver(int deviceId, int nx, int ny) {
	dim3 numThreads(THREADS_X, THREADS_Y);
	dim3 numBlocks(nx / THREADS_X, ny / THREADS_X);

	float *a, *b, *b_ref, *d_a, *d_b;
	a = (float*) malloc(nx * ny * sizeof(float));
	b = (float*) malloc(nx * ny * sizeof(float));
	b_ref = (float*) malloc(nx * ny * sizeof(float));

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			a[i + j * nx] = i + j * nx;
		}
	}

	// correct result
	for (int j = 0; j < ny; j++)
		for (int i = 0; i < nx; i++)
			b_ref[j*nx + i] = a[i*nx + j];

	CudaSafeCall(cudaMalloc((float**)&d_a, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMalloc((float**)&d_b, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMemcpy(d_a, a, nx * ny * sizeof(float), cudaMemcpyHostToDevice));

	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ==================================
	// ROUTINE: Naive Transpose
	printf("%20s", "Transpose no bank conf.");
	float millis = 0;

	// warm up
	transpose_no_conflict << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	cudaDeviceSynchronize();

	// start the timer
	cudaEventRecord(start);
	for (int rep = 0; rep < NUM_REPS; rep++) {
		transpose_no_conflict << <numBlocks, numThreads >> > (d_a, d_b, nx, ny);
	}

	// stop the timer
	cudaEventRecord(stop);

	// sync the events
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	// copy memory back to the host
	cudaMemcpy(b, d_b, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);

	postprocess(b_ref, b, 2, millis, nx*ny, NUM_REPS);

	printf("%30s: %d x %d\n", "Matrix Size", nx, ny);

	printf("%30s (%d,%d), %10s (%d,%d) \n", "Threads/Block:", numThreads.x, numThreads.y,
		"Number of Blocks:", numBlocks.x, numBlocks.y);
}