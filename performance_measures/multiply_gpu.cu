#include "multiply_gpu.cuh"

int TILE_SIZE = 32;
const int NUM_REPS = 100;

__global__ void naive_mult(float *a, float *b, float *c, int nx) {
	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
	int stride = blockDim.x * gridDim.x;

	float sum;

	for (int tidxx = tidx; tidxx < nx; tidxx += stride) {
		for (int tidyy = tidy; tidyy < nx; tidyy += stride){
			sum = 0.0f;
			for (int j = 0; j < nx; j++) {
				sum += a[tidyy * nx + j] * b[j * nx + tidxx];
			}
			c[tidyy * nx + tidxx] = sum;
		}
	}
}


//__global__ void tiled_mult(float *a, float *b, float *c, int nx) {
//	int tidx = blockIdx.x * blockDim.x + threadIdx.x;
//	int tidy = blockIdx.y * blockDim.y + threadIdx.y;
//	int stride = blockDim.x * gridDim.x;
//
//	float sum;
//
//	for (int tidxx = tidx; tidxx < nx; tidxx += stride) {
//		for (int tidyy = tidy; tidyy < nx; tidyy += stride) {
//			sum = 0.0f;
//			for (int j = 0; j < nx; j++) {
//				sum += a[tidyy * nx + j] * b[j * nx + tidxx];
//			}
//			c[tidyy * nx + tidxx] = sum;
//		}
//	}
//}


extern "C" void naive_mult_driver(int deviceId, int nx, int ny) {
	dim3 numThreads(TILE_SIZE, TILE_SIZE);
	dim3 numBlocks(nx / TILE_SIZE, ny / TILE_SIZE);

	float *a, *b, *c, *c_ref, *d_a, *d_b, *d_c;
	a = (float*)malloc(nx * ny * sizeof(float));
	b = (float*)malloc(nx * ny * sizeof(float));
	c = (float*)malloc(nx * ny * sizeof(float));
	c_ref = (float*)malloc(nx * ny * sizeof(float));

	for (int i = 0; i < nx; i++) {
		for (int j = 0; j < ny; j++) {
			a[i * nx + j] = ((float)i+1.0f)/((float)j+1.0f);
			b[i * nx + j] = ((float)j + 1.0f) / ((float)i + 1.0f);
		}
	}

	float sum;
	int count = 0;
	for (int row = 0; row < nx; row++) {
		for (int col = 0; col < nx; col++) {
			sum = 0.0f;
			for (int k = 0; k < nx; k++) {
				sum += a[row * nx + k] * b[k * nx + col];
			}
			c_ref[row * nx + col] = sum;
		}
		
	}


	CudaSafeCall(cudaMalloc((float**)&d_a, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMalloc((float**)&d_b, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMalloc((float**)&d_c, nx * ny * sizeof(float)));
	CudaSafeCall(cudaMemcpy(d_a, a, nx * ny * sizeof(float), cudaMemcpyHostToDevice));
	CudaSafeCall(cudaMemcpy(d_b, b, nx * ny * sizeof(float), cudaMemcpyHostToDevice));


	// set up the timer
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// ==================================
	// ROUTINE:Matrix Copy
	printf("%20s", "Naive Mat Mult");
	float millis = 0;

	// warm up
	naive_mult << <numBlocks, numThreads >> > (d_a, d_b,d_c, nx);
	CudaCheckError();
	cudaDeviceSynchronize();

	// start the timer
	cudaEventRecord(start, 0);
	for (int rep = 0; rep < NUM_REPS; rep++) {
		naive_mult << <numBlocks, numThreads >> > (d_a, d_b, d_c, nx);
		//cudaDeviceSynchronize();
	}

	// stop the timer
	cudaEventRecord(stop, 0);

	// sync the events
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&millis, start, stop);

	// copy memory back to the host
	cudaMemcpy(c, d_c, nx * ny * sizeof(float), cudaMemcpyDeviceToHost);
	postprocess(c_ref, c, 2*nx+1, millis, nx*ny, NUM_REPS);
	printf("%30s: %d x %d\n", "Matrix Size", nx, ny);
	printf("%30s (%d,%d), %10s (%d,%d) \n", "Threads/Block:", numThreads.x, numThreads.y,
		"Number of Blocks:", numBlocks.x, numBlocks.y);
}