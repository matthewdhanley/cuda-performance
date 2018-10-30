#include "common.h"

void postprocess(const float *reference, const float *result, int reads_writes, float ms, int n, int num_loops)
{
	/*
	Function: postprocess
	Purpose: Assess success of kernel and report effective bandwidth
	param const float *reference - pointer to values to compate to
	param const float *result - pointer to values that need to be checked
	param int reads_writes - number of reads and writes per kernel execution
	param float ms - time that the kernels ran
	param int n - size of array
	param int num_loops - number of loops through kernel launch
	*/
	bool passed = true;
	float thresh = 0.5;
	for (int i = 0; i < n; i++)
		if (fabsf(result[i] - reference[i]) > thresh) {
			printf("\n%d %f %f\n", i, result[i], reference[i]);
			printf("%25s\n", "*** FAILED ***");
			passed = false;
			break;
		}
	if (passed) {
		// number of bytes read and wrote divided by elapsed time
		printf("%20.2f\n", reads_writes * sizeof(float) * 1e-6 * n * num_loops / ms);
	}
}