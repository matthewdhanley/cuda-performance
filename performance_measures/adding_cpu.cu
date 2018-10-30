#include "adding_cpu.cuh"

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
