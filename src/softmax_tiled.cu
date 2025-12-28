#include <cuda_runtime.h>

#define BLOCK 256   // threads per row

__global__ void softmax_tiled(float* C, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    extern __shared__ float smem[];   // size >= N
    float* rowbuf = smem;

    // Load row into shared memory (each thread loads stride elements)
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        rowbuf[j] = C[row*N + j];
    }
    __syncthreads();

    // Compute max reduction
    float local_max = -1e9f;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        local_max = fmaxf(local_max, rowbuf[j]);
    }
    __shared__ float maxval;
    if (threadIdx.x == 0) maxval = -1e9f;
    __syncthreads();

    atomicMax((int*)&maxval, __float_as_int(local_max));
    __syncthreads();

    float m = __int_as_float((int)maxval);

    // Compute exp & sum reduction
    float local_sum = 0.f;
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        rowbuf[j] = expf(rowbuf[j] - m);
        local_sum += rowbuf[j];
    }
    __shared__ float sum;
    if (threadIdx.x == 0) sum = 0.f;
    __syncthreads();

    atomicAdd(&sum, local_sum);
    __syncthreads();

    // Normalize
    for (int j = threadIdx.x; j < N; j += BLOCK) {
        C[row*N + j] = rowbuf[j] / sum;
    }
}
