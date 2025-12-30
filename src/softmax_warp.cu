#include <cuda_runtime.h>

__global__ void softmax_warp(float* C, int M, int N) {
    int row = blockIdx.x;
    if (row >= M) return;

    float* row_ptr = C + row * N;
    float max_val = -1e20f;

    // find max (each thread processes multiple elements)
    for (int j = threadIdx.x; j < N; j += warpSize)
        max_val = fmaxf(max_val, row_ptr[j]);

    // warp reduce max
    for (int offset = 16; offset > 0; offset >>= 1)
        max_val = fmaxf(max_val, __shfl_down_sync(0xffffffff, max_val, offset));

    float sum = 0.0f;

    // compute exp & sum
    for (int j = threadIdx.x; j < N; j += warpSize) {
        row_ptr[j] = expf(row_ptr[j] - max_val);
        sum += row_ptr[j];
    }

    // warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    // write normalized
    for (int j = threadIdx.x; j < N; j += warpSize)
        row_ptr[j] /= sum;
}
