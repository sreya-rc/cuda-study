#include <cuda_runtime.h>

__global__ void softmax_naive(float* C, int M, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    // 1. Compute max
    float max_val = -1e9f;
    for (int j = 0; j < N; j++)
        max_val = fmaxf(max_val, C[row*N + j]);

    // 2. Compute exp + sum
    float sum = 0.0f;
    for (int j = 0; j < N; j++)
        sum += expf(C[row*N + j] - max_val);

    // 3. Normalize
    for (int j = 0; j < N; j++)
        C[row*N + j] = expf(C[row*N + j] - max_val) / sum;
}

