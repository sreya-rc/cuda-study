#include <cuda_runtime.h>
#include <math.h>
#include "../include/attention_kernels.h"

#define Bc           32
#define D            32
#define ROWS_PER_BLK  8

__global__ void flash_attention_fwd(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d)
{
    int i   = blockIdx.x * ROWS_PER_BLK + threadIdx.y;
    int tid = threadIdx.x;
    if (i >= N) return;

    __shared__ float Qi[ROWS_PER_BLK][D];
    __shared__ float Kj[Bc][D];
    __shared__ float Vj[Bc][D];
    __shared__ float scores[ROWS_PER_BLK][Bc];

    if (tid < d)
        Qi[threadIdx.y][tid] = Q[i * d + tid];
    __syncthreads();

    float acc = 0.0f;
    float m   = -1e20f;
    float l   =  0.0f;

    int num_tiles = (N + Bc - 1) / Bc;

    for (int t = 0; t < num_tiles; t++) {
        int kv_row = t * Bc + tid;

        // only first warp loads KV tile, all warps in block reuse it
        if (threadIdx.y == 0) {
            for (int dd = 0; dd < d; dd++) {
                Kj[tid][dd] = (kv_row < N) ? __ldg(&K[kv_row * d + dd]) : 0.0f;
                Vj[tid][dd] = (kv_row < N) ? __ldg(&V[kv_row * d + dd]) : 0.0f;
            }
        }
        __syncthreads();

        // each warp computes scores for its own query row
        float scale = 1.0f / sqrtf((float)d);   // constant for all tiles
        float dot = 0.0f;
        for (int dd = 0; dd < d; dd++)
            dot += Qi[threadIdx.y][dd] * Kj[tid][dd];
        scores[threadIdx.y][tid] = (kv_row < N) ? dot * scale : -1e20f;
        __syncthreads();

        // warp shuffle max within each warp 
        float tile_max = scores[threadIdx.y][tid];
        for (int offset = 16; offset > 0; offset >>= 1)
            tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffff, tile_max, offset));

        // online softmax rescale
        float m_new   = fmaxf(m, tile_max);
        float alpha   = expf(m - m_new);
        float l_new   = alpha * l;
        float acc_new = alpha * acc;

        // fused exp loop: e computed, used, discarded (no register array)
        for (int k = 0; k < Bc && (t * Bc + k) < N; k++) {
            float e  = expf(scores[threadIdx.y][k] - m_new);
            l_new   += e;
            acc_new += e * Vj[k][tid];
        }

        m   = m_new;
        l   = l_new;
        acc = acc_new;

        __syncthreads();
    }

    if (tid < d)
        O[i * d + tid] = acc / l;
}
