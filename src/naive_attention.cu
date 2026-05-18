#include <cuda_runtime.h>
#include <math.h>
#include "../include/attention_kernels.h"

void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, float* S, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);

    // S = Q x Kt [N×N]
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0.f;
            for (int k = 0; k < d; k++)
                dot += Q[i*d+k] * K[j*d+k];
            S[i*N+j] = dot * scale;
        }

    // softmax rows of S
    for (int i = 0; i < N; i++) {
        float mx = -1e9f;
        for (int j = 0; j < N; j++) mx = fmaxf(mx, S[i*N+j]);
        float sum = 0.f;
        for (int j = 0; j < N; j++) sum += expf(S[i*N+j] - mx);
        for (int j = 0; j < N; j++) S[i*N+j] = expf(S[i*N+j] - mx) / sum;
    }

    // O = S x V [N×d]
    for (int i = 0; i < N; i++)
        for (int j = 0; j < d; j++) {
            float acc = 0.f;
            for (int k = 0; k < N; k++)
                acc += S[i*N+k] * V[k*d+j];
            O[i*d+j] = acc;
        }
}
