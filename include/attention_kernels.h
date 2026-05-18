#pragma once

__global__ void flash_attention_fwd(
    const float* Q, const float* K, const float* V,
    float* O, int N, int d);

void naive_attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, float* S, int N, int d);
