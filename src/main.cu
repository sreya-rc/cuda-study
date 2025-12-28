#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>

#include "../include/gemm_kernels.h"
#include "../include/softmax_kernels.h"

// ---------------- CPU Benchmarks ----------------
void cpu_gemm(const float* A, const float* B, float* C,
              int M, int K, int N) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++)
                acc += A[i*K + k] * B[k*N + j];
            C[i*N + j] = acc;
        }
}

void cpu_softmax(float* C, int M, int N) {
    for (int i = 0; i < M; i++) {
        float maxv = -1e9f;
        for (int j = 0; j < N; j++)
            maxv = std::max(maxv, C[i*N + j]);

        float sum = 0.f;
        for (int j = 0; j < N; j++)
            sum += expf(C[i*N + j] - maxv);

        for (int j = 0; j < N; j++)
            C[i*N + j] = expf(C[i*N + j] - maxv) / sum;
    }
}


// ---------------- Helpers ----------------
double max_abs_error(const float* a, const float* b, int size) {
    double max_err = 0.0;
    for (int i = 0; i < size; i++) {
        double diff = fabs(a[i] - b[i]);
        if (diff > max_err) max_err = diff;
    }
    return max_err;
}

int main() {
    int M = 512, K = 512, N = 512;
    int size = M * N;

    std::vector<float> A(M*K), B(K*N), C_cpu(size), C_gpu(size);
    for (auto& x : A) x = static_cast<float>(rand()) / RAND_MAX;
    for (auto& x : B) x = static_cast<float>(rand()) / RAND_MAX;

    // ---------------- CPU Baseline ----------------
    std::cout << "===== CPU Baseline =====\n";
    auto start_cpu = std::chrono::high_resolution_clock::now();
    cpu_gemm(A.data(), B.data(), C_cpu.data(), M, K, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_cpu = end_cpu - start_cpu;
    std::cout << "CPU GEMM time: " << t_cpu.count() << " s\n\n";

    // ---------------- GPU Setup ----------------
    float *A_d, *B_d, *C_d;
    cudaMalloc(&A_d, M*K*sizeof(float));
    cudaMalloc(&B_d, K*N*sizeof(float));
    cudaMalloc(&C_d, size*sizeof(float));

    cudaMemcpy(A_d, A.data(), M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B.data(), K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/block.x, (M + block.y - 1)/block.y);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ---------------- Naive GEMM ----------------
    std::cout << "===== GEMM (GPU) =====\n";
    cudaEventRecord(start);
    gemm_naive<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);
    cudaMemcpy(C_gpu.data(), C_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Naive GEMM:  " << ms_naive/1000.0 << " s  "
              << "(" << (t_cpu.count()/(ms_naive/1000.0)) << "x faster)\n";

    // ---------------- Tiled GEMM ----------------
    cudaEventRecord(start);
    gemm_tiled<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);
    cudaMemcpy(C_gpu.data(), C_d, size*sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Tiled GEMM:  " << ms_tiled/1000.0 << " s  "
              << "(" << (t_cpu.count()/(ms_tiled/1000.0)) << "x faster, "
              << (ms_naive/ms_tiled) << "x vs naive)\n";

    // correctness test GEMM
    std::cout << "GEMM Max abs error: "
              << max_abs_error(C_cpu.data(), C_gpu.data(), size) << "\n\n";

    // ----- CPU softmax baseline -----
    auto cpu_soft_start = std::chrono::high_resolution_clock::now();
    cpu_softmax(C_cpu.data(), M, N);
    auto cpu_soft_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_soft_t = cpu_soft_end - cpu_soft_start;
    std::cout << "CPU softmax time: " << cpu_soft_t.count() << " s\n\n";

    std::cout << "===== Softmax (GPU) =====\n";

    // naive GPU
    cudaEventRecord(start);
    softmax_naive<<<(M+255)/256, 256>>>(C_d, M, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_soft_naive;
    cudaEventElapsedTime(&ms_soft_naive, start, stop);
    std::cout << "Naive softmax: " << ms_soft_naive/1000.0 << " s  ("
              << cpu_soft_t.count() / (ms_soft_naive/1000.0)
              << "× faster vs CPU)\n";

    // tiled GPU
    cudaEventRecord(start);
    softmax_tiled<<<M, 256, N * sizeof(float)>>>(C_d, M, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_soft_tiled;
    cudaEventElapsedTime(&ms_soft_tiled, start, stop);
    std::cout << "Tiled softmax: " << ms_soft_tiled/1000.0 << " s  ("
              << cpu_soft_t.count() / (ms_soft_tiled/1000.0)
              << "× faster vs CPU, "
              << (ms_soft_naive / ms_soft_tiled)
              << "× vs naive)\n";

    cudaFree(A_d); cudaFree(B_d); cudaFree(C_d);
    return 0;
}

