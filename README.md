# CUDA Kernel Engineering: GEMM → Softmax → FlashAttention

A ground-up implementation of core ML inference primitives in CUDA. Each primitive progresses from a CPU baseline through naive CUDA to shared memory and warp-level optimized kernels, ending with a fused FlashAttention forward pass.

Demonstrates practical GPU kernel engineering:
- memory access patterns and global memory bottlenecks
- shared memory tiling to reduce memory traffic
- warp-level reductions via shuffle intrinsics
- online softmax for fused, memory-efficient attention
- correctness verification and timing via CUDA events

---

## Benchmarks — NVIDIA T4 (FP32, M = K = N = 512)

### GEMM Performance
| Variant | Time (s) | Speedup vs CPU |
|--------|----------:|----------------:|
| CPU baseline | 0.193987 | 1× |
| Naive CUDA | 0.00130179 | 149.0× |
| Tiled CUDA (16×16 shared memory) | 0.000983168 | 197.3× |

### Softmax Performance
| Variant | Time (s) | Speedup vs CPU | Speedup vs naive |
|--------|----------:|----------------:|------------------:|
| CPU baseline | 0.00299208 | 1× | — |
| Naive CUDA (1 thread/row) | 0.00112278 | 2.66× | — |
| Tiled CUDA (shared memory, block-wide reduction) | 0.000224576 | 13.32× | 4.99× |
| Warp-shuffle optimized softmax (`__shfl_down_sync`) | 0.000071904 | 41.61× | 15.62× |

**Correctness:**  
All kernels produce a maximum absolute error ≤ **4.57e-05** relative to CPU reference.

**Timing method:**  
CUDA events (`cudaEventRecord`), not wall-clock time.

---

## Benchmarks — NVIDIA A100 (FP32)

### GEMM Performance (M = K = N = 512)
| Variant | Time (s) | Speedup vs CPU |
|--------|----------:|----------------:|
| CPU baseline | 0.202 | 1× |
| Naive CUDA | 0.000365 | 553× |
| Tiled CUDA (16×16 shared memory) | 0.000129 | 1,535× |

### Softmax Performance (M = N = 512)
| Variant | Time (s) | Speedup vs CPU | Speedup vs naive |
|--------|----------:|----------------:|------------------:|
| CPU baseline | 0.00329 | 1× | — |
| Naive CUDA (1 thread/row) | 0.000567 | 5.8× | — |
| Tiled CUDA (shared memory, block-wide reduction) | 0.0000686 | 48× | 8.3× |
| Warp-shuffle optimized softmax (`__shfl_down_sync`) | 0.0000451 | 73× | 12.6× |

### FlashAttention Forward Pass (N = 4096, d = 32)
| Variant | Time (s) | Speedup vs CPU |
|--------|----------:|----------------:|
| CPU naive attention | 1.337 | 1× |
| Naive GPU attention (3-kernel) | 0.00149 | 898× |
| FlashAttention (this work) | 0.00644 | 208× |

**Correctness:**  
Max absolute error vs CPU reference: **3.0e-06**

**Timing method:**  
CUDA events (`cudaEventRecord`), not wall-clock time.

---

## Implementation Overview

### CPU Reference
Used only for correctness and baseline comparison.
- triple-nested GEMM
- row-wise softmax with max-subtraction for numerical stability

### Naive CUDA GEMM
Each thread computes one output element `C[i,j]`.  
Global memory reused repeatedly → memory-bound performance.

### Tiled CUDA GEMM
Each thread block:
- loads tiles of A and B into `__shared__` memory
- synchronizes (`__syncthreads()`)
- reuses data across 16×16 threads

Reduces global memory traffic significantly.

### Softmax Kernels
- Naive softmax: 1 thread per row
- Tiled softmax: 1 block per row + shared memory scratch buffer
- Warp softmax: no shared memory or atomics; uses warp shuffles for reductions

### FlashAttention Forward Pass
Implements the forward pass of [FlashAttention (Dao et al. 2022)](https://arxiv.org/abs/2205.14135) as a single fused CUDA kernel.

Standard attention computes `O = softmax(QKᵀ / √d) V` by materializing the full N×N score matrix in global memory, 64MB at N=4096. This kernel never writes that matrix. Instead it sweeps over K and V in tiles of 32 rows, maintaining a running `(max, sum, accumulator)` triple per query row. Each time a new tile raises the running max, previous accumulations are rescaled by `exp(m_old - m_new)` before adding the new tile's contribution. When the sweep completes the accumulator holds the exact softmax-weighted sum without any intermediate N×N allocation.

Key implementation details:
- 4 query rows per thread block (4 warps), sharing one KV tile load per tile - reduces global memory reads for K and V by 4×
- Warp shuffle (`__shfl_xor_sync`) for per-tile max reduction instead of shared memory atomics
- `__ldg()` cache hints on K and V loads (read-only texture cache)
- Score scaling (`1/√d`) hoisted outside the tile loop

The remaining gap vs naive GPU attention reflects the difference between scalar FMA and the tensor core `wmma` instructions used in production FA implementations, and sequential KV sweep vs split-K parallelism across thread blocks.

---

## Build and Run

### Requirements
- NVIDIA GPU with CUDA support
- CUDA toolkit (`nvcc`)
- C++ compiler

### Build from source
```bash
nvcc -arch=sm_75 -O3 -rdc=true \
  src/main.cu src/gemm_naive.cu src/gemm_tiled.cu \
  src/softmax_naive.cu src/softmax_tiled.cu src/softmax_warp.cu \
  -o run
```

### Build with FlashAttention (requires sm_80+)
```bash
nvcc -arch=sm_80 -O3 -rdc=true \
  src/main.cu src/gemm_naive.cu src/gemm_tiled.cu \
  src/softmax_naive.cu src/softmax_tiled.cu src/softmax_warp.cu \
  src/naive_attention.cu src/flash_attention_fwd.cu \
  -o run
```
