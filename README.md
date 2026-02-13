<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:006400&height=180&section=header&text=viva_tensor&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=High-Performance%20Tensors%20for%20Gleam&descSize=20&descAlignY=55" width="100%"/>

[![Gleam](https://img.shields.io/badge/Gleam-FFAFF3?style=for-the-badge&logo=gleam&logoColor=000)](https://gleam.run/)
[![Tests](https://img.shields.io/badge/Tests-187%20passed-2E8B57?style=for-the-badge)](./test)
[![License](https://img.shields.io/badge/MIT-2E8B57?style=for-the-badge)](./LICENSE)

**The fastest tensor library on the BEAM**

</div>

---

## Performance

### GPU Tensor Cores (RTX 4090)

| Backend | Throughput | % of Peak |
|:--------|----------:|----------:|
| FP8 E4M3 (CUTLASS) | **660 TOPS** | 100% |
| INT8 Dense (IMMA) | **604 TOPS** | 92% |
| FP16 Dense (cublasGemmEx) | **284 TFLOPS** | 86% |
| FP32/TF32 (cuBLAS) | **84.5 TFLOPS** | 102% |
| Fused GEMM+ReLU | **162 TFLOPS** | free activation |

### GPU 2:4 Structured Sparsity

| Backend | Throughput | % of Peak |
|:--------|----------:|----------:|
| INT4 Sparse (CUTLASS) | **1854 TOPS** | 70% |
| INT8 Sparse (cuSPARSELt) | **1094 TOPS** | 83% |
| INT8 Sparse (CUTLASS) | **841 TOPS** | 64% |
| FP8 Sparse (cuSPARSELt) | **702 TOPS** | 53% |
| FP16 Sparse (cuSPARSELt) | **355 TFLOPS** | 53% |

### CPU (Intel MKL)

| Size | viva_tensor | PyTorch | NumPy | vs PyTorch |
|:----:|:-----------:|:-------:|:-----:|:----------:|
| 5000x5000 | **931 GFLOPS** | 620 | 368 | **+50%** |

> Xeon 24-core (AVX2), MKL dgemm FP64, compact affinity, MADV_HUGEPAGE.
> All numbers verified with CUDA events and IQR outlier removal.

---

## Install

```bash
gleam add viva_tensor
```

## Architecture

```mermaid
graph TB
    subgraph "Gleam Layer (44 modules, 67K lines)"
        A[viva_tensor API]
        B[core/ - tensor, ops, shape, ffi]
        C[quant/ - INT8, NF4, AWQ]
        D[nn/ - autograd, layers, flash_attention]
    end

    subgraph "Erlang Layer"
        E[viva_tensor_zig.erl - NIF wrapper]
    end

    subgraph "Native Layer (13K+ lines C/CUDA)"
        F[nif_entry.c - dispatch]
        G[nif_cpu_ops.c - AVX2 SIMD]
        H[nif_cuda_fp32/fp16/int8.c - Tensor Cores]
        I[nif_sparse.c - 2:4 sparsity]
        J[nif_specialized.c - fused GEMM]
    end

    subgraph "Backend Libraries"
        K[Intel MKL]
        L[CUDA cuBLAS/cuBLASLt]
        M[cuSPARSELt]
        N[CUTLASS]
    end

    A --> B & C & D
    B --> E
    E --> F
    F --> G & H & I & J
    G --> K
    H --> L
    I --> M & N
    J --> L

    style A fill:#FFAFF3
    style K fill:#0071C5,color:#fff
    style L fill:#76B900,color:#fff
    style M fill:#76B900,color:#fff
    style N fill:#76B900,color:#fff
```

## Quick Start

```gleam
import viva_tensor as t

// Create tensors
let a = t.zeros([1000, 1000])
let b = t.random_uniform([1000, 1000])

// Matrix multiplication (auto-selects best backend)
let c = t.matmul(a, b)

// Activations
let activated = t.relu(c) |> t.sigmoid()
```

## Features

```mermaid
mindmap
  root((viva_tensor))
    Core Ops
      add/sub/mul/div
      matmul/transpose
      sum/mean/max/min
      dot/outer/broadcast
    GPU Backends
      FP32/TF32 cuBLAS
      FP16 Tensor Cores
      INT8 IMMA
      FP8 E4M3 CUTLASS
    Sparsity
      INT4 2:4 CUTLASS
      INT8 2:4 cuSPARSELt
      FP8/FP16 Sparse
    Quantization
      INT8 4x compress
      NF4 7.5x compress
      AWQ 7.7x compress
    Neural Networks
      autograd
      linear layers
      flash attention
      fused GEMM+act
    CNN
      conv2d
      max/avg pool2d
      global_avg_pool2d
```

### Quantization

| Method | Compression | Quality | Use Case |
|:------:|:-----------:|:-------:|:--------:|
| INT8 | 4x | 96% | Inference |
| NF4 | 7.5x | 99% | QLoRA Fine-tuning |
| AWQ | 7.7x | 97% | Edge Deployment |

## Build

```bash
# Pure Gleam (no native deps)
make build && make test

# With NIF acceleration (Intel MKL + CUDA)
make zig && make build

# Full build
make build-all
```

### Requirements

- Gleam 1.14.0+
- OTP 27+
- Zig 0.14+ (for NIF build)
- Intel MKL (CPU BLAS)
- CUDA 13+ with cuBLAS, cuBLASLt (GPU)
- cuSPARSELt 0.8.1+ (sparse ops)
- CUTLASS 4.3+ (FP8, INT4 sparse)

## GPU Benchmark Suite

```bash
# Individual benchmarks (Erlang escripts)
./bench/bench_gpu_peak.erl       # FP32/TF32
./bench/bench_fp16_imma.erl      # FP16 Tensor Cores
./bench/bench_int8_imma.erl      # INT8 IMMA
./bench/bench_fp8_peak.erl       # FP8 E4M3
./bench/bench_sparse_peak.erl    # 2:4 Sparsity
./bench/bench_fused_peak.erl     # Fused GEMM+activation
./bench/bench_batched_peak.erl   # Batched GEMM
```

---

<div align="center">

```mermaid
flowchart LR
    G[Gleam] --> Z[Zig NIF] --> M[Intel MKL]
    Z --> C[CUDA Tensor Cores]
    Z --> S[cuSPARSELt]
    Z --> CU[CUTLASS]
    style G fill:#FFAFF3,color:#000
    style Z fill:#F7A41D,color:#000
    style M fill:#0071C5,color:#fff
    style C fill:#76B900,color:#fff
    style S fill:#76B900,color:#fff
    style CU fill:#76B900,color:#fff
```

**Built with love for the BEAM**

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:006400,100:8B0000&height=80&section=footer" width="100%"/>

</div>
