# viva_tensor vs PyTorch vs NumPy — Fair Matmul Benchmark

**Hardware**

- CPU: 24-core Ryzen, AVX2, Intel MKL
- GPU: NVIDIA RTX 4090 (Ada Lovelace, sm_89, 24GB)
- Driver: 595.71.05, CUDA 12.9

**Software**

- viva_tensor: this repo (Gleam → Zig NIF → MKL/cuBLAS/CUTLASS)
- PyTorch: 2.11.0+cu129
- NumPy: 2.4.4

**Methodology**

- 30 iterations, warmup excluded
- CPU paths: `time.perf_counter()` (wall clock)
- GPU paths: `cudaEvent_t` (kernel-only timing, sync after)
- Matrix shape: square N×N @ N×N → 2·N³ FLOPs

---

## 4096 × 4096 results (head-to-head)

### FP32 dense — CPU

| Backend                      |    TFLOPS | ms/iter | Notes                    |
|------------------------------|----------:|--------:|--------------------------|
| NumPy CPU FP32               |       0.8 |   162.0 | OpenBLAS auto            |
| PyTorch CPU FP32             |       1.2 |   116.2 | oneDNN + MKL, 24 threads |
| **viva_tensor MKL CPU FP32** | **~0.57** |    ~280 | MKL `dgemm` via NIF      |

> CPU FP32: PyTorch wins (~2× viva_tensor). PyTorch's oneDNN does aggressive blocking + thread pool that the raw
`cblas_dgemm` call doesn't match. Future viva_tensor work: blocked GEMM + thread pool.

### FP32 dense — GPU

| Backend                             |          TFLOPS | ms/iter |
|-------------------------------------|----------------:|--------:|
| PyTorch GPU FP32                    |            14.3 |    9.63 |
| viva_tensor RTX FP32 (cuBLAS SGEMM) | ~10 (rtx bench) |     ~10 |

> Both hit the 4090's FP32 nerf (~25 TFLOPS theoretical, ~14 achievable on cuBLAS).

### FP16 / BF16 — GPU

| Backend                                     |    TFLOPS | ms/iter | Path                                       |
|---------------------------------------------|----------:|--------:|--------------------------------------------|
| PyTorch GPU FP16                            |     102.6 |    1.34 | cuBLAS HGEMM Tensor Core                   |
| PyTorch GPU BF16                            | **149.3** |    0.92 | cuBLAS Tensor Core                         |
| **viva_tensor cublasLt FP16 + COMPUTE_16F** | **102.2** |    1.34 | cublasLt FP16 accum                        |
| viva_tensor legacy `nt_matmul_fp16_tc`      |       ~16 |  varies | cuBLAS HGEMM + FP32 accum (half-rate nerf) |

> **FP16 dense: tied with PyTorch** (102.2 vs 102.6 TFLOPS) after switching the cublasLt path to `CUBLAS_COMPUTE_16F` +
> FP16 alpha/beta. This unlocks the full-rate 165 TFLOPS Tensor Core MMA instead of the half-rate FP32-accum path. BF16
> path still uses FP32 accum on GeForce — adding a `cublaslt_bf16_bench` with COMPUTE_16BF when NVIDIA exposes one would
> close the BF16 gap too.

### FP8 E4M3 — GPU

| Backend                                  |    TFLOPS | ms/iter | Path                                     |
|------------------------------------------|----------:|--------:|------------------------------------------|
| PyTorch GPU FP8 (`_scaled_mm`)           |     307.1 |    0.45 | cuBLASLt FP8 (FP32 accum cap on GeForce) |
| **viva_tensor CUTLASS FP8 + FP16 accum** | **392.5** |    0.35 | CUTLASS bypass + FP16 accum              |

> **viva_tensor wins FP8 by 27%.** This is the "GeForce nerf bypass" trick:
> - PyTorch's `_scaled_mm` goes through cuBLASLt → capped at FP32 accum on GeForce → 330 TFLOPS half-rate.
> - viva_tensor uses a custom CUTLASS GEMM with `ElementAccumulator=half_t` → unlocks
    `mma.sync.aligned.m16n8k32.f16.e4m3.e4m3.f16` → 660 TFLOPS full-rate.
> - Hopper/H100/H200 doesn't have this nerf — PyTorch is competitive there.

### 2:4 Structured Sparse — GPU

PyTorch's `_scaled_mm` doesn't expose 2:4 sparse paths directly. PyTorch has `torch.sparse.SparseSemiStructuredTensor` (
since 2.1), but it doesn't combine with `_scaled_mm` for FP8 or INT4 yet. viva_tensor exposes the cuSPARSELt + CUTLASS
sparse kernels directly.

| Backend                          |     TFLOPS | ms/iter | Path                 |
|----------------------------------|-----------:|--------:|----------------------|
| viva_tensor cuSPARSELt FP8 2:4   |      469.0 |    0.29 | cuSPARSELt FP8 2:4   |
| viva_tensor cuSPARSELt FP16 2:4  |      306.8 |    0.45 | cuSPARSELt FP16 2:4  |
| viva_tensor cuSPARSELt INT8 2:4  |  **627.3** |    0.22 | cuSPARSELt INT8 2:4  |
| **viva_tensor CUTLASS INT4 2:4** | **1073.7** |    0.13 | CUTLASS INT4 sparse  |
| PyTorch equivalent               |        n/a |     n/a | not natively exposed |

> **1.07 PFLOPS effective on INT4 2:4** — this is viva_tensor's clear win-zone. The sparse kernels require
> pre-pruned/packed weights, but every inference framework with quantization can use them.

---

## 2048 × 2048 summary

| Backend                       | TFLOPS |
|-------------------------------|-------:|
| NumPy CPU FP32                |    0.8 |
| PyTorch CPU FP32              |    1.2 |
| PyTorch GPU FP32              |   22.6 |
| PyTorch GPU FP16              |   61.8 |
| PyTorch GPU BF16              |   62.7 |
| PyTorch GPU FP8               |  114.7 |
| viva CUTLASS FP8 (FP16 accum) |  218.0 |
| viva cuSPARSELt FP8 2:4       |  323.3 |
| viva cuSPARSELt INT8 2:4      |  293.3 |
| viva CUTLASS INT4 2:4         |  581.1 |

> At 2048² the GEMM is small enough that launch overhead matters; PyTorch's stream scheduling helps it on dense paths.
> viva's sparse kernels still win the absolute throughput.

---

## Take-aways

1. **CPU dense FP32**: PyTorch wins (~2×). Optimization gap, not a structural one.
2. **GPU dense FP16/BF16**: PyTorch wins. viva needs cuBLASLt + FP16 accum + larger tuned shapes.
3. **GPU FP8 dense**: **viva wins** (392 vs 307 TFLOPS, +27%) thanks to the CUTLASS FP16-accum bypass.
4. **GPU 2:4 sparse**: **viva wins by default** — PyTorch doesn't expose these paths through high-level APIs. 1.07
   PFLOPS effective on INT4 2:4 sparse.

For inference workloads with quantized + sparse weights (the common case for LLM deployment), viva_tensor delivers
competitive-to-superior peak throughput on RTX 4090.

---

## Reproduce

```bash
# Build NIF with CUDA (one-time)
make zig-cuda

# viva_tensor side
gleam run -m viva_tensor/bench/peak

# PyTorch + NumPy side
uv venv .bench-venv
VIRTUAL_ENV=.bench-venv uv pip install torch numpy --index-url https://download.pytorch.org/whl/cu129
.bench-venv/bin/python3 bench/compare/numpy_pytorch.py
```
