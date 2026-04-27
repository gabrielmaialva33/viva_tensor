# Changelog

All notable changes to this project will be documented in this file.

## [1.0.100] - 2026-04-27

### Highlights

Major release covering modular NIF architecture, CUDA Tensor Core backends,
2:4 structured sparsity, and a full RTX 4090 acceleration stack with
persistent GPU workspaces and fused linear layers. Confirmed compatibility
with [VIVA](https://github.com/gabrielmaialva33/viva) holographic mycelium
networks (HRR binding/superposition, HoloNEAT neuroevolution, MAP-Elites
quality-diversity).

### Performance Scorecard

| Backend                          |      Throughput | % of Peak |      vs PyTorch |
|:---------------------------------|----------------:|----------:|----------------:|
| CPU FP64 (MKL dgemm)             |  **931 GFLOPS** |         - |        **+50%** |
| GPU FP32 (TF32 Tensor Cores)     | **84.5 TFLOPS** |      102% |        **+57%** |
| GPU FP16 Dense (cublasGemmEx)    |  **284 TFLOPS** |       86% |               - |
| GPU INT8 Dense (cublasLt IMMA)   |    **604 TOPS** |       92% |               - |
| GPU FP8 E4M3 (cuBLASLt)          |    **344 TOPS** |     104%* |               - |
| GPU FP8 E4M3 (CUTLASS half_t)    |    **660 TOPS** |      100% |               - |
| GPU INT8 2:4 Sparse (cuSPARSELt) |   **1094 TOPS** |       83% |               - |
| GPU INT4 2:4 Sparse (CUTLASS)    |   **1854 TOPS** |       70% |               - |
| GPU FP8 2:4 Sparse (cuSPARSELt)  |    **702 TOPS** |       53% |               - |
| GPU FP16 2:4 Sparse (cuSPARSELt) |  **355 TFLOPS** |       53% |               - |
| GPU INT8 Sparse (CUTLASS)        |    **841 TOPS** |       64% |               - |
| GPU Fused GEMM+ReLU/GELU         |  **162 TFLOPS** |         - | activation free |
| GPU FP16 Batched GEMM            |  **153 TFLOPS** |         - |               - |

> *FP8 cuBLASLt exceeds 330T GeForce FP8+FP32 spec due to internal TF32 promotion.
> Hardware: Xeon 24-core (AVX2) + RTX 4090. Verified with CUDA events, IQR outlier removal.

### Added

- **Modular NIF architecture**: Split monolithic 5500-line C file into 13
  focused modules (13K+ lines total)
  - `nif_entry.c` — dispatch table and resource management
  - `nif_tensor_core.c` — tensor create/read/write operations, zero-copy
    broadcast views, offset-aware storage
  - `nif_cpu_ops.c` — SIMD math (AVX2 dot, exp, sigmoid, relu), preallocated
    `*_into` element-wise kernels
  - `nif_cuda_fp32.c` — FP32/TF32 GPU GEMM with in-place variant
  - `nif_cuda_fp16.c` — FP16 Tensor Core GEMM, in-place, fused activations,
    fused linear layer (GEMM+Bias+activation)
  - `nif_cuda_int8.c` — INT8 IMMA Tensor Core GEMM
  - `nif_quant.c` — INT8/NF4/AWQ quantization
  - `nif_sparse.c` — 2:4 structured sparsity
  - `nif_sage_nif.c` — SageAttention
  - `nif_specialized.c` — fused GEMM+activation, batched GEMM
  - `nif_platform.c` — platform detection and backend selection
  - `nif_legacy.c` — backward-compatible API wrappers
  - `viva_nif.h` — shared header with common types and globals
- **CUDA Tensor Core backends**:
  - FP32/TF32 via cuBLAS with 32MiB workspace
  - FP16 via cublasGemmEx async (outperforms cublasLt for FP16)
  - INT8 IMMA via cublasLtMatmul TN with pre-transposed B upload
  - FP8 E4M3 via cuBLASLt (CUBLAS_COMPUTE_32F) and CUTLASS (half_t accumulator)
  - Fused GEMM+ReLU/GELU via cublasLt epilogues at zero cost
  - Fused GEMM+Bias+ReLU/GELU via cublasLt `RELU_BIAS` / `GELU_BIAS` epilogues
  - Batched GEMM for multi-head attention
  - Stream synchronization (`accelerated_sync`) for accurate timing
- **Structured sparsity (2:4)**:
  - cuSPARSELt 0.8.1 for INT8 (1094 TOPS), FP8 (702 TOPS), FP16 (355 TFLOPS)
  - CUTLASS GemmSparseUniversal for INT8 (841 TOPS) with Swizzle<8>
  - CUTLASS INT4 sparse (1854 TOPS) with 128-bit aligned loads
- **Native tensor storage**:
  - `NativeTensor` variant backed by NIF resources with offset/owner fields
  - Zero-copy broadcast views (`broadcast_to` returns a strided view)
  - Offset-aware kernels across add/sub/mul/scale/negate/dot/sum/max/min and
    activations
  - Preallocated `add_into` / `sub_into` / `mul_into` / `scale_into` /
    `matmul_into` / `linear_relu_into` for zero-allocation hot loops
- **RTX 4090 acceleration API**:
  - `matmul_auto` — RTX-first planner (FP16 → FP32 → MKL → CPU)
  - `AcceleratedTensor` (`CudaFp16`, `CudaFp32`, `Cpu`) for persistent
    placement plus `to_accelerated`, `to_rtx4090_fp16`, `to_rtx4090_fp32`
  - `matmul_accelerated` and `matmul_accelerated_into` for on-device GEMM
    without forced downloads
  - `matmul_relu_accelerated_into` / `matmul_gelu_accelerated_into` for FP16
    fused activations
  - `linear_relu_accelerated_into` / `linear_gelu_accelerated_into` for FP16
    fused linear layers
  - `GpuWorkspace` and `LinearLayer` for persisted forward passes with
    reusable output buffers (`workspace_zeros`, `linear_output`,
    `linear_relu_forward_into`, `linear_gelu_forward_into`)
- **CPU fused kernels**: `linear_relu` / `linear_relu_into` for the dense path
- **Helpers**: `map2` for paired element-wise iteration with shape checking
- **Open-source governance**:
  - `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `SECURITY.md`
  - GitHub issue forms (bug, feature, performance) and pull request template
- **Benchmarks**: RTX 4090 vs MKL benchmark module
  (`viva_tensor/bench/rtx`), zero-allocation `_into` benchmark suite
- **SageAttention**: CUDA-accelerated attention mechanism
- **CUTLASS integration**: SM89-optimized sparse and FP8 kernels compiled
  separately via nvcc
- **Build flags**: `-Dmkl_root` and `-Dcusparselt_path` to configure native
  dependency paths without editing the build script

### Changed

- Public API surface trimmed for stability while still under active
  development; low-level CUDA, tensor implementation, benchmark, and helper
  modules are now marked internal so they stay out of generated package
  documentation.
- HDC, Horde, LNS, and quantization FFI wrappers return `Result(_, String)`
  instead of panicking through `let assert Ok` so NIF failures propagate
  cleanly.
- `ct16_matmul` return type aligned with the FP32 result tensor produced by
  the underlying NIF (FP16 inputs accumulate into FP32 output).
- BLAS backend module documentation promoted to a `////` module doc so the
  auto-detection chain (MKL → OpenBLAS → Zig SIMD → fallback) is visible in
  HexDocs.
- Benchmark and example modules moved under the `viva_tensor` namespace for
  Hex.pm publishing compatibility.

### Architecture

```
Gleam API -> Erlang NIF -> Zig build system
                             |-> Intel MKL (CPU BLAS, 931 GFLOPS)
                             |-> CUDA cuBLAS/cuBLASLt (Tensor Cores)
                             |-> cuSPARSELt (2:4 structured sparsity)
                             |-> CUTLASS (FP8, INT4 sparse kernels)
                             |-> Zig SIMD (AVX2 vectorized ops)
```

### Key Optimizations

- CPU: in-place matmul, MADV_HUGEPAGE, MKL physical cores only, DAZ+FTZ,
  KMP_AFFINITY=compact
- GPU: zero-allocation in-place ops, cublasSetWorkspace_v2 (32MiB),
  per-shape cublasLt heuristic caches keyed on `(M, N, K)`
- INT8 TN path: pre-transpose B on CPU during upload for IMMA Tensor Cores
- FP8 CUTLASS: `half_t` accumulator selects FP16 MMA instruction
  (330T → 660T)
- Sparse: MatmulSearch with 20 iterations, SPARSE_MAT_POINTER hint
- CUTLASS sparse: GemmSparseUniversal + GemmIdentityThreadblockSwizzle<8>
  (+24% vs basic)
- Native tensors: zero-copy broadcast views and contiguous SIMD fast paths
  with strided fallbacks

### Verified

- 187 tests passing (core tensor, operations, autograd, shapes, CNN, NIF)
- Intel MKL NIF loads correctly (24 cores, compact affinity, DAZ+FTZ)
- VIVA project builds cleanly against viva_tensor
- All 14 GPU backends operational (FP32, FP16, INT8, FP8, sparse, fused,
  batched)

---

## [1.3.2] - 2026-01-26

### Fixed
- Removed all unused function arguments (zero warnings build)
- Aligned gleam.toml version with git tags

### Documentation
- Added comprehensive CHANGELOG.md
- Updated README with conv2d/pooling usage examples and diagrams

## [1.3.1] - 2026-01-26

### Performance
- **O(1) array access**: Replaced list traversal with Erlang `:array` for O(1) index access
- **Tail-recursive loops**: Eliminated stack growth in conv2d and pooling
- **Zero intermediate allocations**: Direct index computation without list creation

### Removed
- NIF stubs (pure Gleam implementation is sufficient)

## [1.3.0] - 2026-01-26

### Added
- **conv2d**: Native 2D convolution supporting multiple input formats
- **pad2d/pad4d**: Zero padding for 2D and 4D tensors
- **max_pool2d**: Max pooling with configurable kernel and stride
- **avg_pool2d**: Average pooling with configurable kernel and stride
- **global_avg_pool2d**: Global average pooling

## [1.2.1] - 2026-01-26

### Added
- **slice**: Tensor slicing with start/end indices

## [1.2.0] - 2026-01-25

### Added
- Quantization support (INT8, NF4, AWQ)
- Auto-backend selection
- 8x memory reduction for quantized tensors

## [1.1.0] - 2026-01-24

### Added
- Named tensors with semantic axes
- Broadcasting operations
- Zero-copy transpose via strides

## [1.0.0] - 2026-01-23

### Added
- Initial release
- Core tensor operations (zeros, ones, fill, from_list)
- Element-wise operations (add, sub, mul, div, scale)
- Reductions (sum, mean, max, min, argmax, argmin)
- Matrix operations (dot, matmul, transpose, outer)
- Shape operations (reshape, flatten, squeeze, unsqueeze)
