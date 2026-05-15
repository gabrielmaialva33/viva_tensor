# zig_src/

Native NIF source tree. Compiled with Zig (`zig build`) plus selective
nvcc steps for the CUDA `.cu` files that go into the CUTLASS static
libraries.

Top-level entry point is [`build.zig`](build.zig); BEAM-facing dispatch
sits in [`nif_entry.c`](nif_entry.c). Public C declarations are in
[`viva_nif.h`](viva_nif.h) (plus
[`nif_packed_weight.h`](nif_packed_weight.h) for the inference
resource type).

## File groups

The current layout is flat. The implicit groupings — useful when
navigating without breaking the build by moving files — are:

### Dispatch + shared types

| File                  | Purpose                                                              |
| :-------------------- | :------------------------------------------------------------------- |
| `nif_entry.c`         | BEAM-side dispatch, resource registration, `nif_funcs[]`.            |
| `nif_tensor_core.c`   | `NativeTensor` resource type + constructors / accessors.             |
| `nif_packed_weight.c` | `PackedWeight` resource type used by every inference NIF.            |
| `nif_packed_weight.h` | Layout of the `PackedWeight` struct (shared across prepack/linear).  |
| `nif_platform.c`      | CPU topology + BLAS backend detection.                               |
| `viva_nif.h`          | Public C declarations for every NIF + `NIF_FUNC_DECL` macro.         |
| `viva_zig.zig`        | Zig entry point (kept minimal — most work is in the `.c` files).     |

### CPU backend

| File              | Purpose                                                       |
| :---------------- | :------------------------------------------------------------ |
| `nif_cpu_ops.c`   | Element-wise / reductions / matmul / activations on CPU.      |
| `nif_legacy.c`    | Legacy list-based fallbacks (no native deps).                 |
| `cuda_gemm.c`     | MKL DGEMM dispatch (named historical; CPU path lives here).   |
| `accelerate.c`    | macOS Apple Accelerate path.                                  |

### CUDA backend — generic ops

| File                 | Purpose                                                    |
| :------------------- | :--------------------------------------------------------- |
| `nif_cuda_fp32.c`    | CUDA tensor type (FP32), persistent device memory.         |
| `nif_cuda_fp16.c`    | CUDA tensor type (FP16) + Tensor Core HGEMM benchmark NIFs. |
| `nif_cuda_int8.c`    | CUDA tensor type (INT8 IMMA Tensor Cores).                 |
| `nif_quant.c`        | Quantisation (INT8 / NF4 fused matmul).                    |
| `nif_specialized.c`  | Resonance / LNS, Horde physics, HDC backends.              |
| `nif_softmax.c`      | Softmax / LayerNorm / GELU scaffolding.                    |

### CUDA backend — sparse

| File                          | Purpose                                                                          |
| :---------------------------- | :------------------------------------------------------------------------------- |
| `nif_sparse.c`                | SparseTensor resource (cuSPARSELt FP16/INT8).                                    |
| `cuda_sparselt.c`             | cuSPARSELt 2:4 sparse GEMM wrapper.                                              |
| `cuda_sparse_int8_cutlass.cu` | CUTLASS INT8 2:4 sparse benchmarks (cfg 0–28).                                   |
| `cuda_cusparselt_int8.cu`     | cuSPARSELt INT8 2:4 sparse benchmark.                                            |
| `cuda_int4_sparse_cutlass.cu` | CUTLASS INT4 2:4 sparse benchmarks (cfg 0–36).                                   |
| `cuda_int_sparse_run.cu`      | Caller-allocated-memory CUTLASS launchers used by the prepack/linear NIFs below. |
| `cuda_cusparselt_layout_test.cu` | Layout validation for cuSPARSELt descriptors.                                  |

### CUDA backend — FP8 + sage attention

| File                        | Purpose                                                       |
| :-------------------------- | :------------------------------------------------------------ |
| `cuda_fp8_cutlass.cu`       | CUTLASS FP8 E4M3 GEMM (f16-accum + f32-accum + bench).        |
| `nif_sage_nif.c`            | SageAttention NIF (CPU + GPU paths).                          |
| `cuda_sage.c`               | SageAttention CUDA kernels.                                   |
| `sage/`                     | Vendored SageAttention kernels (fused / quant / gemm).        |

### Benchmark `.cu` files (packed into `libcutlass_fp8.a`)

These are the throughput probes used by `dev/viva_tensor/bench/*.gleam`:

| File                              | Purpose                                                |
| :-------------------------------- | :----------------------------------------------------- |
| `cuda_fp16_bench.cu`              | cublasLt FP16 `COMPUTE_16F` bench.                     |
| `cuda_fp16_fused_bench.cu`        | cublasLt FP16 with epilogue fusion (BIAS / GELU).      |
| `cuda_graph_bench.cu`             | CUDA Graphs vs loop-launch overhead probe.             |
| `cuda_multistream_bench.cu`       | Two-stream concurrent FP8 GEMM bench.                  |
| `cuda_cublaslt_algo_sweep.cu`     | cublasLt FP16 algorithm sweep (best of N heuristics).  |
| `cuda_nvfp4_emu.cu`               | NVFP4 dequant emulation kernel + bandwidth probe.      |
| `cuda_nvfp4_fused.cu`             | NVFP4 fused dequant + GEMM PoC.                        |

### Inference API (`PackedWeight` callers)

This is the stable production path delivered in 2.2.101. Every file
shares the `PackedWeight` resource type from `nif_packed_weight.{h,c}`:

| File                          | Purpose                                                              |
| :---------------------------- | :------------------------------------------------------------------- |
| `nif_prepack_fp8.c`           | FP32 host weight → FP8 E4M3 quantize (per-channel) → device upload.  |
| `nif_linear_fp8.c`            | FP8 GEMM with cublasLt epilogue (BIAS / GELU). Per-row activation +  |
|                               | per-channel weight dequant on the FP16 output.                       |
| `nif_prepack_int_sparse.c`    | Per-channel INT8 / INT4 quant + 2:4 magnitude prune + cuSPARSELt /   |
|                               | CUTLASS metadata layout.                                             |
| `nif_linear_int_sparse.c`     | Dispatch INT8 / INT4 2:4 sparse linear (cuSPARSELt ≥ 8192, CUTLASS   |
|                               | otherwise).                                                          |
| `nif_linear_swiglu_fp8.cu`    | Two FP8 GEMMs (gate + up) + `silu_mul` fused kernel for Llama FFN.   |

### Pre-compiled CUTLASS object archives

| File                          | Built from                                                 |
| :---------------------------- | :--------------------------------------------------------- |
| `libcutlass_fp8.a`            | `cuda_fp8_cutlass.cu` + every `cuda_*_bench.cu` + `nif_linear_swiglu_fp8.cu` |
| `libcusparselt_int8.a`        | `cuda_sparse_int8_cutlass.cu` + `cuda_cusparselt_int8.cu`  |
| `libcutlass_int4_sparse.a`    | `cuda_int4_sparse_cutlass.cu`                              |

Built by `make cutlass-libs` (calls nvcc on each `.cu` then `ar rcs`).
Build details and architectures controlled via `Makefile` variables —
see [`../Makefile`](../Makefile).

## Why is the layout flat?

Moving sources into subdirectories (`backend/cpu/…`, `backend/cuda/…`,
`inference/…`) would force matching changes in `build.zig`,
`viva_nif.h`'s `#include` paths, the Makefile's `cutlass-libs` archive
rules, and every relative `extern` declaration. The reorganisation is
on the roadmap once the inference API surface stabilises further;
until then, this README provides the same navigation without the
risk of breaking the build.

The historical context: `ggml` (under `tmp/ggml/`) keeps each operator
in a one-file `.cu` + `.cuh` pair under `src/ggml-cuda/`, which works
beautifully for kernels that are conceptually independent. Our path
through CUTLASS templates couples many of these (sparse INT4 ↔ INT8
share the `ElementE` layout probe, FP8 dense and FP8 SwiGLU both use
`cutlass_fp8_gemm_f16acc`), so a flat layout with this README is
clearer than a forced hierarchy.
