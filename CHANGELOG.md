# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Fixed

- **cuBLASLt FP8 path now writes FP32 output buffers** (was FP16): the
  earlier path cast the f32 accumulator down to FP16 inside cuBLASLt and
  saturated ~5 cells per linear at large K, producing Inf for outlier
  activations. The fix mirrors what the CUTLASS path already did — use
  `CUDA_R_32F` for `layout_c`, return `float**` from `run_cublaslt_path`,
  and run per-row × per-channel dequant on FP32 host before the FP16 cast.
  Validated on the TinyLlama-1.1B smoke test: O proj output went from
  `mean_abs=160 + 5 Inf` to `mean_abs=0.0013`, final hidden state from
  `352` to `0.504` (now matches input magnitude as a healthy transformer
  block should). The same change applies to the GELU epilogue path.

### Added

- **TinyLlama-1.1B layer-0 forward smoke test** (`dev/llama_smoke.erl`):
  end-to-end forward pass through a real transformer block using
  HuggingFace TinyLlama-1.1B-Chat weights. Loads SafeTensors via the
  new `viva_tensor_safetensors_ffi`, converts bf16 → fp32, transposes
  HF [out, in] → viva [in, out], prepacks all 7 linears in FP8, runs
  RMSNorm + Q/K/V projections + GQA attention + output projection +
  residual + post-attention RMSNorm + gate/up/silu·×/down + residual.
  All 2048 final hidden state values are finite. Validates that the
  inference API stack works on real ML weights, not just synthetic.
- `src/viva_tensor_safetensors_ffi.erl`: minimal SafeTensors loader.
  Parses JSON header via OTP 27's `json` module, exposes
  `read_tensor_bf16/2`, `bf16_to_fp32_binary/1`, `transpose_fp32/3`,
  and `rmsnorm_weight_to_fp32_list/1`. Enough to drive a Llama-style
  model from a `model.safetensors` blob.
- `cuda_int_sparse_run.cu`: three new debug-friendly C entrypoints —
  `cutlass_int4_sparse_reorder_meta_e` (direct shim to
  `cutlass::reorder_meta`), `cutlass_int4_sparse_uncompress_to_dense`
  (host round-trip validator using `cutlass::uncompress`), and
  `cutlass_int4_sparse_self_test` (pure-CUTLASS-driven kernel
  validation against a host dense reference). The self-test produces
  `diffs=0 max_abs_diff=0` on (256, 256, 256), proving the
  INT4 sparse Tensor Op kernel + reorder + uncompress path is
  byte-exact end-to-end.

### Changed

- `nif_prepack_int_sparse`: INT4 2:4 metadata is now derived from the
  already-pruned `h_quant` buffer (row-major) instead of re-reading `W` in
  column-major. The previous code disagreed with the quant loop and named
  pairs that were not the ones actually preserved.
- INT4 prepack metadata reorder is delegated to the CUTLASS-native shim
  `cutlass_int4_sparse_reorder_meta_e` instead of a hand-ported C version,
  eliminating any chance of layout transcription drift.

### Fixed

- **INT4 sparse `ldE` was wrong**: the run launcher passed
  `ldE = K / kSparse / kElementsPerElementE` (= K_words, the column count of
  E) when `GemmSparseUniversal` expects the `LayoutE` stride
  `extent.row() * kInterleave = M * 2` for `ColumnMajorInterleaved<2>`.
  After fixing this, the numerical error against the dense FP32 reference
  dropped from ~108% → ~55%. The same fix was applied to the INT8 sparse
  run launcher even though its primary path is cuSPARSELt.
- INT4 metadata loop layout bug: second loop read `W[k * out_features + o]`
  (column-major) while the first quantization loop used row-major, leading to
  the metadata naming different pairs than the ones in `h_quant`.

### Validated

- INT4 2:4 sparse pipeline is internally byte-exact:
  * Kernel + reorder + uncompress: byte-exact vs CUTLASS reference
    (`cutlass_int4_sparse_self_test` returns `diffs=0`).
  * Encoding round-trip: `uncompress(h_packed, h_meta) == h_quant`
    element-wise.
  * Reorder: hand-ported reorder produced same output as `cutlass::reorder_meta`.
- Remaining ~55% L2 error vs dense FP32 on random uniform weights is the
  inherent noise floor of 50% structured sparsity + INT4 quantization
  (variance scaling: √(K/2)/√K = 0.707 magnitude alone, plus quant). Real
  LLM weights with magnitude structure will see much smaller error.
- Numerical test tolerance set to 0.65 (above measured 0.55, with headroom).

## [2.2.101] - 2026-05-15

### Added

- High-level inference API in `viva_tensor/native/inference` (re-exported from
  `viva_tensor`) with opaque `PackedWeightFp8` / `PackedWeightInt8Sparse` /
  `PackedWeightInt4Sparse` handles and `prepack_*` / `linear_*` /
  `linear_gelu_fp8` / `linear_swiglu_fp8` functions. Replaces the bench-only
  `cutlass_*_bench` NIFs for actual inference workloads.
- C/NIF backend: `nif_packed_weight.{h,c}` (Erlang resource type with device
  memory lifetime), `nif_prepack_fp8.c` (host absmax → FP8 E4M3 quantize +
  device upload), `nif_linear_fp8.c` (CUTLASS f32acc + cuBLASLt BIAS/GELU
  epilogue), `nif_prepack_int_sparse.c` and `nif_linear_int_sparse.c` (INT8 /
  INT4 2:4 sparse via cuSPARSELt + CUTLASS), `nif_linear_swiglu_fp8.cu` (two
  FP8 GEMMs + custom `silu_mul` kernel).
- `viva_tensor_inference_ffi.erl`: helpers for FP32/FP16 binary marshalling
  used by the inference wiring.
- ggml-inspired dual scaling on the FP8 path: per-output-channel weight
  scales (one FP32 per column on device) and per-batch-row activation scales
  (one FP32 per row on host).
- Numerical validation suite (`test/inference_numerical_test.gleam`) with
  5 quantization paths (FP8 / INT8 sparse / INT4 sparse / GELU FP8 / SwiGLU
  FP8) measured against a reference FP32 matmul.

### Changed

- Inference NIFs use FP32 weight binaries and FP16 activation binaries (vs
  passing `List(Float)` from BEAM); switched to faster zero-copy paths.
- FP8 GEMM moved from f16-accum (660 TOPS, FP16 saturation at K ≥ 32) to
  f32-accum (330 TOPS on GeForce, numerically safe up to Llama-7B K=4096).
- CUTLASS sparse `ElementE` metadata buffers now sized via a runtime probe
  (`cutlass_int*_sparse_run_info`) instead of guessed constants. Prevents
  the heap-corruption that earlier assumptions caused on shapes ≥ K=256.

### Fixed

- INT4 sparse prepack no longer writes past its metadata buffer (was over-
  allocated mismatched against the encoder loop, stomping the heap).
- `floats_to_fp32_binary` / `floats_to_fp16_binary` / `fp16_binary_to_floats`
  produce correct round-trip values for the activation/weight pipelines.

### Performance (RTX 4090, end-to-end FP8 linear vs FP32 reference)

| K (hidden) | Relative L2 error |
|-----------:|------------------:|
|         32 |               5%  |
|        128 |               8%  |
|        256 |              10%  |
|        512 |              12%  |
|       1024 |              13%  |
|       4096 |              13%  |

Numbers are finite for every shape (no `Inf`); the cap around 13% comes from
the FP16 output cast saturation. A future FP32 output buffer (CUTLASS
template change) is what unlocks `<5%` end-to-end on Llama-scale `K`.

## [2.2.100] - 2026-04-27

### Changed

- Kept the package on the `2.2.100` release line while the public API continues
  to mature toward a stable tensor-library contract.
- Clarified `gleam.toml` package metadata with a less over-specific
  description and an explicit changelog documentation link.
- Rewrote the English API guide around the stable public modules exposed in the
  generated documentation.
- Cleaned public module documentation comments to keep the library surface
  professional and focused.
- Moved benchmark and example Gleam entrypoints from `src/` to `dev/` so the
  packaged library source only contains runtime modules.
- Added a `make test-no-nif` quality gate for CI-style validation when the
  native shared library is unavailable.
- Centralized row-major layout/indexing helpers in an internal
  `viva_tensor/core/layout_math` module and optimized `softmax_axis` to
  normalize each axis slice once.
- Added zero-stride broadcast views to `viva_tensor/core/tensor` and made
  `core/ops.broadcast_to` preserve views instead of materializing expanded
  lists.
- Exposed `softmax_axis`, `sub_broadcast`, `div_broadcast`, and
  `capabilities()` from the stable root API.
- Preserved dense and strided storage through safe `reshape`, `squeeze`, and
  `unsqueeze` operations where view semantics are valid.
- Added a small `make bench-regression` benchmark for stable public API hot
  paths.
- Added stable backend capability records and `plan_backend()` so callers can
  inspect BEAM CPU, native CPU, CUDA FP32/FP16/INT8, and sparse backend
  availability without touching experimental modules.
- Added structured backend rejection reasons to `TensorBackendPlan`.
- Added `matmul_planned()` to execute matrix multiplication through the stable
  backend planner with automatic fallback to pure Gleam.
- Added `device()`/`dtype()` helpers for basic tensor placement metadata.
- Added an English stability policy and a public API contract test for the
  stable root facade.
- Added third-party notice tracking for future permissively licensed ports from
  mature tensor libraries.
- Added public `broadcast_shape()`, `broadcast_shapes()`, and `broadcast_pair()`
  helpers inspired by mature NumPy/PyTorch shape contracts, with shared
  zero-stride view planning for broadcast element-wise operations.
- Added a project-structure guide based on current Gleam package conventions,
  documenting the stable facade, internal modules, development-only code,
  native FFI boundaries, and HexDocs pages.
- Reorganized internal implementation modules into `native/`, `observability/`,
  and `experimental/` namespaces, and removed the generated
  `bench/test_int8_imma` executable from source control.
- Split axis reduction, softmax-axis, boolean-axis reducers, and pure
  broadcasting helpers out of the large internal `viva_tensor/tensor` module
  into focused `core/tensor_axis` and `core/tensor_broadcast` modules.
- Split dense linear-algebra kernels for dot, matrix-vector multiplication,
  matrix multiplication, transpose, and outer product into
  `core/tensor_linalg`.
- Added `try_to_list()` and routed fallible dense fallback paths through it so
  native materialization failures are not silently converted into empty tensors.
- Extended error hygiene into internal core tensor, shape, broadcast, and
  auto-dispatch paths so fallible operations propagate materialization and
  indexing failures instead of filling with zero-like defaults.
- Added fallible `try_map()`, `try_scale()`, and `try_sum()` variants for code
  that must preserve native materialization failures while keeping existing
  infallible convenience functions compatible.
- Added fallible shape/layout helpers (`try_unsqueeze()`, `try_to_strided()`,
  `try_to_contiguous()`) so invalid axes and native materialization failures
  are explicit in code that needs strong error contracts.
- Exposed scalar convenience helpers (`add_scalar()`, `try_add_scalar()`,
  `negate()`, `try_negate()`) from the stable root facade.
- Added fallible quantization metrics (`try_mse()`, `try_mae()`, `try_rmse()`,
  `try_cosine_similarity()`, `try_snr_db()`, `try_max_error()`,
  `try_error_percentile()`, `try_outlier_percentage()`, `try_compute_all()`) so
  metrics code can reject shape mismatches, empty tensors, and native
  materialization failures instead of returning zero-like placeholders.
- Hardened axis reductions with fallible `try_sum_axis()`/`try_mean_axis()`,
  root-facade exports, and `keepdims` variants that avoid silent indexing
  defaults during dense fallback.
- Added `try_softmax_axis()` and routed public softmax-axis execution through a
  fallible implementation that preserves native materialization and slice
  indexing failures.
- Added fallible `try_max()`, `try_min()`, `try_argmax()`, and `try_argmin()`
  variants so empty tensors and native materialization failures are explicit
  instead of collapsing to zero-like defaults.
- Added fallible `try_mean()`, `try_product()`, `try_variance()`, and
  `try_std()` variants, plus root exports, so scalar reductions consistently
  expose empty-tensor and materialization failures.
- Added fallible shape/materialization helpers `try_clone()`, `try_flatten()`,
  and `try_concat()`, and hardened `reshape()`, `concat_axis()`, and `slice()`
  to propagate materialization and indexing failures instead of filling with
  empty or zero-like defaults.
- Added fallible `try_take_first()`, `try_take_last()`, `try_norm()`, and
  `try_normalize()` helpers so utility operations can preserve materialization
  failures while keeping the existing convenience API compatible.
- Added fallible scalar utility helpers `try_add_scalar()`, `try_negate()`, and
  `try_clamp()`, with `try_clamp()` exposed from the stable root facade.
- Added fallible quantization metrics and backed scalar math helpers with
  `gleam_community_maths`, replacing local logarithm/mean/percentile/cosine
  approximations with maintained community implementations.
- Added stable root utilities backed by `gleam_community_maths`: `linspace()`,
  `try_linspace()`, `logspace()`, `try_logspace()`, `is_close()`, and
  `all_close()`.
- Added cumulative and order-statistic reductions backed by
  `gleam_community_maths`: `cumsum()`, `try_cumsum()`, `cumprod()`,
  `try_cumprod()`, `median()`, `try_median()`, `percentile()`, and
  `try_percentile()`.
- Added NumPy-style creation helpers (`zeros_like()`, `ones_like()`,
  `full_like()`, `eye()`, `try_eye()`, `identity()`, `diag()`, `try_diag()`),
  vector distances/similarity, statistical normalization helpers, and
  `max_axis()`/`min_axis()` reductions with `keepdims` variants.
- Added axis-aware cumulative operations and statistical reductions:
  `cumsum_axis()`, `try_cumsum_axis()`, `cumprod_axis()`,
  `try_cumprod_axis()`, `variance_axis()`, `try_variance_axis()`,
  `std_axis()`, and `try_std_axis()`, including `keepdims` variants for
  variance and standard deviation.
- Added axis arg index reductions (`argmax_axis()`, `try_argmax_axis()`,
  `argmin_axis()`, `try_argmin_axis()`) and common element-wise math helpers
  (`abs()`, `square()`, `sqrt()`, `exp()`, `log()`) with fallible variants for
  domain-sensitive operations.
- Added more NumPy-style element-wise utilities (`clip()`, `floor()`,
  `ceil()`, `round()`, `sign()`, `reciprocal()`) plus broadcasting-aware
  `maximum()` and `minimum()` helpers.
- Added broadcasting-aware comparison masks (`equal()`, `not_equal()`,
  `greater()`, `greater_equal()`, `less()`, `less_equal()`) and `where()` for
  conditional tensor selection.
- Added numeric-mask logic helpers (`logical_not()`, `logical_and()`,
  `logical_or()`, `logical_xor()`) plus `any()`, `all()`, and
  `count_nonzero()` reductions.
- Added axis mask reductions (`any_axis()`, `all_axis()`,
  `count_nonzero_axis()`) and flattened indexing helpers (`take()`,
  `nonzero()`, `masked_select()`).
- Added native `NativeTensor` fast paths for broadcasting-aware `maximum()`,
  `minimum()`, comparison masks, numeric-mask logic, `where()`, and
  `count_nonzero()` while preserving pure Gleam fallbacks when the NIF is not
  available.
- Added future hardware profiles for Ada, Blackwell, Rubin, Vera, and Rubin CPX,
  plus quantization layout metadata for NVFP4/INT2/INT3 and reversible
  Hadamard preprocessing for low-bit quantization experiments.

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
- **TurboQuant-inspired vector compression**:
  - Data-oblivious randomized Hadamard rotation for online quantization
  - Low-bit scalar quantization in the rotated basis
  - Optional 1-bit residual correction for inner-product workloads
  - Pure Gleam reference path with tests before moving hot loops to NIF/CUDA
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
- Neural-network, quantization, sparse, telemetry, and BLAS helper modules are
  also hidden from generated package documentation until their public error
  types and resource ownership contracts are stable.
- Autograd backward functions now propagate `TensorError` instead of using
  `let assert` or `panic` inside library gradient code.
- Native tensor constructors now return `Error("nif_not_loaded")` instead of
  raising `NifNotLoaded` when CI or user environments do not have the Zig NIF
  shared library built.
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
