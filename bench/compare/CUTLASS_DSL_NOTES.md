# CUTLASS 4 CuTeDSL Migration — Design Notes

Status: research / not started.

## Why migrate

Today viva_tensor's CUDA paths are hand-written C++ (`zig_src/cuda_*.cu`)
that template-instantiate CUTLASS kernels. Pros: stable, predictable.
Cons:

- Each new tile shape / data type / epilogue combo needs a new template
  instantiation + recompile (~30-60s).
- 7 hand-written `.cu` files (`cuda_fp8_cutlass.cu`,
  `cuda_int4_sparse_cutlass.cu`, etc) collectively define ~50 GEMM
  variants. Maintenance burden grows linearly.
- Hard to JIT-specialise for a runtime-known shape — we always pay
  ahead-of-time compile cost for every config in the autotune sweep.

CUTLASS 4 (April 2026) ships **CuTeDSL**: a Python-embedded DSL that
generates and JIT-compiles equivalent kernels in ~seconds. Same
performance, faster iteration, smaller surface area.

## Reference

- Official docs: https://docs.nvidia.com/cutlass/latest/overview.html
- FlashAttention 4 is written entirely in CuTeDSL — proof it's
  production-ready for Hopper/Blackwell. Ada SM89 is supported too.

## Minimal example (FP16 GEMM, SM89)

```python
# Hypothetical — adapt from NVIDIA's examples/python_dsl
import cutlass
import cutlass.cute as cute

@cute.kernel
def fp16_gemm_sm89(
    a: cute.Tensor[cute.Float16, cute.RowMajor],
    b: cute.Tensor[cute.Float16, cute.ColMajor],
    c: cute.Tensor[cute.Float16, cute.RowMajor],
):
    cute.gemm.tensor_op(
        compute_type=cute.Float16,
        accumulator=cute.Float16,    # full-rate Ada bypass
        tile_shape=(128, 128, 64),
        cluster_shape=(1, 1, 1),     # SM89 has no thread block clusters
        stages=3,
    )(a, b, c)

# Compile once per shape, cache the .cubin.
plan = fp16_gemm_sm89.specialise(M=4096, N=4096, K=4096)
plan.warmup()
plan.run(d_a, d_b, d_c)  # microseconds
```

## Migration plan (when we tackle it)

1. **Install CUTLASS Python package** — pin to a known-good version
   alongside our pinned Zig 0.15.2 / nvcc 13.2.

2. **First parity kernel: replace `cuda_fp8_cutlass.cu`** with the DSL
   equivalent. Existing `cutlass_fp8_bench` becomes a thin Python
   wrapper. Validate: same TFLOPS @ 4096² (~615 in current code).

3. **JIT cache**: emit `.cubin` to `priv/cutlass_cache/` keyed by
   `(shape, dtype, epilogue, sm_arch)`. Reuse across NIF calls.

4. **Wire JIT through NIF**: BEAM side calls `nt_jit_gemm_fp8(M, N, K,
   epilogue, ...)` which:
   - Looks up cached `.cubin` for the shape.
   - On miss, spawns Python/CuTeDSL subprocess to generate + cache it.
   - Loads `.cubin` via CUDA driver API, launches kernel.

5. **Autotune via DSL**: the current `autotune.gleam` sweeps fixed
   config IDs. CuTeDSL lets us sweep over `(tile_shape, stages,
   cluster_shape, swizzle)` as a 4-D space and cache the winner per
   shape. Same algorithm as today, much broader search space.

## What blocks "do it now"

- DSL is Python-only; viva_tensor's runtime is BEAM. The JIT subprocess
  bridge adds complexity that doesn't pay off unless we're generating
  many kernels at runtime.
- Our current 50-ish hand-written kernels already cover the shape /
  dtype / epilogue combinations our benchmarks exercise. Marginal value
  of DSL today: lower iteration cost for future kernels, not raw perf.

## Decision

Hold for now. Revisit when:
- We need a kernel that's not in our current matrix (e.g. NVFP4 fused
  GEMM, attention with custom mask, MoE expert dispatch).
- CUTLASS 4 ships a stable Ada SM89 examples directory we can crib from.
- Someone files a real bug that requires regenerating a kernel per-shape.
