# Stable inference API — implementation roadmap

The Gleam-side surface for the championship CUTLASS / cuSPARSELt kernels
has landed in `src/viva_tensor/native/inference.gleam` and is re-exported
from `viva_tensor`. This doc tracks the work still needed on the C/NIF
side to make every function actually run instead of returning
`Error("nif_not_implemented")`.

## What ships today (May 2026)

Gleam API (stable, 775 tests passing):

```
type PackedWeightFp8
type PackedWeightInt8Sparse
type PackedWeightInt4Sparse

prepack_fp8_weight(Tensor) -> Result(PackedWeightFp8, _)
prepack_int8_sparse_24_weight(Tensor) -> Result(PackedWeightInt8Sparse, _)
prepack_int4_sparse_24_weight(Tensor) -> Result(PackedWeightInt4Sparse, _)

linear_fp8(input, weight, bias?) -> Result(Tensor, _)
linear_int4_sparse(input, weight, bias?) -> Result(Tensor, _)
linear_int8_sparse(input, weight, bias?) -> Result(Tensor, _)
linear_gelu_fp8(input, weight, bias?) -> Result(Tensor, _)
linear_swiglu_fp8(input, gate, up, bias?) -> Result(Tensor, _)

fp8_features / int8_features / int4_features  -- introspection
```

Today every function rejects non-2D weights and feature-dim mismatches up
front. When it reaches a valid shape, it returns
`Error(DimensionError("..._nif not yet wired"))`. That contract is
captured by `test/inference_test.gleam` (8 tests).

## What still needs to be built

### 1. Resource type for packed weights

Today's `cutlass_*_bench` NIFs allocate device memory and free it before
returning. For inference we want the device memory to **outlive** the
NIF call.

Erlang NIF resources handle this — register a `PackedWeight` resource
type in `nif_load()` and return a handle Gleam can keep:

```c
// zig_src/nif_entry.c
typedef struct {
    void* d_weight;          // FP8 / INT8 / INT4 device buffer
    void* d_metadata;        // sparsity metadata (INT8/INT4 only)
    void* d_scales;          // per-channel or per-tensor scales
    int in_features;
    int out_features;
    int dtype;               // FP8=0, INT8_SPARSE=1, INT4_SPARSE=2
    cublasLtMatmulAlgo_t cached_algo;  // best algo from sweep
} PackedWeight;

static ErlNifResourceType* PACKED_WEIGHT_RES = NULL;

static void packed_weight_destructor(ErlNifEnv* env, void* obj) {
    PackedWeight* w = (PackedWeight*)obj;
    if (w->d_weight)   cudaFree(w->d_weight);
    if (w->d_metadata) cudaFree(w->d_metadata);
    if (w->d_scales)   cudaFree(w->d_scales);
}
```

### 2. Prepack NIFs

For each packed-weight type:

```c
ERL_NIF_TERM nt_prepack_fp8(ErlNifEnv*, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_prepack_int8_sparse(ErlNifEnv*, int argc, const ERL_NIF_TERM argv[]);
ERL_NIF_TERM nt_prepack_int4_sparse(ErlNifEnv*, int argc, const ERL_NIF_TERM argv[]);
```

Each:
1. Reads the input weight tensor (FP16/FP32 host memory).
2. Computes scales (per-tensor for FP8, per-channel for INT8/INT4).
3. Quantizes element-wise.
4. (Sparse only) Applies 2:4 magnitude pruning, emits the metadata
   buffer expected by cuSPARSELt / CUTLASS.
5. Uploads to device memory.
6. Calls the autotuner once to find the best algo for this shape, caches
   it in the resource.
7. Returns the resource handle.

### 3. Linear forward NIFs

```c
ERL_NIF_TERM nt_linear_fp8(env, argc, argv);
// argv: [input_tensor, packed_weight, bias_or_nil, epilogue_code]
//   - input_tensor: ErlNifBinary (FP16, [B, in_features])
//   - packed_weight: PackedWeight resource handle
//   - bias_or_nil: optional FP16 [out_features]
//   - epilogue_code: 1=DEFAULT, 6=BIAS+RELU, 36=BIAS+GELU
// returns: {ok, output_binary} where output is FP16 [B, out_features]
```

Same pattern for `nt_linear_int8_sparse` and `nt_linear_int4_sparse`.

### 4. SwiGLU NIF

```c
ERL_NIF_TERM nt_linear_swiglu_fp8(env, argc, argv);
// argv: [input, gate_weight, up_weight, bias_or_nil]
```

Internally launches:
1. GEMM with `gate_weight` → `gate_out`
2. GEMM with `up_weight` → `up_out`
3. Element-wise `silu(gate_out) * up_out` (custom kernel)
4. Optional bias add

The two GEMMs can run on two streams concurrently (we already validated
multi-stream gives 1.00× on Ada SMs — so they'll serialize, but that's
fine for correctness).

### 5. Wire Gleam stubs to NIFs

`src/viva_tensor/native/inference.gleam` currently has `Error(...)` in
each function body. Replace with `@external(erlang, "viva_tensor_zig",
"nt_prepack_fp8")` calls and decode the binary outputs back into
`Tensor` records.

### 6. Numerical validation suite

`test/inference_numerical_test.gleam`:
- Reference matmul in FP32 on CPU.
- Prepack the same weight in FP8 / INT8-sparse / INT4-sparse.
- Run `linear_fp8` / `linear_int8_sparse` / `linear_int4_sparse`.
- Assert L2 norm difference is within the dtype's expected quantization
  error band (FP8 ≈ 1%, INT8 ≈ 1.5%, INT4 sparse ≈ 5%).

## Effort estimate

Item | Lines of code | Hours
---- | ---- | ----
1. Resource type | ~80 (C) | 1-2
2. Prepack NIFs × 3 | ~600 (C) | 6-8 (quant math is the bulk)
3. Linear NIFs × 3 | ~400 (C) | 3-4
4. SwiGLU NIF | ~200 (C/CUDA) | 2-3
5. Gleam stub wiring | ~50 lines | 1
6. Numerical tests | ~200 lines | 2

Total: ~1500 lines, 15-20 hours of focused work.

## Why ship the Gleam API first

The opaque types lock the user-facing contract. Once the NIFs land, no
caller has to change a single line — the same `t.prepack_fp8_weight(w)`
+ `t.linear_fp8(input, w, None)` calls just start returning real
results instead of `DimensionError("nif_not_implemented")`.

Inference frameworks (vivino, transformers-style) can build against
this surface today and have a runnable test suite for the shape /
error paths.
