//// High-level inference API on top of the championship CUTLASS / cuSPARSELt
//// kernels. The benchmark NIFs (`cutlass_*_bench`) are not meant for user
//// code — they allocate and free GEMM tensors on every call. This module
//// exposes a stable surface that's appropriate for actual inference:
////
////   1. **Prepack** an FP16/FP32 weight tensor into the layout the target
////      kernel expects (FP8 E4M3, INT8 2:4 sparse, INT4 2:4 sparse). The
////      returned handle owns device memory and amortises the packing cost
////      across many forward passes.
////
////   2. **Linear forward** takes an FP16/FP32 input plus the prepacked
////      weight and runs the matmul, optionally fusing an activation
////      (ReLU/GELU/SwiGLU) and bias via the cuBLASLt epilogue path.
////
//// ## Status
////
//// The Gleam-side API is stable; the C/NIF side is still partial. Many
//// functions in this module currently emit
//// `Error("nif_not_implemented")` and need the underlying `nt_prepack_*`
//// / `nt_linear_*` NIFs wired into `zig_src/nif_*.c`. The roadmap is in
//// `bench/compare/INFERENCE_API_PLAN.md`.
////
//// ## Naming convention
////
//// All functions accept dimensions in HuggingFace convention: input
//// `[B, in_features]` (or `[T, B, in_features]` for sequences) and
//// produce `[B, out_features]`. The prepacked weight conceptually
//// represents a `[in_features, out_features]` matrix.

import gleam/option.{type Option}
import viva_tensor/core/error.{type TensorError, DimensionError}
import viva_tensor/tensor.{type Tensor}

// =============================================================================
// Packed-weight handle types (opaque)
// =============================================================================

/// FP8 E4M3 prepacked weight: row-major `[in_features, out_features]` on
/// device memory, with optional per-tensor FP32 dequant scale.
pub opaque type PackedWeightFp8 {
  PackedWeightFp8(
    handle: Int,
    /// Stored just for introspection / shape validation.
    in_features: Int,
    out_features: Int,
    scale: Float,
  )
}

/// INT8 2:4 structured-sparse prepacked weight (cuSPARSELt or CUTLASS
/// backend chosen automatically based on shape).
pub opaque type PackedWeightInt8Sparse {
  PackedWeightInt8Sparse(
    handle: Int,
    in_features: Int,
    out_features: Int,
    /// Per-output-channel scales for dequant.
    channel_scales: List(Float),
  )
}

/// INT4 2:4 structured-sparse prepacked weight (CUTLASS backend).
pub opaque type PackedWeightInt4Sparse {
  PackedWeightInt4Sparse(
    handle: Int,
    in_features: Int,
    out_features: Int,
    channel_scales: List(Float),
  )
}

// =============================================================================
// Prepack — quantize + reorder a dense FP16/FP32 weight into the kernel layout
// =============================================================================

/// Quantize a dense FP16/FP32 weight into FP8 E4M3 and lay it out for the
/// CUTLASS FP8 GEMM kernel. The per-tensor `scale` is the absmax-scaled
/// dequantization factor.
///
/// `weight` shape must be exactly `[in_features, out_features]`. Errors
/// with `DimensionError` otherwise.
pub fn prepack_fp8_weight(
  weight: Tensor,
) -> Result(PackedWeightFp8, TensorError) {
  case tensor.shape(weight) {
    [in_f, out_f] -> {
      // TODO: NIF call into `nt_prepack_fp8(weight_ref, in_f, out_f)`
      // which: (1) finds absmax over the full tensor, (2) divides by
      // 448.0 (E4M3 max), (3) quantizes each element to FP8, (4) uploads
      // to device, (5) returns a resource handle.
      Error(DimensionError("prepack_fp8_weight: NIF not yet wired"))
      |> with_shape(in_f, out_f)
    }
    other ->
      Error(DimensionError(
        "prepack_fp8_weight: expected 2-D weight, got " <> shape_string(other),
      ))
  }
}

/// Quantize + 2:4-prune a weight into INT8 with structured sparsity, layout
/// matching the cuSPARSELt + CUTLASS sparse kernels.
pub fn prepack_int8_sparse_24_weight(
  weight: Tensor,
) -> Result(PackedWeightInt8Sparse, TensorError) {
  case tensor.shape(weight) {
    [in_f, out_f] -> {
      // TODO: NIF call into `nt_prepack_int8_sparse(weight_ref, in_f,
      // out_f)`. The NIF will:
      //   1. Compute per-output-channel absmax → per-channel scale
      //   2. Quantize element-wise to INT8 with channel scale
      //   3. Apply 2:4 magnitude pruning per row of 4
      //   4. Reorder into the cuSPARSELt / CUTLASS sparse layout
      //      (compressed indices + metadata `E`).
      let _ = #(in_f, out_f)
      Error(DimensionError("prepack_int8_sparse_24_weight: NIF not yet wired"))
    }
    other ->
      Error(DimensionError(
        "prepack_int8_sparse_24_weight: expected 2-D weight, got "
        <> shape_string(other),
      ))
  }
}

/// Quantize + 2:4-prune a weight into INT4 with structured sparsity,
/// layout matching the CUTLASS INT4 sparse kernel (winner at 1074 TFLOPS
/// @ 4096²). This is the highest-throughput inference path on Ada SM89.
pub fn prepack_int4_sparse_24_weight(
  weight: Tensor,
) -> Result(PackedWeightInt4Sparse, TensorError) {
  case tensor.shape(weight) {
    [in_f, out_f] -> {
      let _ = #(in_f, out_f)
      Error(DimensionError("prepack_int4_sparse_24_weight: NIF not yet wired"))
    }
    other ->
      Error(DimensionError(
        "prepack_int4_sparse_24_weight: expected 2-D weight, got "
        <> shape_string(other),
      ))
  }
}

// =============================================================================
// Linear forward — input × prepacked-weight + optional bias + optional activation
// =============================================================================

/// FP8 linear: `output = input @ weight + bias?`. Input is FP16/FP32, weight
/// is the FP8-prepacked handle, output is FP16. Bias (if present) is
/// added inside the cuBLASLt epilogue, no extra HBM round-trip.
pub fn linear_fp8(
  input: Tensor,
  weight: PackedWeightFp8,
  bias: Option(Tensor),
) -> Result(Tensor, TensorError) {
  case tensor.shape(input) {
    [_, in_f] if in_f == weight.in_features -> {
      let _ = bias
      Error(DimensionError("linear_fp8: NIF not yet wired"))
    }
    other ->
      Error(DimensionError(
        "linear_fp8: input feature dim mismatch (got "
        <> shape_string(other)
        <> ", weight expects "
        <> int_to_string(weight.in_features)
        <> ")",
      ))
  }
}

/// INT4 2:4 sparse linear: `output = input @ weight + bias?`. The fastest
/// path we have on Ada (1074 TFLOPS @ 4096²). Weight must have been
/// prepacked via `prepack_int4_sparse_24_weight`.
pub fn linear_int4_sparse(
  input: Tensor,
  weight: PackedWeightInt4Sparse,
  bias: Option(Tensor),
) -> Result(Tensor, TensorError) {
  case tensor.shape(input) {
    [_, in_f] if in_f == weight.in_features -> {
      let _ = bias
      Error(DimensionError("linear_int4_sparse: NIF not yet wired"))
    }
    other ->
      Error(DimensionError(
        "linear_int4_sparse: input feature dim mismatch (got "
        <> shape_string(other)
        <> ", weight expects "
        <> int_to_string(weight.in_features)
        <> ")",
      ))
  }
}

/// INT8 2:4 sparse linear with automatic backend dispatch — CUTLASS for
/// shapes ≤ 4096, cuSPARSELt for ≥ 8192 (the crossover discovered in
/// `autotune.gleam`).
pub fn linear_int8_sparse(
  input: Tensor,
  weight: PackedWeightInt8Sparse,
  bias: Option(Tensor),
) -> Result(Tensor, TensorError) {
  case tensor.shape(input) {
    [_, in_f] if in_f == weight.in_features -> {
      let _ = bias
      Error(DimensionError("linear_int8_sparse: NIF not yet wired"))
    }
    other ->
      Error(DimensionError(
        "linear_int8_sparse: input feature dim mismatch (got "
        <> shape_string(other)
        <> ", weight expects "
        <> int_to_string(weight.in_features)
        <> ")",
      ))
  }
}

/// FP8 linear fused with bias + GELU activation. Uses cuBLASLt epilogue
/// `BIAS_GELU` (code 36). Costs ~30% throughput vs plain linear due to
/// the SFU tanh in GELU, but avoids a second HBM round-trip vs the
/// non-fused version.
pub fn linear_gelu_fp8(
  input: Tensor,
  weight: PackedWeightFp8,
  bias: Option(Tensor),
) -> Result(Tensor, TensorError) {
  case tensor.shape(input) {
    [_, in_f] if in_f == weight.in_features -> {
      let _ = bias
      Error(DimensionError("linear_gelu_fp8: NIF not yet wired"))
    }
    other ->
      Error(DimensionError(
        "linear_gelu_fp8: input feature dim mismatch (got "
        <> shape_string(other)
        <> ", weight expects "
        <> int_to_string(weight.in_features)
        <> ")",
      ))
  }
}

/// FP8 SwiGLU: a fused `linear(input) ⊙ silu(linear(input_gate))` block,
/// the building block of Llama/Mistral FFN. Requires two prepacked
/// weights (gate and up projections); the down projection is a separate
/// `linear_fp8` call after this.
///
/// Output: `silu(input @ gate_weight) * (input @ up_weight)` then
/// optionally `+ bias`.
pub fn linear_swiglu_fp8(
  input: Tensor,
  gate_weight: PackedWeightFp8,
  up_weight: PackedWeightFp8,
  bias: Option(Tensor),
) -> Result(Tensor, TensorError) {
  case
    tensor.shape(input),
    gate_weight.in_features == up_weight.in_features,
    gate_weight.out_features == up_weight.out_features
  {
    [_, in_f], True, True if in_f == gate_weight.in_features -> {
      let _ = bias
      Error(DimensionError("linear_swiglu_fp8: NIF not yet wired"))
    }
    _, False, _ ->
      Error(DimensionError("linear_swiglu_fp8: gate/up in_features mismatch"))
    _, _, False ->
      Error(DimensionError("linear_swiglu_fp8: gate/up out_features mismatch"))
    other_shape, _, _ ->
      Error(DimensionError(
        "linear_swiglu_fp8: bad input shape " <> shape_string(other_shape),
      ))
  }
}

// =============================================================================
// Introspection helpers — accessor for testing / debug
// =============================================================================

/// Returns `(in_features, out_features)` of an FP8 packed weight.
pub fn fp8_features(w: PackedWeightFp8) -> #(Int, Int) {
  #(w.in_features, w.out_features)
}

/// Returns `(in_features, out_features)` of an INT8 2:4 packed weight.
pub fn int8_features(w: PackedWeightInt8Sparse) -> #(Int, Int) {
  #(w.in_features, w.out_features)
}

/// Returns `(in_features, out_features)` of an INT4 2:4 packed weight.
pub fn int4_features(w: PackedWeightInt4Sparse) -> #(Int, Int) {
  #(w.in_features, w.out_features)
}

// =============================================================================
// Private helpers
// =============================================================================

fn with_shape(
  res: Result(PackedWeightFp8, TensorError),
  in_f: Int,
  out_f: Int,
) -> Result(PackedWeightFp8, TensorError) {
  let _ = #(in_f, out_f)
  res
}

@external(erlang, "erlang", "integer_to_binary")
fn int_to_string(i: Int) -> String

fn shape_string(shape: List(Int)) -> String {
  "[" <> join_int_list(shape) <> "]"
}

fn join_int_list(xs: List(Int)) -> String {
  case xs {
    [] -> ""
    [x] -> int_to_string(x)
    [x, ..rest] -> int_to_string(x) <> ", " <> join_int_list(rest)
  }
}
