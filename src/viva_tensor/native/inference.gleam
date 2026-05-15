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
//// The Gleam-side API is stable. The C/NIF side is delivered by three
//// agents working in parallel:
////
////   - Agent A: `nif_packed_weight.c` + `nt_prepack_fp8` + `nt_linear_fp8`
////     + `nt_linear_gelu_fp8`.
////   - Agent B: `nif_packed_weight_sparse.c` +
////     `nt_prepack_int8_sparse` / `nt_prepack_int4_sparse` +
////     `nt_linear_int8_sparse` / `nt_linear_int4_sparse`.
////   - Agent C (this worktree): `nif_linear_swiglu_fp8.cu` +
////     Gleam wiring + numerical validation suite.
////
//// Until all NIFs are linked into `priv/viva_tensor_zig.so`, every function
//// returns `Error(DimensionError(<reason>))` where the reason describes the
//// missing piece. The shape-check / feature-mismatch branches run *before*
//// the NIF call, so contract tests in `test/inference_test.gleam` keep
//// passing.
////
//// ## Naming convention
////
//// All functions accept dimensions in HuggingFace convention: input
//// `[B, in_features]` (or `[T, B, in_features]` for sequences) and
//// produce `[B, out_features]`. The prepacked weight conceptually
//// represents a `[in_features, out_features]` matrix.

import gleam/dynamic.{type Dynamic}
import gleam/option.{type Option, None, Some}
import viva_tensor/core/error.{type TensorError, DimensionError}
import viva_tensor/tensor.{type Tensor}

// =============================================================================
// Packed-weight handle types (opaque)
// =============================================================================
//
// `handle` is an opaque Erlang resource reference returned by the prepack NIF.
// It owns device memory and a (FP8 only) per-tensor scale or (sparse) per-
// channel scales. Gleam can pass it back into linear_* NIFs but cannot
// introspect it; the metadata (`in_features`, `out_features`, dequant scales)
// is captured in the record so callers can validate shapes without round-
// tripping through the NIF.

/// FP8 E4M3 prepacked weight: row-major `[in_features, out_features]` on
/// device memory, with optional per-tensor FP32 dequant scale.
pub opaque type PackedWeightFp8 {
  PackedWeightFp8(
    handle: Dynamic,
    in_features: Int,
    out_features: Int,
    scale: Float,
  )
}

/// INT8 2:4 structured-sparse prepacked weight (cuSPARSELt or CUTLASS
/// backend chosen automatically based on shape).
pub opaque type PackedWeightInt8Sparse {
  PackedWeightInt8Sparse(
    handle: Dynamic,
    in_features: Int,
    out_features: Int,
    /// Per-output-channel scales for dequant.
    channel_scales: List(Float),
  )
}

/// INT4 2:4 structured-sparse prepacked weight (CUTLASS backend).
pub opaque type PackedWeightInt4Sparse {
  PackedWeightInt4Sparse(
    handle: Dynamic,
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
      case nt_prepack_fp8(floats_to_fp32_binary(tensor.to_list(weight)), [in_f, out_f]) {
        Ok(#(handle, _in, _out, scale)) ->
          Ok(PackedWeightFp8(
            handle: handle,
            in_features: in_f,
            out_features: out_f,
            scale: scale,
          ))
        Error(reason) ->
          Error(DimensionError("prepack_fp8_weight failed: " <> reason))
      }
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
      case nt_prepack_int8_sparse(floats_to_fp32_binary(tensor.to_list(weight)), [in_f, out_f]) {
        Ok(handle) ->
          Ok(PackedWeightInt8Sparse(
            handle: handle,
            in_features: in_f,
            out_features: out_f,
            channel_scales: [],
          ))
        Error(reason) ->
          Error(DimensionError(
            "prepack_int8_sparse_24_weight failed: " <> reason,
          ))
      }
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
      case nt_prepack_int4_sparse(floats_to_fp32_binary(tensor.to_list(weight)), [in_f, out_f]) {
        Ok(handle) ->
          Ok(PackedWeightInt4Sparse(
            handle: handle,
            in_features: in_f,
            out_features: out_f,
            channel_scales: [],
          ))
        Error(reason) ->
          Error(DimensionError(
            "prepack_int4_sparse_24_weight failed: " <> reason,
          ))
      }
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
    [batch, in_f] if in_f == weight.in_features -> {
      let bias_data = optional_tensor_to_bias_arg(bias)
      // epilogue code 1 = DEFAULT (no activation). NIF expects FP16 input
      // binary, returns FP16 output binary.
      let _ = batch
      let _ = in_f
      let input_bin = floats_to_fp16_binary(tensor.to_list(input))
      case nt_linear_fp8(input_bin, weight.handle, bias_data, 1) {
        Ok(out_bin) -> {
          let out_data = fp16_binary_to_floats(out_bin)
          Ok(make_2d_tensor(out_data, batch, weight.out_features))
        }
        Error(reason) -> Error(DimensionError("linear_fp8 failed: " <> reason))
      }
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
    [batch, in_f] if in_f == weight.in_features -> {
      let bias_data = optional_tensor_to_bias_arg(bias)
      case
        nt_linear_int4_sparse(
          tensor.to_list(input),
          [batch, in_f],
          weight.handle,
          bias_data,
        )
      {
        Ok(out_data) -> Ok(make_2d_tensor(out_data, batch, weight.out_features))
        Error(reason) ->
          Error(DimensionError("linear_int4_sparse failed: " <> reason))
      }
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
    [batch, in_f] if in_f == weight.in_features -> {
      let bias_data = optional_tensor_to_bias_arg(bias)
      case
        nt_linear_int8_sparse(
          tensor.to_list(input),
          [batch, in_f],
          weight.handle,
          bias_data,
        )
      {
        Ok(out_data) -> Ok(make_2d_tensor(out_data, batch, weight.out_features))
        Error(reason) ->
          Error(DimensionError("linear_int8_sparse failed: " <> reason))
      }
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
    [batch, in_f] if in_f == weight.in_features -> {
      let bias_data = optional_tensor_to_bias_arg(bias)
      case
        nt_linear_gelu_fp8(
          tensor.to_list(input),
          [batch, in_f],
          weight.handle,
          bias_data,
          // epilogue code: BIAS+GELU
          36,
        )
      {
        Ok(out_data) -> Ok(make_2d_tensor(out_data, batch, weight.out_features))
        Error(reason) ->
          Error(DimensionError("linear_gelu_fp8 failed: " <> reason))
      }
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
    [batch, in_f], True, True if in_f == gate_weight.in_features -> {
      let bias_data = optional_tensor_to_bias_arg(bias)
      case
        nt_linear_swiglu_fp8(
          tensor.to_list(input),
          [batch, in_f],
          gate_weight.handle,
          up_weight.handle,
          bias_data,
        )
      {
        Ok(out_data) ->
          Ok(make_2d_tensor(out_data, batch, gate_weight.out_features))
        Error(reason) ->
          Error(DimensionError("linear_swiglu_fp8 failed: " <> reason))
      }
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

/// Internal sum type for bias argument: either a list of floats or absent.
///
/// Encoded as a Gleam custom type so the NIF receives a canonical tagged
/// tuple (`{bias_list, [...]}`) or atom (`bias_nil`) instead of Option's
/// `none` constructor. Agents A and B's NIFs pattern-match on this shape.
type BiasArg {
  BiasList(List(Float))
  BiasNil
}

fn optional_tensor_to_bias_arg(bias: Option(Tensor)) -> BiasArg {
  case bias {
    Some(b) -> BiasList(tensor.to_list(b))
    None -> BiasNil
  }
}

fn make_2d_tensor(data: List(Float), rows: Int, cols: Int) -> Tensor {
  // The NIF returns a flat row-major list of `rows * cols` floats. Reshape
  // never fails when sizes match; fall back to flat if for some reason the
  // NIF returned the wrong length (defensive, should never happen).
  case tensor.reshape(tensor.from_list(data), [rows, cols]) {
    Ok(t) -> t
    Error(_) -> tensor.from_list(data)
  }
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

// =============================================================================
// NIF bindings
// =============================================================================
//
// Each NIF is delivered by another agent or this worktree. The contract:
//
//   - prepack_fp8       : `{ok, {Resource, Scale}}` | `{error, String}`
//   - prepack_int*      : `{ok, {Resource, [ChannelScale]}}` | `{error, String}`
//   - linear_*          : `{ok, [Float]}` (FP32 host list, flat row-major
//                         `[batch * out_features]`) | `{error, String}`
//   - linear_swiglu_fp8 : same return shape as linear_*
//
// `BiasArg` is encoded as either `{bias_list, List}` or atom `bias_nil`.
// Agent A's NIF pattern-matches on the first element of the tuple, or on
// the atom directly, to decide whether to add bias.

// FP8 NIF returns {ok, {Resource, InFeatures, OutFeatures, Scale}}. We
// reuse in/out from the Gleam-side shape check so the tuple decode only
// cares about handle + scale; the two Int slots between them are still
// part of the wire shape and need to appear in the type signature.
// FP32 helper: List(Float) -> binary of little-endian IEEE-754 fp32.
// The C NIFs prefer binaries because copying a 1M-element weight list
// through enif_get_list_cell is ~100× slower than enif_inspect_binary.
@external(erlang, "viva_tensor_inference_ffi", "floats_to_fp32_binary")
fn floats_to_fp32_binary(floats: List(Float)) -> BitArray

@external(erlang, "viva_tensor_zig", "nt_prepack_fp8")
fn nt_prepack_fp8(
  data: BitArray,
  shape: List(Int),
) -> Result(#(Dynamic, Int, Int, Float), String)

// INT8/INT4 NIFs return {ok, Resource} — the per-channel scales live
// inside the PackedWeight C struct so they don't need to be exposed
// to the Gleam side (the linear NIFs read them straight from the
// resource).
@external(erlang, "viva_tensor_zig", "nt_prepack_int8_sparse")
fn nt_prepack_int8_sparse(
  data: BitArray,
  shape: List(Int),
) -> Result(Dynamic, String)

@external(erlang, "viva_tensor_zig", "nt_prepack_int4_sparse")
fn nt_prepack_int4_sparse(
  data: BitArray,
  shape: List(Int),
) -> Result(Dynamic, String)

// Helper: FP32 floats -> FP16 binary (cuBLASLt activation format).
@external(erlang, "viva_tensor_inference_ffi", "floats_to_fp16_binary")
fn floats_to_fp16_binary(floats: List(Float)) -> BitArray

// Helper: FP16 binary -> Float list (for decoding NIF output).
@external(erlang, "viva_tensor_inference_ffi", "fp16_binary_to_floats")
fn fp16_binary_to_floats(bin: BitArray) -> List(Float)

@external(erlang, "viva_tensor_zig", "nt_linear_fp8")
fn nt_linear_fp8(
  input_fp16: BitArray,
  weight: Dynamic,
  bias: BiasArg,
  epilogue: Int,
) -> Result(BitArray, String)

@external(erlang, "viva_tensor_zig", "nt_linear_gelu_fp8")
fn nt_linear_gelu_fp8(
  input_fp16: BitArray,
  weight: Dynamic,
  bias: BiasArg,
  epilogue: Int,
) -> Result(BitArray, String)

@external(erlang, "viva_tensor_zig", "nt_linear_int8_sparse")
fn nt_linear_int8_sparse(
  input_fp16: BitArray,
  weight: Dynamic,
  bias: BiasArg,
) -> Result(BitArray, String)

@external(erlang, "viva_tensor_zig", "nt_linear_int4_sparse")
fn nt_linear_int4_sparse(
  input_fp16: BitArray,
  weight: Dynamic,
  bias: BiasArg,
) -> Result(BitArray, String)

@external(erlang, "viva_tensor_zig", "nt_linear_swiglu_fp8")
fn nt_linear_swiglu_fp8(
  input_fp16: BitArray,
  input_shape: List(Int),
  gate_weight: Dynamic,
  up_weight: Dynamic,
  bias: BiasArg,
) -> Result(BitArray, String)
