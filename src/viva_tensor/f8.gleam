//// FP8 (E4M3) quantization — CPU numerical reference.
////
//// Implements the OFP8 **E4M3** format from Micikevicius et al., "FP8 Formats
//// for Deep Learning" (arXiv:2209.05433, NVIDIA/Arm/Intel): 1 sign / 4 exponent
//// / 3 mantissa bits, bias 7, no infinities, max magnitude **448**. Quantization
//// uses NVIDIA Transformer Engine's **per-tensor current scaling**
//// (`s = 448 / amax`).
////
//// No CPU has FP8 math units, so this is a *numerical* reference — it validates
//// quantization error on any machine. The production accelerator is the CUDA
//// CUTLASS FP8 path. `matmul` here emulates an FP8 GEMM by quantizing both
//// inputs to E4M3, multiplying in FP32, and rescaling.
////
//// ```gleam
//// import viva_tensor
//// import viva_tensor/f8
////
//// let assert Ok(w) = viva_tensor.native_from_list(weights, [256, 256])
//// let assert Ok(wq) = f8.quantize(w)          // round-trip through E4M3
//// let err = f8.relative_l2_error(w, wq)        // typically < 6%
//// ```

import gleam/float
import gleam/list
import viva_tensor.{type Tensor, type TensorError}
import viva_tensor/core/ffi

/// Largest finite magnitude representable in E4M3 (`1.75 * 2^8`).
pub const e4m3_max: Float = 448.0

/// Fake-quantize a native FP64 tensor through E4M3 with per-tensor current
/// scaling and return the reconstructed FP64 tensor. Input must be a native
/// tensor (`viva_tensor.native_*`).
pub fn quantize(t: Tensor) -> Result(Tensor, String) {
  case viva_tensor.native_ref(t) {
    Ok(ref) ->
      case ffi.nt_quantize_e4m3(ref) {
        Ok(q) -> Ok(viva_tensor.from_native_ref(q, viva_tensor.shape(t)))
        Error(e) -> Error(e)
      }
    Error(_) -> Error("not_a_native_tensor")
  }
}

/// Emulated FP8 (E4M3) matrix multiply `[m,k] @ [k,n]`. Both inputs are
/// quantized to E4M3 (per-tensor scaling), multiplied in FP32, and rescaled.
/// Returns an FP64 tensor carrying the FP8 quantization error.
pub fn matmul(a: Tensor, b: Tensor) -> Result(Tensor, String) {
  case viva_tensor.shape(a), viva_tensor.shape(b) {
    [m, k], [k2, n] ->
      case k == k2 {
        True ->
          case viva_tensor.native_ref(a), viva_tensor.native_ref(b) {
            Ok(ra), Ok(rb) ->
              case ffi.nt_matmul_e4m3(ra, rb, m, n, k) {
                Ok(c) -> Ok(viva_tensor.from_native_ref(c, [m, n]))
                Error(e) -> Error(e)
              }
            _, _ -> Error("not_a_native_tensor")
          }
        False -> Error("shape_mismatch")
      }
    _, _ -> Error("expected_two_matrices")
  }
}

/// Relative L2 error `||reference - approx||_2 / ||reference||_2`, the standard
/// metric for quantization quality. Both tensors must have the same length.
pub fn relative_l2_error(reference: Tensor, approx: Tensor) -> Float {
  let r = viva_tensor.to_list(reference)
  let a = viva_tensor.to_list(approx)
  let #(num_sq, den_sq) =
    list.zip(r, a)
    |> list.fold(#(0.0, 0.0), fn(acc, pair) {
      let #(rv, av) = pair
      let d = rv -. av
      let #(num, den) = acc
      #(num +. d *. d, den +. rv *. rv)
    })
  case den_sq {
    0.0 -> 0.0
    _ -> sqrt(num_sq) /. sqrt(den_sq)
  }
}

fn sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

/// Re-export for callers building reference matmuls in full precision.
pub fn matmul_fp64(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  viva_tensor.matmul(a, b)
}
