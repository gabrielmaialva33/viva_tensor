//// First-class single-precision (FP32) tensors.
////
//// `TensorF32` stores `float` data natively (half the memory of the default
//// FP64 `Tensor`) and runs matmul through SGEMM with no per-call conversion —
//// so it is faster than `viva_tensor.matmul_f32` even on small matrices.
////
//// Bridge to/from the FP64 world with `from_tensor` / `to_tensor`.
////
//// ```gleam
//// import viva_tensor/f32
////
//// let assert Ok(a) = f32.fill([1024, 1024], 0.5)
//// let assert Ok(b) = f32.fill([1024, 1024], 0.25)
//// let assert Ok(c) = f32.matmul(a, b)
//// ```

import gleam/result
import viva_tensor/core/ffi
import viva_tensor/tensor.{type Tensor}

/// A native FP32 tensor: opaque handle + cached shape.
pub opaque type TensorF32 {
  TensorF32(ref: ffi.NativeTensorF32Ref, shape: List(Int))
}

/// The tensor shape (row-major dimensions).
pub fn shape(t: TensorF32) -> List(Int) {
  t.shape
}

/// FP32 tensor of zeros.
pub fn zeros(shape: List(Int)) -> Result(TensorF32, String) {
  use ref <- result.map(ffi.ntf_zeros(shape))
  TensorF32(ref, shape)
}

/// FP32 tensor filled with `value`.
pub fn fill(shape: List(Int), value: Float) -> Result(TensorF32, String) {
  use ref <- result.map(ffi.ntf_fill(shape, value))
  TensorF32(ref, shape)
}

/// Build an FP32 tensor from a flat float list and a shape.
pub fn from_floats(
  data: List(Float),
  shape: List(Int),
) -> Result(TensorF32, String) {
  use ref <- result.map(ffi.ntf_from_list(data, shape))
  TensorF32(ref, shape)
}

/// Read an FP32 tensor back to a flat float list.
pub fn to_floats(t: TensorF32) -> Result(List(Float), String) {
  ffi.ntf_to_list(t.ref)
}

/// Total number of elements.
pub fn size(t: TensorF32) -> Result(Int, String) {
  ffi.ntf_size(t.ref)
}

/// Matrix multiply `[m, k] @ [k, n]` in pure FP32 (native SGEMM).
pub fn matmul(a: TensorF32, b: TensorF32) -> Result(TensorF32, String) {
  case a.shape, b.shape {
    [m, k], [k2, n] ->
      case k == k2 {
        True -> {
          use ref <- result.map(ffi.ntf_matmul(a.ref, b.ref, m, n, k))
          TensorF32(ref, [m, n])
        }
        False -> Error("shape_mismatch")
      }
    _, _ -> Error("expected_two_matrices")
  }
}

/// Down-convert a native FP64 `Tensor` to FP32. The source must be a native
/// tensor (e.g. built with `viva_tensor.native_*`); otherwise returns an error.
pub fn from_tensor(t: Tensor) -> Result(TensorF32, String) {
  case tensor.native_ref(t) {
    Ok(ref64) -> {
      use ref <- result.map(ffi.ntf_from_f64(ref64))
      TensorF32(ref, tensor.shape(t))
    }
    Error(_) -> Error("not_a_native_tensor")
  }
}

/// Up-convert this FP32 tensor to a native FP64 `Tensor`.
pub fn to_tensor(t: TensorF32) -> Result(Tensor, String) {
  use ref64 <- result.map(ffi.ntf_to_f64(t.ref))
  tensor.from_native_ref(ref64, t.shape)
}
