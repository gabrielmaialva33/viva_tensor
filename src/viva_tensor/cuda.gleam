//// CudaTensor - Persistent GPU Memory
////
//// Tensors that live on the GPU. Ideal for weights and heavy compute.
////
//// - **FP32 (CudaTensor)**: Standard precision. 40+ TFLOPS on RTX 4090.
//// - **FP16 (CudaTensor16)**: Low precision, high throughput using Tensor Cores. 330+ TFLOPS!
////
//// Data is uploaded once and stays on device.
//// Operations are launched asynchronously (mostly).

import viva_tensor/core/ffi

// =============================================================================
// FP32 CudaTensor
// =============================================================================

/// Reference to a tensor stored in GPU memory (FP32)
pub type CudaTensor =
  ffi.CudaTensorRef

/// Upload data to GPU (FP32)
pub fn new(data: List(Float), shape: List(Int)) -> Result(CudaTensor, String) {
  ffi.ct_from_list(data, shape)
}

/// Download data from GPU (FP32)
pub fn to_list(tensor: CudaTensor) -> Result(List(Float), String) {
  ffi.ct_to_list(tensor)
}

/// Get shape of tensor
pub fn shape(tensor: CudaTensor) -> Result(List(Int), String) {
  ffi.ct_shape(tensor)
}

/// Matrix Multiplication (FP32)
/// C = A @ B
pub fn matmul(
  a: CudaTensor,
  b: CudaTensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensor, String) {
  ffi.ct_matmul(a, b, m, n, k)
}

// =============================================================================
// FP16 CudaTensor (Tensor Cores)
// =============================================================================

/// Reference to a tensor stored in GPU memory (FP16)
pub type CudaTensor16 =
  ffi.CudaTensor16Ref

/// Check if FP16 Tensor Cores are available
pub fn fp16_available() -> Bool {
  ffi.ct16_available()
}

/// Upload data to GPU (converts f64 -> f16)
pub fn new16(
  data: List(Float),
  shape: List(Int),
) -> Result(CudaTensor16, String) {
  ffi.ct16_from_list(data, shape)
}

/// Download data from GPU (converts f16 -> f64)
pub fn to_list16(tensor: CudaTensor16) -> Result(List(Float), String) {
  ffi.ct16_to_list(tensor)
}

/// Get shape of FP16 tensor
pub fn shape16(tensor: CudaTensor16) -> Result(List(Int), String) {
  ffi.ct16_shape(tensor)
}

/// Matrix Multiplication (FP16 Tensor Cores)
/// C = A @ B
///
/// Uses HMMA (Half-precision Matrix Multiply Accumulate) instructions.
/// Expect massive speedups (up to 330 TFLOPS) if dimensions align with 16x16.
pub fn matmul16(
  a: CudaTensor16,
  b: CudaTensor16,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensor16, String) {
  ffi.ct16_matmul(a, b, m, n, k)
}
