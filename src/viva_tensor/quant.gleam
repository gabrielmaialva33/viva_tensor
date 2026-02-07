//// Quantization - NF4 / INT8 Compression
////
//// Run larger models by compressing weights.
////
//// - **NF4 (Normal Float 4)**: 4-bit quantization optimized for normal distributions (neural weights).
////   Compresses 16-bit floats to 4 bits (4x memory reduction) with negligible accuracy loss.
////   Used in QLoRA and modern LLM inference.

import viva_tensor/core/ffi

/// Perform Matrix Multiplication with NF4 Quantized Weights
///
/// C = A @ Quantized(B)
///
/// Dequantizes B on-the-fly during computation. High compute density, low memory bandwidth.
///
/// - `a`: Input activations (NativeTensorRef)
/// - `b_indices`: Packed 4-bit indices (2 per byte) for weights
/// - `b_scales`: Scales for blocks of weights
/// - `block_size`: Size of quantization block (usually 64 or 128)
pub fn matmul_nf4(
  a: ffi.NativeTensorRef,
  b_indices: List(Int),
  b_scales: List(Float),
  m: Int,
  n: Int,
  k: Int,
  block_size: Int,
) -> ffi.NativeTensorRef {
  let assert Ok(res) =
    ffi.nt_matmul_nf4(a, b_indices, b_scales, m, n, k, block_size)
  res
}
