//// Data types (dtype) for tensors using phantom types
////
//// Phantom types provide compile-time type safety without runtime overhead.
//// This prevents accidentally mixing tensors of different dtypes.
////
//// ## Example
//// ```gleam
//// import viva_tensor/core/dtype
////
//// // These types are never constructed - they're just markers
//// let _: dtype.Float32 = panic  // Would fail
////
//// // Instead they're used as type parameters:
//// // Tensor(Float32) can only interact with Tensor(Float32)
//// ```

// =============================================================================
// PHANTOM TYPES - Data Types
// =============================================================================

/// 32-bit floating point (default)
pub type Float32

/// 16-bit floating point (half precision)
pub type Float16

/// 16-bit brain float (for ML)
pub type BFloat16

/// 8-bit signed integer (for INT8 quantization)
pub type Int8

/// 4-bit NormalFloat (for NF4/QLoRA quantization)
pub type NF4

/// 4-bit Activation-Aware (for AWQ quantization)
pub type AWQ

// =============================================================================
// DTYPE INFO
// =============================================================================

/// Runtime dtype information
pub type DtypeInfo {
  DtypeInfo(
    /// Name of the dtype
    name: String,
    /// Size in bytes per element
    bytes_per_element: Int,
    /// Whether it's a quantized type
    is_quantized: Bool,
  )
}

/// Get dtype info for Float32
pub fn float32_info() -> DtypeInfo {
  DtypeInfo(name: "float32", bytes_per_element: 4, is_quantized: False)
}

/// Get dtype info for Float16
pub fn float16_info() -> DtypeInfo {
  DtypeInfo(name: "float16", bytes_per_element: 2, is_quantized: False)
}

/// Get dtype info for BFloat16
pub fn bfloat16_info() -> DtypeInfo {
  DtypeInfo(name: "bfloat16", bytes_per_element: 2, is_quantized: False)
}

/// Get dtype info for Int8
pub fn int8_info() -> DtypeInfo {
  DtypeInfo(name: "int8", bytes_per_element: 1, is_quantized: True)
}

/// Get dtype info for NF4
pub fn nf4_info() -> DtypeInfo {
  // NF4 uses 4 bits = 0.5 bytes per element (plus scale overhead)
  DtypeInfo(name: "nf4", bytes_per_element: 1, is_quantized: True)
}

/// Get dtype info for AWQ
pub fn awq_info() -> DtypeInfo {
  DtypeInfo(name: "awq", bytes_per_element: 1, is_quantized: True)
}
