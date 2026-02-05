//// FFI - Foreign Function Interface to Erlang
////
//// Centralizes all Erlang external function bindings for the tensor library.
//// This module provides:
//// - O(1) array access via Erlang :array
//// - Math functions (sqrt, log, cos, sin, etc.)
//// - Random number generation

// =============================================================================
// TYPES
// =============================================================================

/// Opaque type for Erlang :array (immutable functional array)
/// Provides O(1) get/set operations vs O(n) for lists
pub type ErlangArray

// =============================================================================
// ARRAY OPERATIONS - O(1) Access
// =============================================================================

/// Convert list to Erlang array for O(1) access
pub fn list_to_array(lst: List(Float)) -> ErlangArray {
  list_to_array_ffi(lst)
}

/// Get element from array at index (O(1))
pub fn array_get(arr: ErlangArray, index: Int) -> Float {
  array_get_ffi(arr, index)
}

/// Get array size
pub fn array_size(arr: ErlangArray) -> Int {
  array_size_ffi(arr)
}

/// Convert array back to list
pub fn array_to_list(arr: ErlangArray) -> List(Float) {
  array_to_list_ffi(arr)
}

// =============================================================================
// OPTIMIZED ARRAY OPERATIONS - Fast Math
// =============================================================================

/// Dot product using Erlang arrays (O(1) access per element)
/// Much faster than list-based dot product for large vectors
pub fn array_dot(a: ErlangArray, b: ErlangArray) -> Float {
  array_dot_ffi(a, b)
}

/// Matrix multiplication using Erlang arrays
/// A is MxK, B is KxN -> Result is MxN
/// Uses O(1) array access, significantly faster than list indexing
pub fn array_matmul(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray {
  array_matmul_ffi(a, b, m, n, k)
}

/// Sum all elements in array
pub fn array_sum(arr: ErlangArray) -> Float {
  array_sum_ffi(arr)
}

/// Scale all elements by scalar
pub fn array_scale(arr: ErlangArray, scalar: Float) -> ErlangArray {
  array_scale_ffi(arr, scalar)
}

// =============================================================================
// NIF OPERATIONS - Apple Accelerate (macOS only)
// =============================================================================

/// Check if native NIF is loaded
/// Returns True on macOS with built NIF, False otherwise
pub fn is_nif_loaded() -> Bool {
  nif_is_loaded_ffi()
}

/// Get backend info string
pub fn nif_backend_info() -> String {
  nif_backend_info_ffi()
}

/// NIF-accelerated matrix multiplication (uses cblas_dgemm on macOS)
/// Falls back to pure Erlang if NIF not available
/// A[m,k] @ B[k,n] -> C[m,n]
pub fn nif_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  nif_matmul_ffi(a, b, m, n, k)
}

/// NIF-accelerated dot product (uses vDSP on macOS)
/// Falls back to pure Erlang if NIF not available
pub fn nif_dot(a: List(Float), b: List(Float)) -> Result(Float, String) {
  nif_dot_ffi(a, b)
}

/// NIF-accelerated sum (uses vDSP on macOS)
pub fn nif_sum(data: List(Float)) -> Result(Float, String) {
  nif_sum_ffi(data)
}

/// NIF-accelerated scale (uses vDSP on macOS)
pub fn nif_scale(data: List(Float), scalar: Float) -> Result(List(Float), String) {
  nif_scale_ffi(data, scalar)
}

// =============================================================================
// MATH FUNCTIONS
// =============================================================================

/// Square root
pub fn sqrt(x: Float) -> Float {
  sqrt_ffi(x)
}

/// Natural logarithm
pub fn log(x: Float) -> Float {
  log_ffi(x)
}

/// Exponential (e^x)
pub fn exp(x: Float) -> Float {
  exp_ffi(x)
}

/// Cosine
pub fn cos(x: Float) -> Float {
  cos_ffi(x)
}

/// Sine
pub fn sin(x: Float) -> Float {
  sin_ffi(x)
}

/// Tangent
pub fn tan(x: Float) -> Float {
  tan_ffi(x)
}

/// Hyperbolic tangent
pub fn tanh(x: Float) -> Float {
  tanh_ffi(x)
}

/// Power (x^y)
pub fn pow(x: Float, y: Float) -> Float {
  pow_ffi(x, y)
}

/// Absolute value
pub fn abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

// =============================================================================
// RANDOM
// =============================================================================

/// Uniform random float in [0, 1)
pub fn random_uniform() -> Float {
  random_uniform_ffi()
}

// =============================================================================
// CONSTANTS
// =============================================================================

/// Pi constant
pub const pi: Float = 3.14159265358979323846

/// Euler's number
pub const e: Float = 2.71828182845904523536

// =============================================================================
// FFI BINDINGS - Erlang External Functions
// =============================================================================

@external(erlang, "viva_tensor_ffi", "list_to_array")
fn list_to_array_ffi(lst: List(Float)) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_get")
fn array_get_ffi(arr: ErlangArray, index: Int) -> Float

@external(erlang, "viva_tensor_ffi", "array_size")
fn array_size_ffi(arr: ErlangArray) -> Int

@external(erlang, "viva_tensor_ffi", "array_to_list")
fn array_to_list_ffi(arr: ErlangArray) -> List(Float)

@external(erlang, "viva_tensor_ffi", "array_dot")
fn array_dot_ffi(a: ErlangArray, b: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_matmul")
fn array_matmul_ffi(
  a: ErlangArray,
  b: ErlangArray,
  m: Int,
  n: Int,
  k: Int,
) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_sum")
fn array_sum_ffi(arr: ErlangArray) -> Float

@external(erlang, "viva_tensor_ffi", "array_scale")
fn array_scale_ffi(arr: ErlangArray, scalar: Float) -> ErlangArray

@external(erlang, "math", "sqrt")
fn sqrt_ffi(x: Float) -> Float

@external(erlang, "math", "log")
fn log_ffi(x: Float) -> Float

@external(erlang, "math", "exp")
fn exp_ffi(x: Float) -> Float

@external(erlang, "math", "cos")
fn cos_ffi(x: Float) -> Float

@external(erlang, "math", "sin")
fn sin_ffi(x: Float) -> Float

@external(erlang, "math", "tan")
fn tan_ffi(x: Float) -> Float

@external(erlang, "math", "tanh")
fn tanh_ffi(x: Float) -> Float

@external(erlang, "math", "pow")
fn pow_ffi(x: Float, y: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_uniform_ffi() -> Float

// NIF bindings (viva_tensor_nif module)
@external(erlang, "viva_tensor_nif", "is_nif_loaded")
fn nif_is_loaded_ffi() -> Bool

@external(erlang, "viva_tensor_nif", "backend_info")
fn nif_backend_info_ffi() -> String

@external(erlang, "viva_tensor_nif", "matmul")
fn nif_matmul_ffi(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String)

@external(erlang, "viva_tensor_nif", "dot")
fn nif_dot_ffi(a: List(Float), b: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_nif", "sum")
fn nif_sum_ffi(data: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_nif", "scale")
fn nif_scale_ffi(data: List(Float), scalar: Float) -> Result(List(Float), String)

// =============================================================================
// TIMING
// =============================================================================

/// Get current time in microseconds (for benchmarking)
pub fn now_microseconds() -> Int {
  now_microseconds_ffi()
}

@external(erlang, "viva_tensor_ffi", "now_microseconds")
fn now_microseconds_ffi() -> Int

// =============================================================================
// ZIG SIMD NIF OPERATIONS
// =============================================================================

/// Check if Zig SIMD NIF is loaded
pub fn zig_is_loaded() -> Bool {
  zig_is_loaded_ffi()
}

/// Get Zig backend info
pub fn zig_backend_info() -> String {
  zig_backend_info_ffi()
}

/// Zig SIMD dot product
pub fn zig_dot(a: List(Float), b: List(Float)) -> Result(Float, String) {
  zig_dot_ffi(a, b)
}

/// Zig SIMD sum
pub fn zig_sum(data: List(Float)) -> Result(Float, String) {
  zig_sum_ffi(data)
}

/// Zig SIMD scale
pub fn zig_scale(data: List(Float), scalar: Float) -> Result(List(Float), String) {
  zig_scale_ffi(data, scalar)
}

/// Zig SIMD element-wise add
pub fn zig_add(a: List(Float), b: List(Float)) -> Result(List(Float), String) {
  zig_add_ffi(a, b)
}

/// Zig SIMD matrix multiplication
pub fn zig_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String) {
  zig_matmul_ffi(a, b, m, n, k)
}

// Zig NIF FFI bindings
@external(erlang, "viva_tensor_zig", "is_loaded")
fn zig_is_loaded_ffi() -> Bool

@external(erlang, "viva_tensor_zig", "backend_info")
fn zig_backend_info_ffi() -> String

@external(erlang, "viva_tensor_zig", "simd_dot")
fn zig_dot_ffi(a: List(Float), b: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "simd_sum")
fn zig_sum_ffi(data: List(Float)) -> Result(Float, String)

@external(erlang, "viva_tensor_zig", "simd_scale")
fn zig_scale_ffi(data: List(Float), scalar: Float) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_add")
fn zig_add_ffi(a: List(Float), b: List(Float)) -> Result(List(Float), String)

@external(erlang, "viva_tensor_zig", "simd_matmul")
fn zig_matmul_ffi(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> Result(List(Float), String)
