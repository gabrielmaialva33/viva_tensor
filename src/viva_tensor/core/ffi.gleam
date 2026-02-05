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
