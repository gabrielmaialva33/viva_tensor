//// HDC - Hyperdimensional Computing
////
//// "One-shot learning via binary vectors"
////
//// Operations on high-dimensional binary vectors (default 10,048 dimensions).
////
//// ## Core Concepts
////
//// - **Binding**: XOR operation. Associates two concepts. Invertible (A XOR B XOR B = A).
//// - **Bundling**: Majority vote. Combines multiple vectors into a superposition.
//// - **Permutation**: Cyclic shift. Encodes sequence/order.
//// - **Similarity**: Hamming distance. Measures relatedness (1.0 = identical, 0.5 = random).
////
//// ## Example
////
//// ```gleam
//// import gleam/result
//// import viva_tensor/hdc
////
//// use a <- result.try(hdc.random(dim: hdc.default_dim, seed: 1))
//// use b <- result.try(hdc.random(dim: hdc.default_dim, seed: 2))
//// use c <- result.try(hdc.bind(a, b))
////
//// hdc.similarity(c, a)
//// ```

import viva_tensor/core/ffi

/// Hypervector reference (binary vector handle)
pub type HyperVector =
  ffi.HdcVectorRef

/// Default dimensionality (10,048 bits)
pub const default_dim = 10_048

/// Create a new empty hypervector (all zeros)
pub fn new(dim: Int) -> Result(HyperVector, String) {
  ffi.hdc_create(dim)
}

/// Create a random hypervector
pub fn random(dim: Int, seed: Int) -> Result(HyperVector, String) {
  ffi.hdc_random(dim, seed)
}

/// Bind two hypervectors (XOR)
///
/// Use to associate concepts: bind(role, filler)
/// e.g., bind(key_name, "Alice")
pub fn bind(a: HyperVector, b: HyperVector) -> Result(HyperVector, String) {
  ffi.hdc_bind(a, b)
}

/// Calculate similarity between two vectors
///
/// Returns Float in [0.0, 1.0]
/// 1.0 = Identical
/// 0.5 = Orthogonal (Unrelated)
/// 0.0 = Opposite (but in HDC usually everything is >= 0.5)
pub fn similarity(a: HyperVector, b: HyperVector) -> Result(Float, String) {
  ffi.hdc_similarity(a, b)
}

/// Permute vector (cyclic shift)
///
/// Use to encode sequence or order.
/// permute(A, 1) is different from A, but related.
pub fn permute(vec: HyperVector, shift: Int) -> Result(HyperVector, String) {
  ffi.hdc_permute(vec, shift)
}

/// Get dimensionality of the vector
pub fn dim(vec: HyperVector) -> Result(Int, String) {
  ffi.hdc_dim(vec)
}
