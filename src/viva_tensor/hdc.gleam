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
//// import viva_tensor/hdc
////
//// let a = hdc.random(seed: 1)
//// let b = hdc.random(seed: 2)
//// let c = hdc.bind(a, b)
////
//// hdc.similarity(c, a) // -> 0.5 (orthogonal)
//// hdc.similarity(hdc.bind(c, b), a) // -> 1.0 (retrieved)
//// ```

import viva_tensor/core/ffi

/// Hypervector reference (binary vector handle)
pub type HyperVector =
  ffi.HdcVectorRef

/// Default dimensionality (10,048 bits)
pub const default_dim = 10_048

/// Create a new empty hypervector (all zeros)
pub fn new(dim: Int) -> HyperVector {
  let assert Ok(vec) = ffi.hdc_create(dim)
  vec
}

/// Create a random hypervector
pub fn random(dim: Int, seed: Int) -> HyperVector {
  let assert Ok(vec) = ffi.hdc_random(dim, seed)
  vec
}

/// Bind two hypervectors (XOR)
///
/// Use to associate concepts: bind(role, filler)
/// e.g., bind(key_name, "Alice")
pub fn bind(a: HyperVector, b: HyperVector) -> HyperVector {
  let assert Ok(res) = ffi.hdc_bind(a, b)
  res
}

/// Calculate similarity between two vectors
///
/// Returns Float in [0.0, 1.0]
/// 1.0 = Identical
/// 0.5 = Orthogonal (Unrelated)
/// 0.0 = Opposite (but in HDC usually everything is >= 0.5)
pub fn similarity(a: HyperVector, b: HyperVector) -> Float {
  let assert Ok(sim) = ffi.hdc_similarity(a, b)
  sim
}

/// Permute vector (cyclic shift)
///
/// Use to encode sequence or order.
/// permute(A, 1) is different from A, but related.
pub fn permute(vec: HyperVector, shift: Int) -> HyperVector {
  let assert Ok(res) = ffi.hdc_permute(vec, shift)
  res
}

/// Get dimensionality of the vector
pub fn dim(vec: HyperVector) -> Int {
  let assert Ok(d) = ffi.hdc_dim(vec)
  d
}
