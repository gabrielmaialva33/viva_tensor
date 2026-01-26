//// viva_tensor - Pure Gleam tensor library
////
//// N-dimensional arrays with named axes, broadcasting, and zero-copy views.
////
//// ## Features
//// - NumPy-inspired API
//// - Named tensors (Batch, Seq, Feature axes)
//// - Broadcasting
//// - Zero-copy transpose/reshape via strides
//// - O(1) random access with Erlang arrays
////
//// ## Quick Start
//// ```gleam
//// import viva_tensor as t
//// import viva_tensor/axis
////
//// // Create tensors
//// let a = t.zeros([2, 3])
//// let b = t.ones([2, 3])
////
//// // Operations
//// let c = t.add(a, b)
//// let d = t.matmul(a, t.transpose(b))
////
//// // Named tensors
//// let named = t.named.zeros([axis.batch(32), axis.feature(128)])
//// let summed = t.named.sum_along(named, axis.Batch)
//// ```

// Re-export core tensor module
import viva_tensor/tensor

// =============================================================================
// TENSOR CONSTRUCTORS (re-exports)
// =============================================================================

/// Create tensor of zeros
pub fn zeros(shape: List(Int)) -> tensor.Tensor {
  tensor.zeros(shape)
}

/// Create tensor of ones
pub fn ones(shape: List(Int)) -> tensor.Tensor {
  tensor.ones(shape)
}

/// Create tensor filled with value
pub fn fill(shape: List(Int), value: Float) -> tensor.Tensor {
  tensor.fill(shape, value)
}

/// Create tensor from list (1D)
pub fn from_list(data: List(Float)) -> tensor.Tensor {
  tensor.from_list(data)
}

/// Create 2D tensor from list of lists
pub fn from_list2d(
  rows: List(List(Float)),
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.from_list2d(rows)
}

/// Create vector (1D tensor)
pub fn vector(data: List(Float)) -> tensor.Tensor {
  tensor.vector(data)
}

/// Create matrix (2D tensor)
pub fn matrix(
  rows: Int,
  cols: Int,
  data: List(Float),
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.matrix(rows, cols, data)
}

// =============================================================================
// RANDOM CONSTRUCTORS
// =============================================================================

/// Tensor with uniform random values [0, 1)
pub fn random_uniform(shape: List(Int)) -> tensor.Tensor {
  tensor.random_uniform(shape)
}

/// Tensor with normal random values
pub fn random_normal(shape: List(Int), mean: Float, std: Float) -> tensor.Tensor {
  tensor.random_normal(shape, mean, std)
}

/// Xavier initialization for neural network weights
pub fn xavier_init(fan_in: Int, fan_out: Int) -> tensor.Tensor {
  tensor.xavier_init(fan_in, fan_out)
}

/// He initialization (for ReLU networks)
pub fn he_init(fan_in: Int, fan_out: Int) -> tensor.Tensor {
  tensor.he_init(fan_in, fan_out)
}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

/// Element-wise addition
pub fn add(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.add(a, b)
}

/// Element-wise subtraction
pub fn sub(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.sub(a, b)
}

/// Element-wise multiplication
pub fn mul(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.mul(a, b)
}

/// Element-wise division
pub fn div(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.div(a, b)
}

/// Scale by constant
pub fn scale(t: tensor.Tensor, s: Float) -> tensor.Tensor {
  tensor.scale(t, s)
}

/// Apply function to each element
pub fn map(t: tensor.Tensor, f: fn(Float) -> Float) -> tensor.Tensor {
  tensor.map(t, f)
}

// =============================================================================
// REDUCTION OPERATIONS
// =============================================================================

/// Sum all elements
pub fn sum(t: tensor.Tensor) -> Float {
  tensor.sum(t)
}

/// Mean of all elements
pub fn mean(t: tensor.Tensor) -> Float {
  tensor.mean(t)
}

/// Maximum value
pub fn max(t: tensor.Tensor) -> Float {
  tensor.max(t)
}

/// Minimum value
pub fn min(t: tensor.Tensor) -> Float {
  tensor.min(t)
}

/// Index of maximum value
pub fn argmax(t: tensor.Tensor) -> Int {
  tensor.argmax(t)
}

/// Index of minimum value
pub fn argmin(t: tensor.Tensor) -> Int {
  tensor.argmin(t)
}

/// Variance
pub fn variance(t: tensor.Tensor) -> Float {
  tensor.variance(t)
}

/// Standard deviation
pub fn std(t: tensor.Tensor) -> Float {
  tensor.std(t)
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

/// Dot product of two vectors
pub fn dot(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(Float, tensor.TensorError) {
  tensor.dot(a, b)
}

/// Matrix-matrix multiplication
pub fn matmul(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.matmul(a, b)
}

/// Matrix-vector multiplication
pub fn matmul_vec(
  mat: tensor.Tensor,
  vec: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.matmul_vec(mat, vec)
}

/// Transpose matrix
pub fn transpose(t: tensor.Tensor) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.transpose(t)
}

/// Outer product
pub fn outer(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.outer(a, b)
}

// =============================================================================
// SHAPE OPERATIONS
// =============================================================================

/// Reshape tensor
pub fn reshape(
  t: tensor.Tensor,
  new_shape: List(Int),
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.reshape(t, new_shape)
}

/// Flatten to 1D
pub fn flatten(t: tensor.Tensor) -> tensor.Tensor {
  tensor.flatten(t)
}

/// Remove dimensions of size 1
pub fn squeeze(t: tensor.Tensor) -> tensor.Tensor {
  tensor.squeeze(t)
}

/// Add dimension of size 1
pub fn unsqueeze(t: tensor.Tensor, axis_idx: Int) -> tensor.Tensor {
  tensor.unsqueeze(t, axis_idx)
}

// =============================================================================
// UTILITY
// =============================================================================

/// L2 norm
pub fn norm(t: tensor.Tensor) -> Float {
  tensor.norm(t)
}

/// Normalize to unit length
pub fn normalize(t: tensor.Tensor) -> tensor.Tensor {
  tensor.normalize(t)
}

/// Clamp values
pub fn clamp(t: tensor.Tensor, min_val: Float, max_val: Float) -> tensor.Tensor {
  tensor.clamp(t, min_val, max_val)
}

/// Get shape
pub fn shape(t: tensor.Tensor) -> List(Int) {
  t.shape
}

/// Get total size
pub fn size(t: tensor.Tensor) -> Int {
  tensor.size(t)
}

/// Get rank (number of dimensions)
pub fn rank(t: tensor.Tensor) -> Int {
  tensor.rank(t)
}

/// Convert to list
pub fn to_list(t: tensor.Tensor) -> List(Float) {
  tensor.to_list(t)
}

// =============================================================================
// BROADCASTING
// =============================================================================

/// Check if shapes can broadcast
pub fn can_broadcast(a: List(Int), b: List(Int)) -> Bool {
  tensor.can_broadcast(a, b)
}

/// Add with broadcasting
pub fn add_broadcast(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.add_broadcast(a, b)
}

/// Multiply with broadcasting
pub fn mul_broadcast(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.mul_broadcast(a, b)
}

// =============================================================================
// STRIDED / ZERO-COPY
// =============================================================================

/// Convert to strided tensor (O(1) access)
pub fn to_strided(t: tensor.Tensor) -> tensor.Tensor {
  tensor.to_strided(t)
}

/// Convert to contiguous tensor
pub fn to_contiguous(t: tensor.Tensor) -> tensor.Tensor {
  tensor.to_contiguous(t)
}

/// Zero-copy transpose
pub fn transpose_strided(
  t: tensor.Tensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  tensor.transpose_strided(t)
}

/// Check if contiguous
pub fn is_contiguous(t: tensor.Tensor) -> Bool {
  tensor.is_contiguous(t)
}
