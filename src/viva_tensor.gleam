//// viva_tensor - NumPy for the BEAM.
////
//// Born from the frustration of "why can't I do tensor math in Erlang/Elixir
//// without calling Python?" Now you can.
////
//// The name: "viva" = alive in Portuguese/Spanish. Tensors that live on the BEAM.
//// Also, "viva" sounds better than "gleam_tensor" (sorry, marketing decision).
////
//// Architecture:
//// - core/ = the fundamentals (tensor, ops, shape, error)
//// - nn/ = neural network building blocks (layers, autograd, attention)
//// - quant/ = quantization for memory efficiency (INT8, NF4, AWQ)
//// - optim/ = hardware-specific optimizations
////
//// Performance tip: for matrices > 100x100, make sure the NIF is compiled.
//// The difference is ~100-1000x. No, that's not a typo.
////
//// ```gleam
//// import viva_tensor as t
////
//// let a = t.zeros([2, 3])
//// let b = t.ones([2, 3])
//// let assert Ok(c) = t.add(a, b)  // [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
//// ```

// Public API - re-exports from internal modules for convenience.
// Users can import viva_tensor and get everything they need.

import viva_tensor/tensor

// --- Types ------------------------------------------------------------------

pub type Tensor =
  tensor.Tensor

pub type TensorError =
  tensor.TensorError

pub type Conv2dConfig =
  tensor.Conv2dConfig

// --- Constructors -----------------------------------------------------------

/// All zeros. The tensor equivalent of a blank canvas.
pub fn zeros(shape: List(Int)) -> Tensor {
  tensor.zeros(shape)
}

/// Create tensor of ones
pub fn ones(shape: List(Int)) -> Tensor {
  tensor.ones(shape)
}

/// Create tensor filled with value
pub fn fill(shape: List(Int), value: Float) -> Tensor {
  tensor.fill(shape, value)
}

/// Create tensor from list (1D)
pub fn from_list(data: List(Float)) -> Tensor {
  tensor.from_list(data)
}

/// Create 2D tensor from list of lists
pub fn from_list2d(rows: List(List(Float))) -> Result(Tensor, TensorError) {
  tensor.from_list2d(rows)
}

/// Create vector (1D tensor)
pub fn vector(data: List(Float)) -> Tensor {
  tensor.vector(data)
}

/// Create matrix (2D tensor)
pub fn matrix(
  rows: Int,
  cols: Int,
  data: List(Float),
) -> Result(Tensor, TensorError) {
  tensor.matrix(rows, cols, data)
}

// --- Random -----------------------------------------------------------------

/// Random uniform [0, 1)
pub fn random_uniform(shape: List(Int)) -> Tensor {
  tensor.random_uniform(shape)
}

/// Tensor with normal random values
pub fn random_normal(shape: List(Int), mean: Float, std: Float) -> Tensor {
  tensor.random_normal(shape, mean, std)
}

/// Xavier initialization for neural network weights
pub fn xavier_init(fan_in: Int, fan_out: Int) -> Tensor {
  tensor.xavier_init(fan_in, fan_out)
}

/// He initialization (for ReLU networks)
pub fn he_init(fan_in: Int, fan_out: Int) -> Tensor {
  tensor.he_init(fan_in, fan_out)
}

// --- Math -------------------------------------------------------------------

/// Add element-wise
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.add(a, b)
}

/// Element-wise subtraction
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.sub(a, b)
}

/// Element-wise multiplication
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.mul(a, b)
}

/// Element-wise division
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.div(a, b)
}

/// Scale by constant
pub fn scale(t: Tensor, s: Float) -> Tensor {
  tensor.scale(t, s)
}

/// Apply function to each element
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  tensor.map(t, f)
}

// --- Reductions -------------------------------------------------------------

/// Sum everything
pub fn sum(t: Tensor) -> Float {
  tensor.sum(t)
}

/// Mean of all elements
pub fn mean(t: Tensor) -> Float {
  tensor.mean(t)
}

/// Maximum value
pub fn max(t: Tensor) -> Float {
  tensor.max(t)
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  tensor.min(t)
}

/// Index of maximum value
pub fn argmax(t: Tensor) -> Int {
  tensor.argmax(t)
}

/// Index of minimum value
pub fn argmin(t: Tensor) -> Int {
  tensor.argmin(t)
}

/// Variance
pub fn variance(t: Tensor) -> Float {
  tensor.variance(t)
}

/// Standard deviation
pub fn std(t: Tensor) -> Float {
  tensor.std(t)
}

// --- Linear Algebra ---------------------------------------------------------

/// Dot product (vectors only)
pub fn dot(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  tensor.dot(a, b)
}

/// Matrix-matrix multiplication
pub fn matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.matmul(a, b)
}

/// Matrix-vector multiplication
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  tensor.matmul_vec(mat, vec)
}

/// Matrix transpose
pub fn transpose(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.transpose(t)
}

/// Outer product
pub fn outer(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.outer(a, b)
}

// --- Shape Ops --------------------------------------------------------------

/// Reshape (total size must match)
pub fn reshape(t: Tensor, new_shape: List(Int)) -> Result(Tensor, TensorError) {
  tensor.reshape(t, new_shape)
}

/// Flatten to 1D
pub fn flatten(t: Tensor) -> Tensor {
  tensor.flatten(t)
}

/// Remove dimensions of size 1
pub fn squeeze(t: Tensor) -> Tensor {
  tensor.squeeze(t)
}

/// Add dimension of size 1
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  tensor.unsqueeze(t, axis)
}

// --- Accessors --------------------------------------------------------------

/// Shape as list of dimensions
pub fn shape(t: Tensor) -> List(Int) {
  tensor.shape(t)
}

/// Get total size
pub fn size(t: Tensor) -> Int {
  tensor.size(t)
}

/// Get rank (number of dimensions)
pub fn rank(t: Tensor) -> Int {
  tensor.rank(t)
}

/// Convert to list
pub fn to_list(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

// --- Utils ------------------------------------------------------------------

/// L2 norm (Euclidean length)
pub fn norm(t: Tensor) -> Float {
  tensor.norm(t)
}

/// Normalize to unit length
pub fn normalize(t: Tensor) -> Tensor {
  tensor.normalize(t)
}

/// Clamp values
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  tensor.clamp(t, min_val, max_val)
}

// --- Broadcasting -----------------------------------------------------------

/// Can these shapes broadcast together?
pub fn can_broadcast(a: List(Int), b: List(Int)) -> Bool {
  tensor.can_broadcast(a, b)
}

/// Add with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.add_broadcast(a, b)
}

/// Multiply with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.mul_broadcast(a, b)
}

// --- Strided (Zero-copy) ----------------------------------------------------

/// Convert to strided representation for O(1) element access
pub fn to_strided(t: Tensor) -> Tensor {
  tensor.to_strided(t)
}

/// Convert to contiguous tensor
pub fn to_contiguous(t: Tensor) -> Tensor {
  tensor.to_contiguous(t)
}

/// Zero-copy transpose
pub fn transpose_strided(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.transpose_strided(t)
}

/// Check if contiguous
pub fn is_contiguous(t: Tensor) -> Bool {
  tensor.is_contiguous(t)
}

// --- CNN Ops ----------------------------------------------------------------
// Convolution: the operation that made deep learning work for images.
// Yann LeCun et al. (1989) showed CNNs could recognize handwritten digits.
// 35 years later, we're using the same basic operation to generate cat pics.

/// Default conv2d config (3x3 kernel, stride 1, no padding)
pub fn conv2d_config() -> Conv2dConfig {
  tensor.conv2d_config()
}

/// Conv2d config with "same" padding
pub fn conv2d_same(kernel_h: Int, kernel_w: Int) -> Conv2dConfig {
  tensor.conv2d_same(kernel_h, kernel_w)
}

/// 2D Convolution
pub fn conv2d(
  input: Tensor,
  kernel: Tensor,
  config: Conv2dConfig,
) -> Result(Tensor, TensorError) {
  tensor.conv2d(input, kernel, config)
}

/// Pad 2D tensor with zeros
pub fn pad2d(t: Tensor, pad_h: Int, pad_w: Int) -> Result(Tensor, TensorError) {
  tensor.pad2d(t, pad_h, pad_w)
}

/// Pad 4D tensor with zeros
pub fn pad4d(t: Tensor, pad_h: Int, pad_w: Int) -> Result(Tensor, TensorError) {
  tensor.pad4d(t, pad_h, pad_w)
}

/// Max pooling 2D
pub fn max_pool2d(
  input: Tensor,
  pool_h: Int,
  pool_w: Int,
  stride_h: Int,
  stride_w: Int,
) -> Result(Tensor, TensorError) {
  tensor.max_pool2d(input, pool_h, pool_w, stride_h, stride_w)
}

/// Average pooling 2D
pub fn avg_pool2d(
  input: Tensor,
  pool_h: Int,
  pool_w: Int,
  stride_h: Int,
  stride_w: Int,
) -> Result(Tensor, TensorError) {
  tensor.avg_pool2d(input, pool_h, pool_w, stride_h, stride_w)
}

/// Global average pooling
pub fn global_avg_pool2d(input: Tensor) -> Result(Tensor, TensorError) {
  tensor.global_avg_pool2d(input)
}
