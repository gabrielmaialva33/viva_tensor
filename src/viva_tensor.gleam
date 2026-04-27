//// High-performance tensor operations for Gleam on the BEAM.
////
//// This module is the stable entry point for the package. It re-exports the
//// tensor type, common constructors, shape operations, linear algebra,
//// element-wise math, reductions, neural-network helpers, and TFLOPS
//// measurement utilities.
////
//// Lower-level implementation modules live under internal package namespaces
//// and are intentionally excluded from the public documentation. Prefer this
//// module unless you need a specialised submodule such as `viva_tensor/quant`
//// or `viva_tensor/nn/autograd`.
////
//// ```gleam
//// import gleam/result
//// import viva_tensor as t
////
//// let a = t.zeros([2, 3])
//// let b = t.ones([2, 3])
//// use c <- result.try(t.add(a, b))
//// c
//// ```

import viva_tensor/core/ffi
import viva_tensor/tensor
import viva_tensor/tflops as tflops_mod

// --- Types ------------------------------------------------------------------

/// A tensor value backed by dense, strided, or native storage.
pub type Tensor =
  tensor.Tensor

/// Error returned by fallible tensor constructors and operations.
pub type TensorError =
  tensor.TensorError

/// Opaque reference to a tensor stored in native NIF memory.
pub type NativeTensorRef =
ffi.NativeTensorRef

/// Configuration for two-dimensional convolution operations.
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

/// Wrap an existing native NIF tensor resource.
pub fn from_native_ref(ref: NativeTensorRef, shape: List(Int)) -> Tensor {
  tensor.from_native_ref(ref, shape)
}

/// Extract the native NIF tensor resource when present.
pub fn native_ref(t: Tensor) -> Result(NativeTensorRef, Nil) {
  tensor.native_ref(t)
}

/// Check whether a tensor is backed by native NIF memory.
pub fn is_native(t: Tensor) -> Bool {
  tensor.is_native(t)
}

/// Create a native-backed tensor of zeros.
pub fn native_zeros(shape: List(Int)) -> Result(Tensor, TensorError) {
  tensor.native_zeros(shape)
}

/// Create a native-backed tensor of ones.
pub fn native_ones(shape: List(Int)) -> Result(Tensor, TensorError) {
  tensor.native_ones(shape)
}

/// Create a native-backed tensor filled with a value.
pub fn native_fill(
shape: List(Int),
value: Float,
) -> Result(Tensor, TensorError) {
  tensor.native_fill(shape, value)
}

/// Create a native-backed tensor from row-major list data.
pub fn native_from_list(
data: List(Float),
shape: List(Int),
) -> Result(Tensor, TensorError) {
  tensor.native_from_list(data, shape)
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

/// Broadcast tensor to a target shape.
pub fn broadcast_to(
t: Tensor,
target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  tensor.broadcast_to(t, target_shape)
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

// --- TFLOPS Benchmarking ----------------------------------------------------

/// Backend used when measuring matrix-multiplication throughput.
pub type TflopsBackend =
  tflops_mod.Backend

/// Result returned by TFLOPS measurement helpers.
pub type TflopsResult =
  tflops_mod.TflopsResult

/// Measure TFLOPS for a single matmul operation
pub fn measure_tflops(
  backend: TflopsBackend,
  m: Int,
  n: Int,
  k: Int,
) -> TflopsResult {
  tflops_mod.measure_matmul(backend, m, n, k)
}

/// Measure averaged TFLOPS (warmup + iterations)
pub fn measure_tflops_averaged(
  backend: TflopsBackend,
  m: Int,
  n: Int,
  k: Int,
  iterations: Int,
) -> TflopsResult {
  tflops_mod.measure_matmul_averaged(backend, m, n, k, iterations)
}

/// Detect available compute backends
pub fn detect_backends() -> List(TflopsBackend) {
  tflops_mod.detect_backends()
}
