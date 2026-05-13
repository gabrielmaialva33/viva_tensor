//// High-performance tensor operations for Gleam on the BEAM.
////
//// This module is the stable entry point for the package. It exposes the
//// tensor type, common constructors, shape operations, linear algebra,
//// element-wise math, reductions, native acceleration helpers, and layout
//// inspection.
////
//// Lower-level implementation, backend, neural-network, quantization, sparse,
//// telemetry, and benchmark modules are intentionally excluded from the public
//// documentation until their contracts are stable. The related
//// `viva_tensor/layout`, `viva_tensor/axis`, and `viva_tensor/named` modules
//// are public when callers need explicit layout metadata or named dimensions.
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

import gleam/list
import gleam/result
import viva_tensor/core/error.{DimensionError}
import viva_tensor/core/ffi
import viva_tensor/cuda
import viva_tensor/layout as tensor_layout
import viva_tensor/tensor
import viva_tensor/tflops as tflops_mod

// --- Types ------------------------------------------------------------------

/// A tensor value backed by dense, strided, or native storage.
pub type Tensor =
  tensor.Tensor

/// Tensor payload representation.
pub type TensorStorage =
  tensor_layout.TensorStorage

/// Tensor payload location.
pub type TensorDevice =
  tensor_layout.TensorDevice

/// Tensor element type.
pub type TensorDtype =
  tensor_layout.TensorDtype

/// Canonical tensor layout metadata.
pub type TensorLayout =
  tensor_layout.TensorLayout

/// Error returned by fallible tensor constructors and operations.
pub type TensorError =
  tensor.TensorError

/// Backend device class used by runtime capability discovery.
pub type BackendDevice {
  BackendBeamCpu
  BackendNativeCpu
  BackendCuda
}

/// Element type supported by a runtime backend.
pub type BackendDtype {
  BackendFloat64
  BackendFloat32
  BackendFloat16
  BackendInt8
  BackendSparseFloat16
}

/// Stable backend names used by capability discovery and operation planning.
pub type TensorBackend {
  BackendPureGleam
  BackendZigSimd
  BackendMkl
  BackendCudaFp32
  BackendCudaFp16
  BackendCudaInt8
  BackendCudaSparse
}

/// Operation family used by the public backend planner.
pub type TensorOperation {
  OperationElementwise
  OperationBroadcast
  OperationReduction
  OperationSoftmax
  OperationMatmul(m: Int, n: Int, k: Int)
}

/// Operation family supported by a backend capability record.
pub type BackendOperation {
  BackendElementwise
  BackendBroadcast
  BackendReduction
  BackendSoftmax
  BackendMatmul
}

/// Runtime capability for one backend.
pub type BackendCapability {
  BackendCapability(
    backend: TensorBackend,
    available: Bool,
    device: BackendDevice,
    dtypes: List(BackendDtype),
    operations: List(BackendOperation),
    reason: String,
  )
}

/// Backend decision for a tensor operation.
pub type BackendRejection {
  BackendRejection(backend: TensorBackend, reason: String)
}

pub type TensorBackendPlan {
  TensorBackendPlan(
    operation: TensorOperation,
    selected: TensorBackend,
    fallbacks: List(TensorBackend),
    rejected: List(BackendRejection),
    reason: String,
  )
}

/// Runtime acceleration capabilities detected for this VM.
pub type TensorCapabilities {
  TensorCapabilities(
    nif_loaded: Bool,
    zig_loaded: Bool,
    backend_info: String,
    tflops_backends: List(TflopsBackend),
    backend_capabilities: List(BackendCapability),
  )
}

/// Result storage selected by the RTX-first planner.
pub type AcceleratedTensor =
  cuda.AcceleratedTensor

/// Backend selected by the RTX-first planner.
pub type AccelerationBackend =
  cuda.AccelerationBackend

/// Workspace for persistent GPU buffers.
pub type GpuWorkspace =
  cuda.GpuWorkspace

/// Persisted linear layer parameters.
pub type LinearLayer =
  cuda.LinearLayer

/// Opaque reference to a tensor stored in native NIF memory.
pub type NativeTensorRef =
  ffi.NativeTensorRef

/// Configuration for two-dimensional convolution operations.
pub type Conv2dConfig =
  tensor.Conv2dConfig

// --- Constructors -----------------------------------------------------------

/// Create a tensor filled with zeros.
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

/// Create a 1D tensor with evenly spaced values over a closed interval.
pub fn linspace(start: Float, stop: Float, steps: Int) -> Tensor {
  tensor.linspace(start, stop, steps)
}

/// Create a 1D tensor with evenly spaced values over a closed interval.
pub fn try_linspace(
  start: Float,
  stop: Float,
  steps: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_linspace(start, stop, steps)
}

/// Create a 1D tensor with logarithmically spaced values.
pub fn logspace(start: Float, stop: Float, steps: Int, base: Float) -> Tensor {
  tensor.logspace(start, stop, steps, base)
}

/// Create a 1D tensor with logarithmically spaced values.
pub fn try_logspace(
  start: Float,
  stop: Float,
  steps: Int,
  base: Float,
) -> Result(Tensor, TensorError) {
  tensor.try_logspace(start, stop, steps, base)
}

/// Create a tensor with the same shape as another tensor, filled with zeros.
pub fn zeros_like(t: Tensor) -> Tensor {
  tensor.zeros_like(t)
}

/// Create a tensor with the same shape as another tensor, filled with ones.
pub fn ones_like(t: Tensor) -> Tensor {
  tensor.ones_like(t)
}

/// Create a tensor with the same shape as another tensor, filled with a value.
pub fn full_like(t: Tensor, value: Float) -> Tensor {
  tensor.full_like(t, value)
}

/// Create a square identity matrix.
pub fn eye(n: Int) -> Tensor {
  tensor.eye(n)
}

/// Create a square identity matrix.
pub fn try_eye(n: Int) -> Result(Tensor, TensorError) {
  tensor.try_eye(n)
}

/// Alias for `eye`.
pub fn identity(n: Int) -> Tensor {
  tensor.identity(n)
}

/// Create a square diagonal matrix from a 1D tensor.
pub fn diag(t: Tensor) -> Tensor {
  tensor.diag(t)
}

/// Create a square diagonal matrix from a 1D tensor.
pub fn try_diag(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_diag(t)
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

/// Move a tensor to the best persistent backend: RTX 4090 first, then MKL/CPU.
pub fn to_accelerated(t: Tensor) -> Result(AcceleratedTensor, TensorError) {
  cuda.to_accelerated(t)
}

/// Upload a tensor to persistent RTX 4090 FP16 memory.
pub fn to_rtx4090_fp16(t: Tensor) -> Result(AcceleratedTensor, TensorError) {
  cuda.to_rtx4090_fp16(t)
}

/// Upload a tensor to persistent RTX 4090 FP32 memory.
pub fn to_rtx4090_fp32(t: Tensor) -> Result(AcceleratedTensor, TensorError) {
  cuda.to_rtx4090_fp32(t)
}

/// Create an RTX 4090 FP16 workspace.
pub fn gpu_workspace() -> Result(GpuWorkspace, TensorError) {
  cuda.gpu_workspace()
}

/// Allocate a reusable zero-filled output buffer in workspace memory.
pub fn workspace_zeros(
  workspace: GpuWorkspace,
  shape: List(Int),
) -> Result(AcceleratedTensor, TensorError) {
  cuda.workspace_zeros(workspace, shape)
}

/// Move a tensor into workspace memory.
pub fn workspace_from_tensor(
  workspace: GpuWorkspace,
  tensor: Tensor,
) -> Result(AcceleratedTensor, TensorError) {
  cuda.workspace_from_tensor(workspace, tensor)
}

/// Create a persisted FP16 linear layer on the RTX.
pub fn linear_layer_fp16(
  weight: Tensor,
  bias: Tensor,
) -> Result(LinearLayer, TensorError) {
  cuda.linear_layer_fp16(weight, bias)
}

/// Create a persisted linear layer in workspace memory.
pub fn linear_layer(
  workspace: GpuWorkspace,
  weight: Tensor,
  bias: Tensor,
) -> Result(LinearLayer, TensorError) {
  cuda.linear_layer(workspace, weight, bias)
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

/// Write out = a + b into a preallocated native tensor.
pub fn add_into(out: Tensor, a: Tensor, b: Tensor) -> Result(Nil, TensorError) {
  tensor.add_into(out, a, b)
}

/// Write out = a - b into a preallocated native tensor.
pub fn sub_into(out: Tensor, a: Tensor, b: Tensor) -> Result(Nil, TensorError) {
  tensor.sub_into(out, a, b)
}

/// Write out = a * b into a preallocated native tensor.
pub fn mul_into(out: Tensor, a: Tensor, b: Tensor) -> Result(Nil, TensorError) {
  tensor.mul_into(out, a, b)
}

/// Scale by constant
pub fn scale(t: Tensor, s: Float) -> Tensor {
  tensor.scale(t, s)
}

/// Scale by constant, preserving materialization failures.
pub fn try_scale(t: Tensor, s: Float) -> Result(Tensor, TensorError) {
  tensor.try_scale(t, s)
}

/// Write out = a * scalar into a preallocated native tensor.
pub fn scale_into(
  out: Tensor,
  a: Tensor,
  scalar: Float,
) -> Result(Nil, TensorError) {
  tensor.scale_into(out, a, scalar)
}

/// Apply function to each element
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  tensor.map(t, f)
}

/// Apply function to each element, preserving materialization failures.
pub fn try_map(
  t: Tensor,
  f: fn(Float) -> Float,
) -> Result(Tensor, TensorError) {
  tensor.try_map(t, f)
}

/// Apply a binary function element-wise over tensors with the same shape.
pub fn map2(
  a: Tensor,
  b: Tensor,
  f: fn(Float, Float) -> Float,
) -> Result(Tensor, TensorError) {
  tensor.map2(a, b, f)
}

/// Softmax along one axis, preserving shape and normalizing each slice.
pub fn softmax_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.softmax_axis(t, axis)
}

/// Softmax along one axis, preserving materialization failures.
pub fn try_softmax_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_softmax_axis(t, axis)
}

// --- Reductions -------------------------------------------------------------

/// Sum everything
pub fn sum(t: Tensor) -> Float {
  tensor.sum(t)
}

/// Sum everything, preserving materialization failures.
pub fn try_sum(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_sum(t)
}

/// Mean of all elements
pub fn mean(t: Tensor) -> Float {
  tensor.mean(t)
}

/// Mean of all elements, preserving materialization and empty-tensor errors.
pub fn try_mean(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_mean(t)
}

/// Product of all elements.
pub fn product(t: Tensor) -> Float {
  tensor.product(t)
}

/// Product of all elements, preserving materialization failures.
pub fn try_product(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_product(t)
}

/// Cumulative sum over the flattened tensor, preserving the original shape.
pub fn cumsum(t: Tensor) -> Tensor {
  tensor.cumsum(t)
}

/// Cumulative sum over the flattened tensor, preserving materialization failures.
pub fn try_cumsum(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_cumsum(t)
}

/// Cumulative product over the flattened tensor, preserving the original shape.
pub fn cumprod(t: Tensor) -> Tensor {
  tensor.cumprod(t)
}

/// Cumulative product over the flattened tensor, preserving materialization failures.
pub fn try_cumprod(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_cumprod(t)
}

/// Cumulative sum along one axis, preserving the original shape.
pub fn cumsum_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.cumsum_axis(t, axis)
}

/// Cumulative sum along one axis, preserving materialization failures.
pub fn try_cumsum_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_cumsum_axis(t, axis)
}

/// Cumulative product along one axis, preserving the original shape.
pub fn cumprod_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.cumprod_axis(t, axis)
}

/// Cumulative product along one axis, preserving materialization failures.
pub fn try_cumprod_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_cumprod_axis(t, axis)
}

/// Median value.
pub fn median(t: Tensor) -> Float {
  tensor.median(t)
}

/// Median value, preserving materialization and empty-tensor errors.
pub fn try_median(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_median(t)
}

/// Percentile using linear interpolation between closest ranks.
pub fn percentile(t: Tensor, percentile: Int) -> Float {
  tensor.percentile(t, percentile)
}

/// Percentile using linear interpolation between closest ranks.
pub fn try_percentile(
  t: Tensor,
  percentile: Int,
) -> Result(Float, TensorError) {
  tensor.try_percentile(t, percentile)
}

/// Sum along one axis.
pub fn sum_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.sum_axis(t, axis)
}

/// Sum along one axis, preserving the reduced dimension as size 1.
pub fn sum_axis_keepdims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.sum_axis_keepdims(t, axis)
}

/// Sum along one axis, preserving materialization failures.
pub fn try_sum_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_sum_axis(t, axis)
}

/// Sum along one axis with keepdims, preserving materialization failures.
pub fn try_sum_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_sum_axis_keepdims(t, axis)
}

/// Mean along one axis.
pub fn mean_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.mean_axis(t, axis)
}

/// Mean along one axis, preserving the reduced dimension as size 1.
pub fn mean_axis_keepdims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.mean_axis_keepdims(t, axis)
}

/// Mean along one axis, preserving materialization failures.
pub fn try_mean_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_mean_axis(t, axis)
}

/// Mean along one axis with keepdims, preserving materialization failures.
pub fn try_mean_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_mean_axis_keepdims(t, axis)
}

/// Maximum along one axis.
pub fn max_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.max_axis(t, axis)
}

/// Maximum along one axis, preserving the reduced dimension as size 1.
pub fn max_axis_keepdims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.max_axis_keepdims(t, axis)
}

/// Maximum along one axis, preserving materialization failures.
pub fn try_max_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_max_axis(t, axis)
}

/// Maximum along one axis with keepdims, preserving materialization failures.
pub fn try_max_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_max_axis_keepdims(t, axis)
}

/// Minimum along one axis.
pub fn min_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.min_axis(t, axis)
}

/// Minimum along one axis, preserving the reduced dimension as size 1.
pub fn min_axis_keepdims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.min_axis_keepdims(t, axis)
}

/// Minimum along one axis, preserving materialization failures.
pub fn try_min_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_min_axis(t, axis)
}

/// Minimum along one axis with keepdims, preserving materialization failures.
pub fn try_min_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_min_axis_keepdims(t, axis)
}

/// Maximum value
pub fn max(t: Tensor) -> Float {
  tensor.max(t)
}

/// Maximum value, preserving materialization and empty-tensor errors.
pub fn try_max(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_max(t)
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  tensor.min(t)
}

/// Minimum value, preserving materialization and empty-tensor errors.
pub fn try_min(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_min(t)
}

/// Index of maximum value
pub fn argmax(t: Tensor) -> Int {
  tensor.argmax(t)
}

/// Index of maximum value, preserving materialization and empty-tensor errors.
pub fn try_argmax(t: Tensor) -> Result(Int, TensorError) {
  tensor.try_argmax(t)
}

/// Index of minimum value
pub fn argmin(t: Tensor) -> Int {
  tensor.argmin(t)
}

/// Index of minimum value, preserving materialization and empty-tensor errors.
pub fn try_argmin(t: Tensor) -> Result(Int, TensorError) {
  tensor.try_argmin(t)
}

/// Variance
pub fn variance(t: Tensor) -> Float {
  tensor.variance(t)
}

/// Variance, preserving materialization and empty-tensor errors.
pub fn try_variance(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_variance(t)
}

/// Standard deviation
pub fn std(t: Tensor) -> Float {
  tensor.std(t)
}

/// Standard deviation, preserving materialization and empty-tensor errors.
pub fn try_std(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_std(t)
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

pub fn matmul_planned(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [m, k], [k2, n] if k == k2 -> {
      let plan = plan_backend(OperationMatmul(m: m, n: n, k: k))
      let caps = capabilities()
      let runnable =
        plan.fallbacks
        |> list.filter(fn(backend) {
          backend_is_available(backend, caps.backend_capabilities)
        })

      run_matmul_backends(a, b, m, n, k, runnable)
    }
    _, _ -> tensor.matmul(a, b)
  }
}

/// Matrix multiplication with priority: RTX 4090 first, then MKL/native CPU.
pub fn matmul_auto(
  a: Tensor,
  b: Tensor,
) -> Result(AcceleratedTensor, TensorError) {
  cuda.matmul_auto(a, b)
}

/// Matrix multiplication between persistent accelerated tensors.
pub fn matmul_accelerated(
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(AcceleratedTensor, TensorError) {
  cuda.matmul_accelerated(a, b)
}

/// Write `out = a @ b` into a persistent accelerated output buffer.
pub fn matmul_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(Nil, TensorError) {
  cuda.matmul_accelerated_into(out, a, b)
}

/// Write `out = relu(a @ b)` using the FP16 Tensor Core fused epilogue.
pub fn matmul_relu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(Nil, TensorError) {
  cuda.matmul_relu_accelerated_into(out, a, b)
}

/// Write `out = gelu(a @ b)` using the FP16 Tensor Core fused epilogue.
pub fn matmul_gelu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(Nil, TensorError) {
  cuda.matmul_gelu_accelerated_into(out, a, b)
}

/// Write `out = relu(a @ b + bias)` using the FP16 Tensor Core fused epilogue.
pub fn linear_relu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  bias: AcceleratedTensor,
) -> Result(Nil, TensorError) {
  cuda.linear_relu_accelerated_into(out, a, b, bias)
}

/// Write `out = gelu(a @ b + bias)` using the FP16 Tensor Core fused epilogue.
pub fn linear_gelu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  bias: AcceleratedTensor,
) -> Result(Nil, TensorError) {
  cuda.linear_gelu_accelerated_into(out, a, b, bias)
}

/// Allocate a reusable output buffer for a persisted linear layer.
pub fn linear_output(
  workspace: GpuWorkspace,
  layer: LinearLayer,
  batch_size: Int,
) -> Result(AcceleratedTensor, TensorError) {
  cuda.linear_output(workspace, layer, batch_size)
}

/// Run `out = relu(input @ layer.weight + layer.bias)`.
pub fn linear_relu_forward_into(
  out: AcceleratedTensor,
  input: AcceleratedTensor,
  layer: LinearLayer,
) -> Result(Nil, TensorError) {
  cuda.linear_relu_forward_into(out, input, layer)
}

/// Run `out = gelu(input @ layer.weight + layer.bias)`.
pub fn linear_gelu_forward_into(
  out: AcceleratedTensor,
  input: AcceleratedTensor,
  layer: LinearLayer,
) -> Result(Nil, TensorError) {
  cuda.linear_gelu_forward_into(out, input, layer)
}

/// Download an accelerated tensor back to a regular CPU tensor.
pub fn accelerated_to_tensor(
  t: AcceleratedTensor,
) -> Result(Tensor, TensorError) {
  cuda.to_cpu_tensor(t)
}

/// Inspect which backend was selected by `matmul_auto`.
pub fn accelerated_backend(t: AcceleratedTensor) -> AccelerationBackend {
  cuda.backend(t)
}

/// Shape of an accelerated tensor without forcing a download.
pub fn accelerated_shape(t: AcceleratedTensor) -> List(Int) {
  cuda.accelerated_shape(t)
}

/// Wait for queued CUDA work to complete.
pub fn accelerated_sync() -> Result(Nil, TensorError) {
  cuda.sync()
}

/// Workspace backend.
pub fn workspace_backend(workspace: GpuWorkspace) -> AccelerationBackend {
  cuda.workspace_backend(workspace)
}

/// Linear layer backend.
pub fn linear_layer_backend(layer: LinearLayer) -> AccelerationBackend {
  cuda.linear_layer_backend(layer)
}

/// Linear layer input feature count.
pub fn linear_layer_input_features(layer: LinearLayer) -> Int {
  cuda.linear_layer_input_features(layer)
}

/// Linear layer output feature count.
pub fn linear_layer_output_features(layer: LinearLayer) -> Int {
  cuda.linear_layer_output_features(layer)
}

/// Write out = a @ b into a preallocated native tensor.
pub fn matmul_into(
  out: Tensor,
  a: Tensor,
  b: Tensor,
) -> Result(Nil, TensorError) {
  tensor.matmul_into(out, a, b)
}

/// Fused linear layer with ReLU: max(0, a @ b + bias).
pub fn linear_relu(
  a: Tensor,
  b: Tensor,
  bias: Tensor,
) -> Result(Tensor, TensorError) {
  tensor.linear_relu(a, b, bias)
}

/// Write out = max(0, a @ b + bias) into a preallocated native tensor.
pub fn linear_relu_into(
  out: Tensor,
  a: Tensor,
  b: Tensor,
  bias: Tensor,
) -> Result(Nil, TensorError) {
  tensor.linear_relu_into(out, a, b, bias)
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

/// Flatten to 1D, preserving materialization failures.
pub fn try_flatten(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_flatten(t)
}

/// Remove dimensions of size 1
pub fn squeeze(t: Tensor) -> Tensor {
  tensor.squeeze(t)
}

/// Add dimension of size 1
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  tensor.unsqueeze(t, axis)
}

/// Add dimension of size 1, preserving invalid-axis errors.
pub fn try_unsqueeze(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_unsqueeze(t, axis)
}

/// Take flattened elements by explicit indices.
pub fn take(t: Tensor, indices: List(Int)) -> Tensor {
  tensor.take(t, indices)
}

/// Take flattened elements by explicit indices, preserving index errors.
pub fn try_take(t: Tensor, indices: List(Int)) -> Result(Tensor, TensorError) {
  tensor.try_take(t, indices)
}

/// Return flattened indices for non-zero values, represented as floats.
pub fn nonzero(t: Tensor) -> Tensor {
  tensor.nonzero(t)
}

/// Return flattened indices for non-zero values, preserving materialization failures.
pub fn try_nonzero(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_nonzero(t)
}

/// Select flattened values where a broadcasted mask is non-zero.
pub fn masked_select(t: Tensor, mask: Tensor) -> Tensor {
  tensor.masked_select(t, mask)
}

/// Select flattened values where a broadcasted mask is non-zero, preserving errors.
pub fn try_masked_select(
  t: Tensor,
  mask: Tensor,
) -> Result(Tensor, TensorError) {
  tensor.try_masked_select(t, mask)
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

/// Inspect storage, device, dtype, shape, strides, offset, size, and rank.
pub fn layout(t: Tensor) -> TensorLayout {
  tensor.layout(t)
}

/// Inspect where a tensor payload lives.
pub fn device(t: Tensor) -> TensorDevice {
  let info = layout(t)
  info.device
}

/// Inspect the tensor element type.
pub fn dtype(t: Tensor) -> TensorDtype {
  let info = layout(t)
  info.dtype
}

/// Convert to list
pub fn to_list(t: Tensor) -> List(Float) {
  tensor.to_list(t)
}

/// Convert to list, preserving native materialization failures.
pub fn try_to_list(t: Tensor) -> Result(List(Float), TensorError) {
  tensor.try_to_list(t)
}

// --- Utils ------------------------------------------------------------------

/// L2 norm (Euclidean length)
pub fn norm(t: Tensor) -> Float {
  tensor.norm(t)
}

/// L2 norm, preserving materialization failures.
pub fn try_norm(t: Tensor) -> Result(Float, TensorError) {
  tensor.try_norm(t)
}

/// Normalize to unit length
pub fn normalize(t: Tensor) -> Tensor {
  tensor.normalize(t)
}

/// Normalize to unit length, preserving materialization failures.
pub fn try_normalize(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_normalize(t)
}

/// Compare two scalars with relative and absolute tolerances.
pub fn is_close(a: Float, b: Float, rtol: Float, atol: Float) -> Bool {
  tensor.is_close(a, b, rtol, atol)
}

/// Compare two tensors element-wise and return whether all pairs are close.
pub fn all_close(
  a: Tensor,
  b: Tensor,
  rtol: Float,
  atol: Float,
) -> Result(Bool, TensorError) {
  tensor.all_close(a, b, rtol, atol)
}

/// Absolute value for every element.
pub fn abs(t: Tensor) -> Tensor {
  tensor.abs(t)
}

/// Absolute value for every element, preserving materialization failures.
pub fn try_abs(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_abs(t)
}

/// Square every element.
pub fn square(t: Tensor) -> Tensor {
  tensor.square(t)
}

/// Square every element, preserving materialization failures.
pub fn try_square(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_square(t)
}

/// Square root every element.
pub fn sqrt(t: Tensor) -> Tensor {
  tensor.sqrt(t)
}

/// Square root every element, rejecting negative values.
pub fn try_sqrt(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_sqrt(t)
}

/// Exponential for every element.
pub fn exp(t: Tensor) -> Tensor {
  tensor.exp(t)
}

/// Exponential for every element, preserving materialization failures.
pub fn try_exp(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_exp(t)
}

/// Natural logarithm for every element.
pub fn log(t: Tensor) -> Tensor {
  tensor.log(t)
}

/// Natural logarithm for every element, rejecting non-positive values.
pub fn try_log(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_log(t)
}

/// Floor every element.
pub fn floor(t: Tensor) -> Tensor {
  tensor.floor(t)
}

/// Floor every element, preserving materialization failures.
pub fn try_floor(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_floor(t)
}

/// Ceiling every element.
pub fn ceil(t: Tensor) -> Tensor {
  tensor.ceil(t)
}

/// Ceiling every element, preserving materialization failures.
pub fn try_ceil(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_ceil(t)
}

/// Round every element to the nearest integer value.
pub fn round(t: Tensor) -> Tensor {
  tensor.round(t)
}

/// Round every element to the nearest integer value, preserving failures.
pub fn try_round(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_round(t)
}

/// Return -1, 0, or 1 for each element.
pub fn sign(t: Tensor) -> Tensor {
  tensor.sign(t)
}

/// Return -1, 0, or 1 for each element, preserving failures.
pub fn try_sign(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_sign(t)
}

/// Reciprocal for every element.
pub fn reciprocal(t: Tensor) -> Tensor {
  tensor.reciprocal(t)
}

/// Reciprocal for every element, rejecting zeros.
pub fn try_reciprocal(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_reciprocal(t)
}

/// Euclidean distance between two same-shaped tensors, flattened as vectors.
pub fn euclidean_distance(a: Tensor, b: Tensor) -> Float {
  tensor.euclidean_distance(a, b)
}

/// Euclidean distance between two same-shaped tensors, preserving errors.
pub fn try_euclidean_distance(
  a: Tensor,
  b: Tensor,
) -> Result(Float, TensorError) {
  tensor.try_euclidean_distance(a, b)
}

/// Manhattan distance between two same-shaped tensors, flattened as vectors.
pub fn manhattan_distance(a: Tensor, b: Tensor) -> Float {
  tensor.manhattan_distance(a, b)
}

/// Manhattan distance between two same-shaped tensors, preserving errors.
pub fn try_manhattan_distance(
  a: Tensor,
  b: Tensor,
) -> Result(Float, TensorError) {
  tensor.try_manhattan_distance(a, b)
}

/// Cosine similarity between two same-shaped tensors, flattened as vectors.
pub fn cosine_similarity(a: Tensor, b: Tensor) -> Float {
  tensor.cosine_similarity(a, b)
}

/// Cosine similarity between two same-shaped tensors, preserving errors.
pub fn try_cosine_similarity(
  a: Tensor,
  b: Tensor,
) -> Result(Float, TensorError) {
  tensor.try_cosine_similarity(a, b)
}

/// Dot similarity between two same-shaped tensors, flattened as vectors.
pub fn dot_similarity(a: Tensor, b: Tensor) -> Float {
  tensor.dot_similarity(a, b)
}

/// Dot similarity between two same-shaped tensors, preserving errors.
pub fn try_dot_similarity(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  tensor.try_dot_similarity(a, b)
}

/// Z-score standardization over all elements, preserving shape.
pub fn zscore(t: Tensor) -> Tensor {
  tensor.zscore(t)
}

/// Z-score standardization over all elements, preserving errors.
pub fn try_zscore(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_zscore(t)
}

/// Alias for `zscore`.
pub fn standardize(t: Tensor) -> Tensor {
  tensor.standardize(t)
}

/// Alias for `try_zscore`.
pub fn try_standardize(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_standardize(t)
}

/// Scale all values into a target interval.
pub fn minmax_scale(
  t: Tensor,
  feature_min: Float,
  feature_max: Float,
) -> Tensor {
  tensor.minmax_scale(t, feature_min, feature_max)
}

/// Scale all values into a target interval, preserving errors.
pub fn try_minmax_scale(
  t: Tensor,
  feature_min: Float,
  feature_max: Float,
) -> Result(Tensor, TensorError) {
  tensor.try_minmax_scale(t, feature_min, feature_max)
}

/// Clip tensor L2 norm to at most `max_norm`.
pub fn clip_by_norm(t: Tensor, max_norm: Float) -> Tensor {
  tensor.clip_by_norm(t, max_norm)
}

/// Clip tensor L2 norm to at most `max_norm`, preserving errors.
pub fn try_clip_by_norm(
  t: Tensor,
  max_norm: Float,
) -> Result(Tensor, TensorError) {
  tensor.try_clip_by_norm(t, max_norm)
}

/// Add a scalar to every element.
pub fn add_scalar(t: Tensor, scalar: Float) -> Tensor {
  tensor.add_scalar(t, scalar)
}

/// Add a scalar to every element, preserving materialization failures.
pub fn try_add_scalar(t: Tensor, scalar: Float) -> Result(Tensor, TensorError) {
  tensor.try_add_scalar(t, scalar)
}

/// Negate every element.
pub fn negate(t: Tensor) -> Tensor {
  tensor.negate(t)
}

/// Negate every element, preserving materialization failures.
pub fn try_negate(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_negate(t)
}

/// Clamp values
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  tensor.clamp(t, min_val, max_val)
}

/// Clamp values, preserving materialization failures.
pub fn try_clamp(
  t: Tensor,
  min_val: Float,
  max_val: Float,
) -> Result(Tensor, TensorError) {
  tensor.try_clamp(t, min_val, max_val)
}

/// Alias for `clamp`.
pub fn clip(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  tensor.clip(t, min_val, max_val)
}

/// Alias for `try_clamp`.
pub fn try_clip(
  t: Tensor,
  min_val: Float,
  max_val: Float,
) -> Result(Tensor, TensorError) {
  tensor.try_clip(t, min_val, max_val)
}

// --- Broadcasting -----------------------------------------------------------

/// Can these shapes broadcast together?
pub fn can_broadcast(a: List(Int), b: List(Int)) -> Bool {
  tensor.can_broadcast(a, b)
}

/// Compute the common shape for two broadcastable shapes.
pub fn broadcast_shape(
  a: List(Int),
  b: List(Int),
) -> Result(List(Int), TensorError) {
  tensor.broadcast_shape(a, b)
}

/// Compute the common shape for any number of broadcastable shapes.
pub fn broadcast_shapes(
  shapes: List(List(Int)),
) -> Result(List(Int), TensorError) {
  tensor.broadcast_shapes(shapes)
}

/// Broadcast tensor to a target shape.
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  tensor.broadcast_to(t, target_shape)
}

/// Broadcast two tensors to their common shape.
pub fn broadcast_pair(
  a: Tensor,
  b: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  tensor.broadcast_pair(a, b)
}

/// Add with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.add_broadcast(a, b)
}

/// Subtract with broadcasting
pub fn sub_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.sub_broadcast(a, b)
}

/// Multiply with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.mul_broadcast(a, b)
}

/// Divide with broadcasting
pub fn div_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.div_broadcast(a, b)
}

/// Element-wise maximum with NumPy-style broadcasting.
pub fn maximum(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.maximum(a, b)
}

/// Alias for `maximum`.
pub fn try_maximum(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_maximum(a, b)
}

/// Element-wise minimum with NumPy-style broadcasting.
pub fn minimum(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.minimum(a, b)
}

/// Alias for `minimum`.
pub fn try_minimum(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_minimum(a, b)
}

/// Element-wise equality mask with NumPy-style broadcasting.
pub fn equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.equal(a, b)
}

/// Alias for `equal`.
pub fn try_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_equal(a, b)
}

/// Element-wise inequality mask with NumPy-style broadcasting.
pub fn not_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.not_equal(a, b)
}

/// Alias for `not_equal`.
pub fn try_not_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_not_equal(a, b)
}

/// Element-wise greater-than mask with NumPy-style broadcasting.
pub fn greater(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.greater(a, b)
}

/// Alias for `greater`.
pub fn try_greater(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_greater(a, b)
}

/// Element-wise greater-than-or-equal mask with NumPy-style broadcasting.
pub fn greater_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.greater_equal(a, b)
}

/// Alias for `greater_equal`.
pub fn try_greater_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_greater_equal(a, b)
}

/// Element-wise less-than mask with NumPy-style broadcasting.
pub fn less(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.less(a, b)
}

/// Alias for `less`.
pub fn try_less(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_less(a, b)
}

/// Element-wise less-than-or-equal mask with NumPy-style broadcasting.
pub fn less_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.less_equal(a, b)
}

/// Alias for `less_equal`.
pub fn try_less_equal(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_less_equal(a, b)
}

/// Select values from two tensors using a non-zero condition mask.
pub fn where(
  condition: Tensor,
  when_true: Tensor,
  when_false: Tensor,
) -> Result(Tensor, TensorError) {
  tensor.where(condition, when_true, when_false)
}

/// Alias for `where`.
pub fn try_where(
  condition: Tensor,
  when_true: Tensor,
  when_false: Tensor,
) -> Result(Tensor, TensorError) {
  tensor.try_where(condition, when_true, when_false)
}

/// Logical NOT over a numeric mask.
pub fn logical_not(t: Tensor) -> Tensor {
  tensor.logical_not(t)
}

/// Logical NOT over a numeric mask, preserving materialization failures.
pub fn try_logical_not(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_logical_not(t)
}

/// Logical AND over numeric masks with broadcasting.
pub fn logical_and(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.logical_and(a, b)
}

/// Alias for `logical_and`.
pub fn try_logical_and(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_logical_and(a, b)
}

/// Logical OR over numeric masks with broadcasting.
pub fn logical_or(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.logical_or(a, b)
}

/// Alias for `logical_or`.
pub fn try_logical_or(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_logical_or(a, b)
}

/// Logical XOR over numeric masks with broadcasting.
pub fn logical_xor(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.logical_xor(a, b)
}

/// Alias for `logical_xor`.
pub fn try_logical_xor(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_logical_xor(a, b)
}

/// Does the mask contain any non-zero value?
pub fn any(t: Tensor) -> Bool {
  tensor.any(t)
}

/// Does the mask contain any non-zero value, preserving materialization failures.
pub fn try_any(t: Tensor) -> Result(Bool, TensorError) {
  tensor.try_any(t)
}

/// Are all mask values non-zero?
pub fn all(t: Tensor) -> Bool {
  tensor.all(t)
}

/// Are all mask values non-zero, preserving materialization failures.
pub fn try_all(t: Tensor) -> Result(Bool, TensorError) {
  tensor.try_all(t)
}

/// Count non-zero values in a tensor.
pub fn count_nonzero(t: Tensor) -> Int {
  tensor.count_nonzero(t)
}

/// Count non-zero values in a tensor, preserving materialization failures.
pub fn try_count_nonzero(t: Tensor) -> Result(Int, TensorError) {
  tensor.try_count_nonzero(t)
}

/// Does each axis slice contain any non-zero value?
pub fn any_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.any_axis(t, axis)
}

/// Does each axis slice contain any non-zero value?
pub fn try_any_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_any_axis(t, axis)
}

/// Does each axis slice contain any non-zero value, preserving the reduced dimension.
pub fn any_axis_keepdims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.any_axis_keepdims(t, axis)
}

/// Does each axis slice contain any non-zero value, preserving the reduced dimension.
pub fn try_any_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_any_axis_keepdims(t, axis)
}

/// Are all values in each axis slice non-zero?
pub fn all_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.all_axis(t, axis)
}

/// Are all values in each axis slice non-zero?
pub fn try_all_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.try_all_axis(t, axis)
}

/// Are all values in each axis slice non-zero, preserving the reduced dimension.
pub fn all_axis_keepdims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.all_axis_keepdims(t, axis)
}

/// Are all values in each axis slice non-zero, preserving the reduced dimension.
pub fn try_all_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_all_axis_keepdims(t, axis)
}

/// Count non-zero values along one axis.
pub fn count_nonzero_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.count_nonzero_axis(t, axis)
}

/// Count non-zero values along one axis.
pub fn try_count_nonzero_axis(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_count_nonzero_axis(t, axis)
}

/// Count non-zero values along one axis, preserving the reduced dimension.
pub fn count_nonzero_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.count_nonzero_axis_keepdims(t, axis)
}

/// Count non-zero values along one axis, preserving the reduced dimension.
pub fn try_count_nonzero_axis_keepdims(
  t: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.try_count_nonzero_axis_keepdims(t, axis)
}

// --- Strided (Zero-copy) ----------------------------------------------------

/// Convert to strided representation for O(1) element access
pub fn to_strided(t: Tensor) -> Tensor {
  tensor.to_strided(t)
}

/// Convert to strided representation, preserving materialization failures.
pub fn try_to_strided(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_to_strided(t)
}

/// Convert to contiguous tensor
pub fn to_contiguous(t: Tensor) -> Tensor {
  tensor.to_contiguous(t)
}

/// Convert to contiguous tensor, preserving materialization failures.
pub fn try_to_contiguous(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_to_contiguous(t)
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

/// Inspect native runtime acceleration availability.
pub fn capabilities() -> TensorCapabilities {
  let nif_loaded = ffi.is_nif_loaded()
  let zig_loaded = case nif_loaded {
    True -> ffi.zig_is_loaded()
    False -> False
  }
  let backend_info = case zig_loaded {
    True -> ffi.zig_backend_info()
    False -> "Zig NIF not loaded"
  }
  let backends = case nif_loaded {
    True -> detect_backends()
    False -> [tflops_mod.PureErlang]
  }

  TensorCapabilities(
    nif_loaded: nif_loaded,
    zig_loaded: zig_loaded,
    backend_info: backend_info,
    tflops_backends: backends,
    backend_capabilities: build_backend_capabilities(zig_loaded, backends),
  )
}

/// Inspect stable backend capability records.
pub fn backend_capabilities() -> List(BackendCapability) {
  let nif_loaded = ffi.is_nif_loaded()
  let zig_loaded = case nif_loaded {
    True -> ffi.zig_is_loaded()
    False -> False
  }
  let backends = case nif_loaded {
    True -> detect_backends()
    False -> [tflops_mod.PureErlang]
  }

  build_backend_capabilities(zig_loaded, backends)
}

/// Plan which backend should handle an operation on this VM.
pub fn plan_backend(operation: TensorOperation) -> TensorBackendPlan {
  let caps = capabilities()
  let available =
    caps.backend_capabilities
    |> list.filter(fn(capability) { capability.available })
    |> list.map(fn(capability) { capability.backend })

  case operation {
    OperationMatmul(m, n, k) ->
      plan_matmul(operation, m, n, k, available, caps.nif_loaded)
    OperationElementwise ->
      plan_first_available(
        operation,
        available,
        [BackendZigSimd, BackendMkl, BackendPureGleam],
        "Element-wise ops prefer SIMD, then native CPU, then pure Gleam.",
        "Backend does not support stable element-wise dispatch.",
      )
    OperationBroadcast ->
      plan_first_available(
        operation,
        available,
        [BackendZigSimd, BackendPureGleam],
        "Broadcasting preserves views and only needs native compute when materialized.",
        "Backend does not support stable broadcast dispatch.",
      )
    OperationReduction ->
      plan_first_available(
        operation,
        available,
        [BackendZigSimd, BackendMkl, BackendPureGleam],
        "Reductions prefer SIMD/native CPU and fall back to pure Gleam.",
        "Backend does not support stable reduction dispatch.",
      )
    OperationSoftmax ->
      plan_first_available(
        operation,
        available,
        [BackendPureGleam],
        "Softmax currently uses the stable Gleam implementation.",
        "Softmax currently only has stable pure Gleam dispatch.",
      )
  }
}

fn build_backend_capabilities(
  zig_loaded: Bool,
  backends: List(TflopsBackend),
) -> List(BackendCapability) {
  [
    BackendCapability(
      backend: BackendPureGleam,
      available: True,
      device: BackendBeamCpu,
      dtypes: [BackendFloat64],
      operations: [
        BackendElementwise,
        BackendBroadcast,
        BackendReduction,
        BackendSoftmax,
        BackendMatmul,
      ],
      reason: "Always available fallback.",
    ),
    BackendCapability(
      backend: BackendZigSimd,
      available: zig_loaded,
      device: BackendNativeCpu,
      dtypes: [BackendFloat64],
      operations: [
        BackendElementwise,
        BackendReduction,
        BackendMatmul,
      ],
      reason: "Portable SIMD NIF for CPU hot paths.",
    ),
    BackendCapability(
      backend: BackendMkl,
      available: zig_loaded,
      device: BackendNativeCpu,
      dtypes: [BackendFloat64, BackendFloat32],
      operations: [BackendMatmul],
      reason: "Native BLAS path exposed through the loaded Zig NIF.",
    ),
    BackendCapability(
      backend: BackendCudaFp32,
      available: list.contains(backends, tflops_mod.CudaFP32),
      device: BackendCuda,
      dtypes: [BackendFloat32],
      operations: [BackendMatmul],
      reason: "CUDA FP32/cuBLAS dense matrix multiplication.",
    ),
    BackendCapability(
      backend: BackendCudaFp16,
      available: list.contains(backends, tflops_mod.CudaFP16),
      device: BackendCuda,
      dtypes: [BackendFloat16],
      operations: [BackendMatmul],
      reason: "CUDA FP16 Tensor Core dense matrix multiplication.",
    ),
    BackendCapability(
      backend: BackendCudaInt8,
      available: list.contains(backends, tflops_mod.CudaINT8),
      device: BackendCuda,
      dtypes: [BackendInt8],
      operations: [BackendMatmul],
      reason: "CUDA INT8 IMMA Tensor Core matrix multiplication.",
    ),
    BackendCapability(
      backend: BackendCudaSparse,
      available: list.contains(backends, tflops_mod.CudaSparse),
      device: BackendCuda,
      dtypes: [BackendSparseFloat16],
      operations: [BackendMatmul],
      reason: "CUDA 2:4 sparse Tensor Core matrix multiplication.",
    ),
  ]
}

fn plan_matmul(
  operation: TensorOperation,
  m: Int,
  n: Int,
  k: Int,
  available: List(TensorBackend),
  nif_loaded: Bool,
) -> TensorBackendPlan {
  let tensor_core_aligned = m % 16 == 0 && n % 16 == 0 && k % 16 == 0
  let candidates = case tensor_core_aligned {
    True -> [
      BackendCudaSparse,
      BackendCudaFp16,
      BackendCudaInt8,
      BackendCudaFp32,
      BackendMkl,
      BackendZigSimd,
      BackendPureGleam,
    ]
    False -> [
      BackendCudaFp32,
      BackendMkl,
      BackendZigSimd,
      BackendPureGleam,
    ]
  }
  let reason = case nif_loaded {
    True ->
      case tensor_core_aligned {
        True -> "Matmul dimensions are Tensor Core aligned; CUDA is preferred."
        False ->
          "Matmul dimensions are not Tensor Core aligned; dense CUDA/CPU fallback is preferred."
      }
    False -> "Native NIF is not loaded; pure Gleam fallback is selected."
  }

  plan_first_available(
    operation,
    available,
    candidates,
    reason,
    "Backend is not part of the stable matmul dispatch path for this shape.",
  )
}

fn plan_first_available(
  operation: TensorOperation,
  available: List(TensorBackend),
  candidates: List(TensorBackend),
  reason: String,
  unsupported_reason: String,
) -> TensorBackendPlan {
  let selected = select_backend(available, candidates)

  TensorBackendPlan(
    operation: operation,
    selected: selected,
    fallbacks: candidates,
    rejected: backend_rejections(
      operation,
      selected,
      available,
      candidates,
      unsupported_reason,
    ),
    reason: reason,
  )
}

fn select_backend(
  available: List(TensorBackend),
  candidates: List(TensorBackend),
) -> TensorBackend {
  candidates
  |> list.find(fn(candidate) { list.contains(available, candidate) })
  |> result.unwrap(BackendPureGleam)
}

fn backend_rejections(
  operation: TensorOperation,
  selected: TensorBackend,
  available: List(TensorBackend),
  candidates: List(TensorBackend),
  unsupported_reason: String,
) -> List(BackendRejection) {
  all_tensor_backends()
  |> list.filter(fn(backend) { backend != selected })
  |> list.map(fn(backend) {
    BackendRejection(
      backend: backend,
      reason: rejection_reason(
        operation,
        backend,
        available,
        candidates,
        unsupported_reason,
      ),
    )
  })
}

fn rejection_reason(
  operation: TensorOperation,
  backend: TensorBackend,
  available: List(TensorBackend),
  candidates: List(TensorBackend),
  unsupported_reason: String,
) -> String {
  case list.contains(candidates, backend), list.contains(available, backend) {
    False, _ ->
      operation_specific_rejection(operation, backend, unsupported_reason)
    True, False -> "Backend is not available in this VM."
    True, True -> "A higher-priority backend was selected."
  }
}

fn operation_specific_rejection(
  operation: TensorOperation,
  backend: TensorBackend,
  fallback_reason: String,
) -> String {
  case operation, backend {
    OperationMatmul(_, _, _), BackendCudaSparse ->
      "Sparse Tensor Core dispatch requires an explicit sparse tensor."
    OperationMatmul(_, _, _), BackendCudaInt8 ->
      "INT8 Tensor Core dispatch requires explicit quantized tensors."
    OperationMatmul(m, n, k), BackendCudaFp16 ->
      case m % 16 == 0 && n % 16 == 0 && k % 16 == 0 {
        True -> fallback_reason
        False -> "FP16 Tensor Core matmul requires dimensions aligned to 16."
      }
    _, _ -> fallback_reason
  }
}

fn all_tensor_backends() -> List(TensorBackend) {
  [
    BackendCudaSparse,
    BackendCudaFp16,
    BackendCudaInt8,
    BackendCudaFp32,
    BackendMkl,
    BackendZigSimd,
    BackendPureGleam,
  ]
}

fn backend_is_available(
  backend: TensorBackend,
  capabilities: List(BackendCapability),
) -> Bool {
  capabilities
  |> list.any(fn(capability) {
    capability.backend == backend && capability.available
  })
}

fn run_matmul_backends(
  a: Tensor,
  b: Tensor,
  m: Int,
  n: Int,
  k: Int,
  backends: List(TensorBackend),
) -> Result(Tensor, TensorError) {
  case backends {
    [] -> tensor.matmul(a, b)
    [backend, ..rest] ->
      case matmul_with_backend(backend, a, b, m, n, k) {
        Ok(result) -> Ok(result)
        Error(_) -> run_matmul_backends(a, b, m, n, k, rest)
      }
  }
}

fn matmul_with_backend(
  backend: TensorBackend,
  a: Tensor,
  b: Tensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(Tensor, TensorError) {
  case backend {
    BackendCudaSparse ->
      Error(DimensionError(
        "Sparse Tensor Core dispatch requires an explicit sparse tensor.",
      ))
    BackendCudaInt8 ->
      Error(DimensionError(
        "INT8 Tensor Core dispatch requires explicit quantized tensors.",
      ))
    BackendCudaFp16 -> matmul_cuda_fp16(a, b)
    BackendCudaFp32 -> matmul_cuda_fp32(a, b)
    BackendMkl -> matmul_native_cpu(a, b, m, n, k)
    BackendZigSimd -> matmul_native_cpu(a, b, m, n, k)
    BackendPureGleam -> tensor.matmul(a, b)
  }
}

fn matmul_cuda_fp16(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use a_gpu <- result.try(cuda.to_rtx4090_fp16(a))
  use b_gpu <- result.try(cuda.to_rtx4090_fp16(b))
  use out <- result.try(cuda.matmul_accelerated(a_gpu, b_gpu))
  cuda.to_cpu_tensor(out)
}

fn matmul_cuda_fp32(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use a_gpu <- result.try(cuda.to_rtx4090_fp32(a))
  use b_gpu <- result.try(cuda.to_rtx4090_fp32(b))
  use out <- result.try(cuda.matmul_accelerated(a_gpu, b_gpu))
  cuda.to_cpu_tensor(out)
}

fn matmul_native_cpu(
  a: Tensor,
  b: Tensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(Tensor, TensorError) {
  use a_native <- result.try(tensor.native_from_list(tensor.to_list(a), [m, k]))
  use b_native <- result.try(tensor.native_from_list(tensor.to_list(b), [k, n]))
  tensor.matmul(a_native, b_native)
}
