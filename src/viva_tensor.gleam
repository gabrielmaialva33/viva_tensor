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

import gleam/dict.{type Dict}
import gleam/list
import gleam/option.{type Option}
import gleam/result
import viva_tensor/backend/capability as backend_capability
import viva_tensor/backend/dispatch as backend_dispatch
import viva_tensor/core/error.{DimensionError}
import viva_tensor/core/ffi
import viva_tensor/core/format as tensor_format
import viva_tensor/core/linalg
import viva_tensor/data/dataloader
import viva_tensor/io/hf_loader as hf_loader_io
import viva_tensor/io/onnx as onnx_io
import viva_tensor/io/safetensors as safetensors_io
import viva_tensor/layout as tensor_layout
import viva_tensor/metrics/classification as metrics_classification
import viva_tensor/metrics/regression as metrics_regression
import viva_tensor/native/cuda
import viva_tensor/native/tflops as tflops_mod
import viva_tensor/nn/activations as nn_activations
import viva_tensor/nn/attention as nn_attention
import viva_tensor/nn/backward as nn_backward
import viva_tensor/nn/conv as nn_conv
import viva_tensor/nn/cv as nn_cv
import viva_tensor/nn/embedding as nn_embedding
import viva_tensor/nn/init as nn_init
import viva_tensor/nn/losses as nn_losses
import viva_tensor/nn/norm as nn_norm
import viva_tensor/nn/optim as nn_optim
import viva_tensor/nn/pool as nn_pool
import viva_tensor/nn/rnn as nn_rnn
import viva_tensor/nn/scheduler as nn_scheduler
import viva_tensor/nn/transformer as nn_transformer
import viva_tensor/quant/hadamard as quant_hadamard
import viva_tensor/quant/layout as quant_layout
import viva_tensor/tensor
import viva_tensor/text/tokenizer as text_tokenizer
import viva_tensor/vision/transforms as vision_transforms

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

/// Hardware generation used by accelerator profile discovery.
pub type HardwareGeneration =
  backend_capability.HardwareGeneration

/// Hardware feature used by accelerator profile discovery.
pub type HardwareFeature =
  backend_capability.HardwareFeature

/// Hardware target profile for current and future accelerator dispatch.
pub type HardwareProfile =
  backend_capability.HardwareProfile

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

/// Quantized storage format metadata.
pub type QuantFormat =
  quant_layout.QuantFormat

/// Quantization scale sharing granularity.
pub type ScaleGranularity =
  quant_layout.ScaleGranularity

/// Accumulator format for quantized kernels.
pub type AccumulatorFormat =
  quant_layout.AccumulatorFormat

/// Quantized tensor layout metadata.
pub type QuantLayout =
  quant_layout.QuantLayout

/// Reversible Hadamard preprocessing result for low-bit quantization.
pub type HadamardPreprocess =
  quant_hadamard.HadamardPreprocess

/// Configuration for two-dimensional convolution operations.
pub type Conv2dConfig =
  tensor.Conv2dConfig

/// Layer normalization layer (normalizes along the last dimension).
pub type LayerNorm =
  nn_norm.LayerNorm

/// Root-mean-square normalization layer.
pub type RmsNorm =
  nn_norm.RmsNorm

/// 1D batch normalization layer.
pub type BatchNorm1d =
  nn_norm.BatchNorm1d

/// 2D batch normalization layer. Normalizes over `[B, H, W]` per channel `C`.
pub type BatchNorm2d =
  nn_cv.BatchNorm2d

/// Group normalization layer.
pub type GroupNorm =
  nn_norm.GroupNorm

/// Config for `max_unpool_2d_forward` — inverse of `max_pool_2d_with_indices`.
pub type MaxUnpool2dConfig =
  nn_cv.MaxUnpool2dConfig

/// Config for `roi_align`.
pub type RoiAlignConfig =
  nn_cv.RoiAlignConfig

/// A labeled training example: an input tensor paired with a target tensor.
pub type Sample =
  dataloader.Sample

/// In-memory dataset of `Sample`s.
pub type Dataset =
  dataloader.Dataset

/// Stacked batch produced by a `DataLoader`.
pub type Batch =
  dataloader.Batch

/// Iterator-style data loader.
pub type DataLoader =
  dataloader.DataLoader

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

/// Einstein summation. See `viva_tensor/tensor.einsum` for the full spec.
pub fn einsum(
  equation: String,
  operands: List(Tensor),
) -> Result(Tensor, TensorError) {
  tensor.einsum(equation, operands)
}

// --- Linear Algebra ---------------------------------------------------------
// Pure-Gleam linear algebra primitives. See `viva_tensor/core/linalg`.

/// Solve `A x = b` for a square `A` using Gaussian elimination with partial
/// pivoting. `b` may be 1D or 2D.
pub fn solve(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  linalg.solve(a, b)
}

/// Matrix inverse via `solve(a, identity)`. Errors when `a` is singular.
pub fn inv(a: Tensor) -> Result(Tensor, TensorError) {
  linalg.inv(a)
}

/// Determinant via LU decomposition. Returns 0.0 for singular matrices.
pub fn det(a: Tensor) -> Result(Float, TensorError) {
  linalg.det(a)
}

/// LU decomposition with partial pivoting. Returns `#(L, U, perm)`.
pub fn lu(a: Tensor) -> Result(#(Tensor, Tensor, List(Int)), TensorError) {
  linalg.lu(a)
}

/// Cholesky decomposition for symmetric positive-definite matrices.
/// Returns lower-triangular `L` with `A = L @ L^T`.
pub fn cholesky(a: Tensor) -> Result(Tensor, TensorError) {
  linalg.cholesky(a)
}

/// QR decomposition via classical Gram-Schmidt. Returns `#(Q, R)`.
pub fn qr(a: Tensor) -> Result(#(Tensor, Tensor), TensorError) {
  linalg.qr(a)
}

/// SVD stub (not implemented in v1).
pub fn svd(a: Tensor) -> Result(#(Tensor, Tensor, Tensor), TensorError) {
  linalg.svd(a)
}

/// Eigendecomposition stub (not implemented in v1).
pub fn eig(a: Tensor) -> Result(#(Tensor, Tensor), TensorError) {
  linalg.eig(a)
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

/// Take flattened elements by explicit indices (legacy: ignores tensor shape).
pub fn take_flat(t: Tensor, indices: List(Int)) -> Tensor {
  tensor.take_flat(t, indices)
}

/// Take flattened elements by explicit indices, preserving index errors.
pub fn try_take_flat(
  t: Tensor,
  indices: List(Int),
) -> Result(Tensor, TensorError) {
  tensor.try_take_flat(t, indices)
}

/// Take flattened elements by explicit indices, preserving index errors.
pub fn try_take(t: Tensor, indices: List(Int)) -> Result(Tensor, TensorError) {
  tensor.try_take(t, indices)
}

/// Gather slices along `axis` at each of the given indices (NumPy-style `take`).
pub fn take(
  t: Tensor,
  indices: List(Int),
  axis: Int,
) -> Result(Tensor, TensorError) {
  tensor.take(t, indices, axis)
}

/// Convenience wrapper around `take` for 1D integer-valued index tensors.
pub fn gather(t: Tensor, indices: Tensor) -> Result(Tensor, TensorError) {
  tensor.gather(t, indices)
}

/// Select elements of `t` where the same-shaped `mask` tensor is non-zero.
pub fn mask_select(t: Tensor, mask: Tensor) -> Result(Tensor, TensorError) {
  tensor.mask_select(t, mask)
}

/// Return flattened indices for non-zero values, represented as floats (legacy).
pub fn nonzero_flat(t: Tensor) -> Tensor {
  tensor.nonzero_flat(t)
}

/// Return flattened indices for non-zero values, represented as floats (legacy).
pub fn try_nonzero_flat(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_nonzero_flat(t)
}

/// Return flattened indices for non-zero values, preserving materialization failures.
pub fn try_nonzero(t: Tensor) -> Result(Tensor, TensorError) {
  tensor.try_nonzero(t)
}

/// Return multi-dimensional indices of non-zero elements of `t` (NumPy `nonzero`).
pub fn nonzero(t: Tensor) -> Result(List(List(Int)), TensorError) {
  tensor.nonzero(t)
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

// --- Embeddings & Positional Encoding --------------------------------------

/// Learnable embedding table: integer ids -> dense vectors.
pub type Embedding =
  nn_embedding.Embedding

/// Learnable positional encoding table.
pub type LearnedPositionalEncoding =
  nn_embedding.LearnedPositionalEncoding

/// Initialize an embedding table with zero weights.
pub fn embedding_init(num_embeddings: Int, embedding_dim: Int) -> Embedding {
  nn_embedding.embedding_init(num_embeddings, embedding_dim)
}

/// Initialize an embedding table with uniform random weights in
/// `[-1/sqrt(embedding_dim), 1/sqrt(embedding_dim)]`.
pub fn embedding_init_uniform(
  num_embeddings: Int,
  embedding_dim: Int,
) -> Embedding {
  nn_embedding.embedding_init_uniform(num_embeddings, embedding_dim)
}

/// Forward pass: gather rows of `weight` by integer indices.
pub fn embedding_forward(
  layer: Embedding,
  indices: Tensor,
) -> Result(Tensor, TensorError) {
  nn_embedding.embedding_forward(layer, indices)
}

/// Sinusoidal positional encoding ("Attention Is All You Need").
pub fn sinusoidal_encoding(
  max_len: Int,
  embedding_dim: Int,
) -> Result(Tensor, TensorError) {
  nn_embedding.sinusoidal_encoding(max_len, embedding_dim)
}

/// Initialize a learned positional encoding table.
pub fn learned_positional_init(
  max_len: Int,
  embedding_dim: Int,
) -> LearnedPositionalEncoding {
  nn_embedding.learned_positional_init(max_len, embedding_dim)
}

/// Look up positions `0..len-1` from a learned positional encoding.
pub fn learned_positional_forward(
  layer: LearnedPositionalEncoding,
  len: Int,
) -> Result(Tensor, TensorError) {
  nn_embedding.learned_positional_forward(layer, len)
}

/// Apply Rotary Positional Embedding (RoPE) to a `[seq_len, dim]` tensor.
pub fn rope(input: Tensor, base: Float) -> Result(Tensor, TensorError) {
  nn_embedding.rope(input, base)
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

// --- Pretty Printing --------------------------------------------------------

/// Options controlling tensor pretty-printing (precision, threshold,
/// edgeitems, linewidth, scientific notation, etc.).
pub type PrintOptions =
  tensor_format.PrintOptions

/// Scientific-notation mode used by `PrintOptions`.
pub type SciMode =
  tensor_format.SciMode

/// Sign mode used by `PrintOptions`.
pub type SignMode =
  tensor_format.SignMode

/// Default print options. Matches a sensible NumPy/PyTorch baseline:
/// `precision=4, threshold=1000, edgeitems=3, linewidth=80`.
pub fn default_print_options() -> PrintOptions {
  tensor_format.default_print_options()
}

/// Render a tensor as a pretty multi-line string with column alignment
/// and elision for large tensors.
pub fn to_string(t: Tensor) -> String {
  tensor_format.to_string(t)
}

/// Render a tensor with caller-supplied print options.
pub fn to_string_with(t: Tensor, opts: PrintOptions) -> String {
  tensor_format.to_string_with(t, opts)
}

/// Alias for `to_string` — matches the NumPy/PyTorch `inspect` /
/// `__repr__` convention.
pub fn inspect(t: Tensor) -> String {
  tensor_format.inspect(t)
}

/// Render an accelerated tensor (CudaFp16/CudaFp32/Cpu) as a pretty
/// string. Large CUDA tensors above the threshold render as
/// header-only to avoid surprise H2D copies.
pub fn accelerated_to_string(t: AcceleratedTensor) -> String {
  tensor_format.accelerated_to_string(t)
}

/// Render an accelerated tensor with caller-supplied print options.
pub fn accelerated_to_string_with(
  t: AcceleratedTensor,
  opts: PrintOptions,
) -> String {
  tensor_format.accelerated_to_string_with(t, opts)
}

// --- SafeTensors I/O --------------------------------------------------------

/// Read a SafeTensors file into a `Dict(String, Tensor)`.
///
/// Supports `F32` and `F64` payloads. See `viva_tensor/io/safetensors.read`.
///
/// ## Example
///
/// ```gleam
/// import gleam/dict
/// import viva_tensor as t
///
/// let assert Ok(weights) = t.safetensors_read("./model.safetensors")
/// let _ = dict.get(weights, "encoder.weight")
/// ```
pub fn safetensors_read(
  path: String,
) -> Result(Dict(String, Tensor), TensorError) {
  safetensors_io.read(path)
}

/// Write a `Dict(String, Tensor)` to disk in SafeTensors format (F64 payload).
///
/// ## Example
///
/// ```gleam
/// import gleam/dict
/// import viva_tensor as t
///
/// let weights = dict.from_list([#("w", t.ones([2, 2]))])
/// let assert Ok(Nil) = t.safetensors_write("./out.safetensors", weights)
/// ```
pub fn safetensors_write(
  path: String,
  tensors: Dict(String, Tensor),
) -> Result(Nil, TensorError) {
  safetensors_io.write(path, tensors)
}

// --- ONNX I/O ---------------------------------------------------------------

/// Re-export of `viva_tensor/io/onnx.OnnxGraph` for the public facade.
pub type OnnxGraph =
  onnx_io.OnnxGraph

/// Re-export of `viva_tensor/io/onnx.OnnxNode`.
pub type OnnxNode =
  onnx_io.OnnxNode

/// Re-export of `viva_tensor/io/onnx.OnnxAttribute`.
pub type OnnxAttribute =
  onnx_io.OnnxAttribute

/// Re-export of `viva_tensor/io/onnx.OnnxError`.
pub type OnnxError =
  onnx_io.OnnxError

/// Parse a JSON-encoded ONNX graph.
///
/// Supported op set (v1): `Add`, `Sub`, `Mul`, `MatMul`, `Gemm`, `Relu`,
/// `Sigmoid`, `Tanh`, `Gelu`, `Softmax`, `Transpose`, `Reshape`, `Constant`,
/// `LayerNormalization`. See `viva_tensor/io/onnx.parse_graph`.
pub fn onnx_parse_graph(json_str: String) -> Result(OnnxGraph, OnnxError) {
  onnx_io.parse_graph(json_str)
}

/// Execute a parsed ONNX graph against a dict of named input tensors.
///
/// Returns the full execution table — pick the named graph outputs from it.
/// See `viva_tensor/io/onnx.run_graph` for the v1 supported op set.
pub fn onnx_run_graph(
  graph: OnnxGraph,
  feeds: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), OnnxError) {
  onnx_io.run_graph(graph, feeds)
}

/// Return the list of ONNX op_types supported by `onnx_run_graph` in v1.
pub fn onnx_supported_ops() -> List(String) {
  onnx_io.supported_ops()
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

// --- NN Normalization Layers -----------------------------------------------

/// Initialize a `LayerNorm` with default `eps = 1.0e-5`.
pub fn layer_norm_init(num_features: Int) -> LayerNorm {
  nn_norm.layer_norm_init(num_features)
}

/// Initialize a `LayerNorm` with custom `eps`.
pub fn layer_norm_init_with_eps(num_features: Int, eps: Float) -> LayerNorm {
  nn_norm.layer_norm_init_with_eps(num_features, eps)
}

/// Forward pass for `LayerNorm` — normalizes along the last dimension.
pub fn layer_norm_forward(
  layer: LayerNorm,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_norm.layer_norm_forward(layer, input)
}

/// Initialize an `RmsNorm` with default `eps = 1.0e-6`.
pub fn rms_norm_init(num_features: Int) -> RmsNorm {
  nn_norm.rms_norm_init(num_features)
}

/// Initialize an `RmsNorm` with custom `eps`.
pub fn rms_norm_init_with_eps(num_features: Int, eps: Float) -> RmsNorm {
  nn_norm.rms_norm_init_with_eps(num_features, eps)
}

/// Forward pass for `RmsNorm` — RMS normalize along the last dimension.
pub fn rms_norm_forward(
  layer: RmsNorm,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_norm.rms_norm_forward(layer, input)
}

/// Initialize a `BatchNorm1d` with default `momentum = 0.1`, `eps = 1.0e-5`.
pub fn batch_norm_1d_init(num_features: Int) -> BatchNorm1d {
  nn_norm.batch_norm_1d_init(num_features)
}

/// Forward pass for `BatchNorm1d`.
///
/// In training mode, updates running stats via EMA and returns the new layer
/// alongside the normalized output. In eval mode, uses running stats and
/// returns the layer unchanged.
pub fn batch_norm_1d_forward(
  layer: BatchNorm1d,
  input: Tensor,
  training: Bool,
) -> Result(#(BatchNorm1d, Tensor), TensorError) {
  nn_norm.batch_norm_1d_forward(layer, input, training)
}

/// Initialize a `GroupNorm` with `num_groups` groups over `num_channels` channels.
pub fn group_norm_init(num_groups: Int, num_channels: Int) -> GroupNorm {
  nn_norm.group_norm_init(num_groups, num_channels)
}

/// Forward pass for `GroupNorm` — supports `[batch, channels]` and
/// `[batch, channels, spatial]` inputs.
pub fn group_norm_forward(
  layer: GroupNorm,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_norm.group_norm_forward(layer, input)
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

// --- Extra Conv Variants (Conv1d / Conv3d / ConvTranspose2d) ---------------

/// Configuration for a 1D convolution layer.
pub type Conv1dConfig =
  nn_conv.Conv1dConfig

/// Configuration for a 3D convolution layer.
pub type Conv3dConfig =
  nn_conv.Conv3dConfig

/// Configuration for a 2D transposed convolution.
pub type ConvTranspose2dConfig =
  nn_conv.ConvTranspose2dConfig

/// Initialize a Conv1d layer with zero weights and bias.
pub fn conv1d_init(
  in_channels in_channels: Int,
  out_channels out_channels: Int,
  kernel_size kernel_size: Int,
  stride stride: Int,
  padding padding: Int,
) -> Conv1dConfig {
  nn_conv.conv1d_init(
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_size: kernel_size,
    stride: stride,
    padding: padding,
  )
}

/// 1D convolution forward pass. Output length =
/// `(L_in + 2*padding - kernel) / stride + 1`.
pub fn conv1d_forward(
  config: Conv1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_conv.conv1d_forward(config, input)
}

/// Initialize a Conv3d layer with zero weights and bias.
pub fn conv3d_init(
  in_channels in_channels: Int,
  out_channels out_channels: Int,
  kernel_size kernel_size: #(Int, Int, Int),
  stride stride: #(Int, Int, Int),
  padding padding: #(Int, Int, Int),
) -> Conv3dConfig {
  nn_conv.conv3d_init(
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_size: kernel_size,
    stride: stride,
    padding: padding,
  )
}

/// 3D convolution forward pass. Output dim =
/// `(In + 2*pad - kernel) / stride + 1` per spatial axis.
pub fn conv3d_forward(
  config: Conv3dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_conv.conv3d_forward(config, input)
}

/// Initialize a ConvTranspose2d layer with zero weights and bias.
pub fn conv_transpose_2d_init(
  in_channels in_channels: Int,
  out_channels out_channels: Int,
  kernel_size kernel_size: #(Int, Int),
  stride stride: #(Int, Int),
  padding padding: #(Int, Int),
  output_padding output_padding: #(Int, Int),
) -> ConvTranspose2dConfig {
  nn_conv.conv_transpose_2d_init(
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_size: kernel_size,
    stride: stride,
    padding: padding,
    output_padding: output_padding,
  )
}

/// 2D transposed convolution (deconv) forward pass. Output dim =
/// `(In - 1) * stride - 2*padding + (kernel - 1) + output_padding + 1`.
pub fn conv_transpose_2d_forward(
  config: ConvTranspose2dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_conv.conv_transpose_2d_forward(config, input)
}

/// Global average pooling
pub fn global_avg_pool2d(input: Tensor) -> Result(Tensor, TensorError) {
  tensor.global_avg_pool2d(input)
}

// --- Pool / Regularization layers ------------------------------------------

/// Inverted-dropout layer (drop probability `p`).
pub type Dropout =
  nn_pool.Dropout

/// 1D max-pool configuration.
pub type MaxPool1dConfig =
  nn_pool.MaxPool1dConfig

/// 1D average-pool configuration.
pub type AvgPool1dConfig =
  nn_pool.AvgPool1dConfig

/// 2D adaptive average-pool configuration.
pub type AdaptiveAvgPool2dConfig =
  nn_pool.AdaptiveAvgPool2dConfig

/// 1D adaptive average-pool configuration.
pub type AdaptiveAvgPool1dConfig =
  nn_pool.AdaptiveAvgPool1dConfig

/// Upsample configuration (mode + integer scale factor).
pub type UpsampleConfig =
  nn_pool.UpsampleConfig

/// Upsample mode selector.
pub type UpsampleMode =
  nn_pool.UpsampleMode

/// Initialize a `Dropout` layer with drop probability `p`. Output shape: same
/// as input.
pub fn dropout_init(p: Float) -> Dropout {
  nn_pool.dropout_init(p)
}

/// Forward pass for inverted dropout. Output shape: same as input.
/// Passthrough when `training = False` or `p == 0.0`. When `p == 1.0` every
/// element is zeroed.
pub fn dropout_forward(
  layer: Dropout,
  input: Tensor,
  training: Bool,
) -> Tensor {
  nn_pool.dropout_forward(layer, input, training)
}

/// 1D max pooling. Input `[batch, channels, length]`, output
/// `[batch, channels, (length + 2*padding - kernel_size) / stride + 1]`.
pub fn max_pool_1d_forward(
  config: MaxPool1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_pool.max_pool_1d_forward(config, input)
}

/// 1D average pooling. Input `[batch, channels, length]`, output
/// `[batch, channels, (length + 2*padding - kernel_size) / stride + 1]`.
pub fn avg_pool_1d_forward(
  config: AvgPool1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_pool.avg_pool_1d_forward(config, input)
}

/// 2D adaptive average pooling. Input `[batch, channels, H, W]`, output
/// `[batch, channels, output_h, output_w]`.
pub fn adaptive_avg_pool_2d_forward(
  config: AdaptiveAvgPool2dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_pool.adaptive_avg_pool_2d_forward(config, input)
}

/// 1D adaptive average pooling. Input `[batch, channels, length]`, output
/// `[batch, channels, output_size]`.
pub fn adaptive_avg_pool_1d_forward(
  config: AdaptiveAvgPool1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_pool.adaptive_avg_pool_1d_forward(config, input)
}

/// 2D upsampling (nearest or bilinear). Input `[batch, channels, H, W]`,
/// output `[batch, channels, H * scale_factor, W * scale_factor]`.
pub fn upsample_forward(
  config: UpsampleConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_pool.upsample_forward(config, input)
}

// --- Computer-vision ops ---------------------------------------------------

/// Run a 2D max-pool returning both pooled values and the flat argmax index
/// per output cell. Input `[N, C, H, W]`; outputs are both
/// `[N, C, H_out, W_out]`. Indices are stored as `Float` (truncated by the
/// unpool consumer); fully-padded windows get `-1.0`.
pub fn max_pool_2d_with_indices(
  input: Tensor,
  kernel_size: Int,
  stride: Int,
  padding: Int,
) -> Result(#(Tensor, Tensor), TensorError) {
  nn_cv.max_pool_2d_with_indices(input, kernel_size, stride, padding)
}

/// Inverse of `max_pool_2d_with_indices`. Scatters pooled values back at the
/// stored indices, zeros elsewhere. Input `[N, C, H_out, W_out]`, indices
/// `[N, C, H_out, W_out]`, output `[N, C, H_in, W_in]` where `(H_in, W_in)`
/// comes from `output_size`.
pub fn max_unpool_2d_forward(
  config: MaxUnpool2dConfig,
  input: Tensor,
  indices: Tensor,
  output_size: #(Int, Int),
) -> Result(Tensor, TensorError) {
  nn_cv.max_unpool_2d_forward(config, input, indices, output_size)
}

/// Greedy Non-Maximum Suppression. `boxes` `[N, 4]` (rows `[x1, y1, x2, y2]`),
/// `scores` `[N]`. Returns the indices of kept boxes, sorted by descending
/// score.
pub fn nms(
  boxes: Tensor,
  scores: Tensor,
  iou_threshold: Float,
) -> Result(List(Int), TensorError) {
  nn_cv.nms(boxes, scores, iou_threshold)
}

/// Bilinear ROIAlign. `features` `[N, C, H, W]`, `rois` `[K, 5]` with rows
/// `[batch_index, x1, y1, x2, y2]`. Output `[K, C, output_h, output_w]`.
pub fn roi_align(
  config: RoiAlignConfig,
  features: Tensor,
  rois: Tensor,
) -> Result(Tensor, TensorError) {
  nn_cv.roi_align(config, features, rois)
}

/// Batched 2-D matmul `[Ba, M, K] @ [Bb, K, N] -> [max(Ba, Bb), M, N]` with
/// broadcasting when either batch dim is `1`.
pub fn batched_matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  nn_cv.batched_matmul(a, b)
}

/// Initialize a `BatchNorm2d` with `scale = ones([C])`, `bias = zeros([C])`,
/// `running_mean = zeros([C])`, `running_var = ones([C])`, `momentum = 0.1`,
/// `eps = 1.0e-5`. `C = num_features`.
pub fn batch_norm_2d_init(num_features: Int) -> BatchNorm2d {
  nn_cv.batch_norm_2d_init(num_features)
}

/// Forward pass for `BatchNorm2d`. Input `[B, C, H, W]`, output same shape.
/// In training mode updates running stats via EMA; in eval mode uses them
/// directly. Returns the (possibly updated) layer plus the output.
pub fn batch_norm_2d_forward(
  layer: BatchNorm2d,
  input: Tensor,
  training: Bool,
) -> Result(#(BatchNorm2d, Tensor), TensorError) {
  nn_cv.batch_norm_2d_forward(layer, input, training)
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

/// Inspect hardware target profiles, including unavailable future targets.
pub fn hardware_profiles() -> List(HardwareProfile) {
  let caps = capabilities()
  backend_capability.hardware_profiles(caps.zig_loaded, caps.tflops_backends)
}

/// Describe a Rubin-ready NVFP4 block-scaled layout using 16-value micro-blocks.
pub fn nvfp4_block_scaled_layout(shape: List(Int)) -> QuantLayout {
  quant_layout.nvfp4_block_scaled(shape)
}

/// Describe an experimental progressive INT2 layout.
pub fn int2_progressive_layout(
  shape: List(Int),
  block_size: Int,
) -> Result(QuantLayout, TensorError) {
  quant_layout.int2_progressive(shape, block_size)
}

/// Describe an experimental progressive INT3 layout.
pub fn int3_progressive_layout(
  shape: List(Int),
  block_size: Int,
) -> Result(QuantLayout, TensorError) {
  quant_layout.int3_progressive(shape, block_size)
}

/// Estimate payload bytes for a quantized layout.
pub fn quant_layout_memory_bytes(layout: QuantLayout) -> Int {
  quant_layout.memory_bytes(layout)
}

/// Estimate compression ratio versus a baseline element width.
pub fn quant_layout_compression_ratio_against(
  layout: QuantLayout,
  baseline_bits_per_value: Int,
) -> Float {
  quant_layout.compression_ratio_against(layout, baseline_bits_per_value)
}

/// Check whether a layout matches Rubin-style native micro-block assumptions.
pub fn quant_layout_is_rubin_native_candidate(layout: QuantLayout) -> Bool {
  quant_layout.is_rubin_native_candidate(layout)
}

/// Apply randomized normalized Hadamard preprocessing to a vector tensor.
pub fn try_hadamard_preprocess(
  input: Tensor,
  seed: Int,
) -> Result(HadamardPreprocess, TensorError) {
  quant_hadamard.try_preprocess(input, seed)
}

/// Invert a previously applied Hadamard preprocessing plan.
pub fn try_inverse_hadamard_preprocess(
  preprocessed: HadamardPreprocess,
) -> Result(Tensor, TensorError) {
  quant_hadamard.inverse(preprocessed)
}

/// Apply a normalized Walsh-Hadamard transform to power-of-two vector data.
pub fn try_normalized_walsh_hadamard(
  values: List(Float),
) -> Result(List(Float), TensorError) {
  quant_hadamard.try_normalized_walsh_hadamard(values)
}

/// Plan which backend should handle an operation on this VM.
pub fn plan_backend(operation: TensorOperation) -> TensorBackendPlan {
  let caps = capabilities()
  let available =
    caps.backend_capabilities
    |> list.map(fn(capability) {
      backend_dispatch.Capability(
        backend: capability.backend,
        available: capability.available,
        device: capability.device,
        dtypes: capability.dtypes,
        operations: capability.operations,
        reason: capability.reason,
      )
    })
    |> backend_dispatch.available_backends

  backend_dispatch.plan_backend(
    operation,
    operation_kind(operation),
    available,
    backend_set(),
    caps.nif_loaded,
  )
  |> to_tensor_backend_plan
}

fn backend_set() -> backend_dispatch.BackendSet(TensorBackend) {
  backend_dispatch.BackendSet(
    pure_gleam: BackendPureGleam,
    zig_simd: BackendZigSimd,
    mkl: BackendMkl,
    cuda_fp32: BackendCudaFp32,
    cuda_fp16: BackendCudaFp16,
    cuda_int8: BackendCudaInt8,
    cuda_sparse: BackendCudaSparse,
  )
}

fn operation_kind(
  operation: TensorOperation,
) -> backend_dispatch.OperationKind {
  case operation {
    OperationElementwise -> backend_dispatch.Elementwise
    OperationBroadcast -> backend_dispatch.Broadcast
    OperationReduction -> backend_dispatch.Reduction
    OperationSoftmax -> backend_dispatch.Softmax
    OperationMatmul(m, n, k) -> backend_dispatch.Matmul(m, n, k)
  }
}

fn to_tensor_backend_plan(
  plan: backend_dispatch.Plan(TensorOperation, TensorBackend),
) -> TensorBackendPlan {
  TensorBackendPlan(
    operation: plan.operation,
    selected: plan.selected,
    fallbacks: plan.fallbacks,
    rejected: list.map(plan.rejected, fn(rejection) {
      BackendRejection(backend: rejection.backend, reason: rejection.reason)
    }),
    reason: plan.reason,
  )
}

fn build_backend_capabilities(
  zig_loaded: Bool,
  backends: List(TflopsBackend),
) -> List(BackendCapability) {
  backend_dispatch.capabilities(
    backend_set(),
    BackendBeamCpu,
    BackendNativeCpu,
    BackendCuda,
    BackendFloat64,
    BackendFloat32,
    BackendFloat16,
    BackendInt8,
    BackendSparseFloat16,
    BackendElementwise,
    BackendBroadcast,
    BackendReduction,
    BackendSoftmax,
    BackendMatmul,
    zig_loaded,
    backends,
  )
  |> list.map(fn(capability) {
    BackendCapability(
      backend: capability.backend,
      available: capability.available,
      device: capability.device,
      dtypes: capability.dtypes,
      operations: capability.operations,
      reason: capability.reason,
    )
  })
}

fn backend_is_available(
  backend: TensorBackend,
  capabilities: List(BackendCapability),
) -> Bool {
  capabilities
  |> list.map(fn(capability) {
    backend_dispatch.Capability(
      backend: capability.backend,
      available: capability.available,
      device: capability.device,
      dtypes: capability.dtypes,
      operations: capability.operations,
      reason: capability.reason,
    )
  })
  |> fn(capabilities) { backend_dispatch.is_available(backend, capabilities) }
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

// --- Loss functions ---------------------------------------------------------
//
// Forward-pass-only re-exports of `viva_tensor/nn/losses`. Backward passes
// (autograd integration) are a follow-up — these functions return a Tensor,
// not a Variable.

/// Reduction strategy for loss functions. See `viva_tensor/nn/losses`.
pub type Reduction =
  nn_losses.Reduction

/// `ReductionNone` keeps the per-element loss tensor.
pub const reduction_none: Reduction = nn_losses.ReductionNone

/// `ReductionMean` returns the arithmetic mean as a 1-element tensor.
pub const reduction_mean: Reduction = nn_losses.ReductionMean

/// `ReductionSum` returns the sum as a 1-element tensor.
pub const reduction_sum: Reduction = nn_losses.ReductionSum

/// Mean Squared Error. See `viva_tensor/nn/losses.mse_loss`.
pub fn mse_loss(
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_losses.mse_loss(prediction, target, reduction)
}

/// L1 / Mean Absolute Error. See `viva_tensor/nn/losses.l1_loss`.
pub fn l1_loss(
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_losses.l1_loss(prediction, target, reduction)
}

/// Binary Cross-Entropy. See `viva_tensor/nn/losses.bce_loss`.
pub fn bce_loss(
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_losses.bce_loss(prediction, target, reduction)
}

/// Softmax cross-entropy with integer-valued class targets. See
/// `viva_tensor/nn/losses.cross_entropy_loss`.
pub fn cross_entropy_loss(
  logits: Tensor,
  targets: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_losses.cross_entropy_loss(logits, targets, reduction)
}

/// Huber loss (smooth L1). See `viva_tensor/nn/losses.huber_loss`.
pub fn huber_loss(
  prediction: Tensor,
  target: Tensor,
  delta: Float,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_losses.huber_loss(prediction, target, delta, reduction)
}

// --- Optimizers -------------------------------------------------------------
//
// Gradient-descent optimizer surface (no autograd integration yet — callers
// hand in `Param`/`GradPair` lists explicitly).

/// Optimizer family tag.
pub type OptimizerKind =
  nn_optim.OptimizerKind

/// A named parameter tensor passed to `step`.
pub type Param =
  nn_optim.Param

/// A gradient paired with the name of the parameter it belongs to.
pub type GradPair =
  nn_optim.GradPair

/// Per-parameter optimizer state (momentum/variance/etc.).
pub type ParamState =
  nn_optim.ParamState

/// Optimizer record carrying hyperparameters and per-parameter state.
pub type Optimizer =
  nn_optim.Optimizer

/// Vanilla stochastic gradient descent. See `viva_tensor/nn/optim`.
pub fn sgd(lr: Float) -> Optimizer {
  nn_optim.sgd(lr)
}

/// SGD with momentum. See `viva_tensor/nn/optim`.
pub fn sgd_momentum(lr: Float, momentum: Float) -> Optimizer {
  nn_optim.sgd_momentum(lr, momentum)
}

/// RMSprop. See `viva_tensor/nn/optim`.
pub fn rmsprop(lr: Float, alpha: Float, eps: Float) -> Optimizer {
  nn_optim.rmsprop(lr, alpha, eps)
}

/// Adam (Kingma & Ba, 2015). See `viva_tensor/nn/optim`.
pub fn adam(lr: Float) -> Optimizer {
  nn_optim.adam(lr)
}

/// AdamW (Loshchilov & Hutter, 2019). See `viva_tensor/nn/optim`.
pub fn adamw(lr: Float, weight_decay: Float) -> Optimizer {
  nn_optim.adamw(lr, weight_decay)
}

/// Apply one optimizer step. See `viva_tensor/nn/optim.step`.
pub fn step(
  opt: Optimizer,
  params: List(Param),
  grads: List(GradPair),
) -> Result(#(Optimizer, List(Param)), TensorError) {
  nn_optim.step(opt, params, grads)
}

/// Zero every gradient tensor, preserving shapes. See `viva_tensor/nn/optim.zero_grad`.
pub fn zero_grad(grads: List(GradPair)) -> List(GradPair) {
  nn_optim.zero_grad(grads)
}

// --- Activations ------------------------------------------------------------
//
// Forward-only neural-network activations. See `viva_tensor/nn/activations`
// for the implementations and formulas. All operate element-wise (or per
// `axis` slice for softmax variants) and never mutate the input tensor.

/// Sigmoid activation: `1 / (1 + exp(-x))`. Numerically stable for large
/// negative inputs via `exp(x) / (1 + exp(x))`.
pub fn sigmoid(t: Tensor) -> Tensor {
  nn_activations.sigmoid(t)
}

/// Hyperbolic tangent activation. Output range `(-1, 1)`.
pub fn tanh(t: Tensor) -> Tensor {
  nn_activations.tanh(t)
}

/// Rectified Linear Unit: `max(0, x)`.
pub fn relu(t: Tensor) -> Tensor {
  nn_activations.relu(t)
}

/// Leaky ReLU: `x` if `x > 0` else `negative_slope * x`.
pub fn leaky_relu(t: Tensor, negative_slope: Float) -> Tensor {
  nn_activations.leaky_relu(t, negative_slope)
}

/// Exponential Linear Unit: `x` if `x > 0` else `alpha * (exp(x) - 1)`.
pub fn elu(t: Tensor, alpha: Float) -> Tensor {
  nn_activations.elu(t, alpha)
}

/// Scaled ELU with the canonical SELU constants from Klambauer et al. (2017).
pub fn selu(t: Tensor) -> Tensor {
  nn_activations.selu(t)
}

/// Gaussian Error Linear Unit: `0.5 * x * (1 + erf(x / sqrt(2)))`.
/// Uses the exact `erf`-based formulation.
pub fn gelu(t: Tensor) -> Tensor {
  nn_activations.gelu(t)
}

/// Swish / SiLU: `x * sigmoid(x)`.
pub fn swish(t: Tensor) -> Tensor {
  nn_activations.swish(t)
}

/// Mish: `x * tanh(softplus(x))`.
pub fn mish(t: Tensor) -> Tensor {
  nn_activations.mish(t)
}

/// Softplus: `log(1 + exp(x))`, numerically stable.
pub fn softplus(t: Tensor) -> Tensor {
  nn_activations.softplus(t)
}

/// Softmax along `axis`: `exp(x - max) / sum(exp(x - max))`.
pub fn softmax(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  nn_activations.softmax(t, axis)
}

/// Log-softmax along `axis`: `x - max - log(sum(exp(x - max)))`.
pub fn log_softmax(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  nn_activations.log_softmax(t, axis)
}

/// HardSwish: `x * relu6(x + 3) / 6`.
pub fn hardswish(t: Tensor) -> Tensor {
  nn_activations.hardswish(t)
}

/// HardTanh: `clamp(x, min_val, max_val)`.
pub fn hardtanh(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  nn_activations.hardtanh(t, min_val, max_val)
}

// --- Backward (gradient) functions ------------------------------------------
//
// Standalone pure backwards for the v1 NN surface. They take `grad_out` plus
// whatever forward inputs/outputs are needed to compute the local
// Jacobian-vector product, and return gradients w.r.t. each differentiable
// input. Several activation/softmax backwards consume the **output** of the
// forward (sigmoid, tanh, softmax) — the future autograd `Tape` will save
// those buffers. See `viva_tensor/nn/backward` for full math.

/// Backward for `relu`. See `viva_tensor/nn/backward.relu_backward`.
pub fn relu_backward(
  grad_out: Tensor,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_backward.relu_backward(grad_out, input)
}

/// Backward for `sigmoid`. Takes the sigmoid **output**, not the original
/// input. See `viva_tensor/nn/backward.sigmoid_backward`.
pub fn sigmoid_backward(
  grad_out: Tensor,
  output: Tensor,
) -> Result(Tensor, TensorError) {
  nn_backward.sigmoid_backward(grad_out, output)
}

/// Backward for `tanh`. Takes the tanh **output**, not the original input.
/// See `viva_tensor/nn/backward.tanh_backward`.
pub fn tanh_backward(
  grad_out: Tensor,
  output: Tensor,
) -> Result(Tensor, TensorError) {
  nn_backward.tanh_backward(grad_out, output)
}

/// Backward for exact `gelu`. See `viva_tensor/nn/backward.gelu_backward`.
pub fn gelu_backward(
  grad_out: Tensor,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_backward.gelu_backward(grad_out, input)
}

/// Backward for `leaky_relu`. See `viva_tensor/nn/backward.leaky_relu_backward`.
pub fn leaky_relu_backward(
  grad_out: Tensor,
  input: Tensor,
  negative_slope: Float,
) -> Result(Tensor, TensorError) {
  nn_backward.leaky_relu_backward(grad_out, input, negative_slope)
}

/// Backward for `elu`. See `viva_tensor/nn/backward.elu_backward`.
pub fn elu_backward(
  grad_out: Tensor,
  input: Tensor,
  alpha: Float,
) -> Result(Tensor, TensorError) {
  nn_backward.elu_backward(grad_out, input, alpha)
}

/// Backward for `mse_loss`. Returns gradient w.r.t. `prediction` only.
/// See `viva_tensor/nn/backward.mse_loss_backward`.
pub fn mse_loss_backward(
  grad_out: Tensor,
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_backward.mse_loss_backward(grad_out, prediction, target, reduction)
}

/// Backward for `l1_loss`. Returns gradient w.r.t. `prediction` only.
/// See `viva_tensor/nn/backward.l1_loss_backward`.
pub fn l1_loss_backward(
  grad_out: Tensor,
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_backward.l1_loss_backward(grad_out, prediction, target, reduction)
}

/// Backward for `bce_loss`. Returns gradient w.r.t. `prediction` only.
/// See `viva_tensor/nn/backward.bce_loss_backward`.
pub fn bce_loss_backward(
  grad_out: Tensor,
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_backward.bce_loss_backward(grad_out, prediction, target, reduction)
}

/// Backward for `cross_entropy_loss`. Returns gradient w.r.t. `logits`.
/// See `viva_tensor/nn/backward.cross_entropy_loss_backward`.
pub fn cross_entropy_loss_backward(
  grad_out: Tensor,
  logits: Tensor,
  targets: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  nn_backward.cross_entropy_loss_backward(grad_out, logits, targets, reduction)
}

/// Backward for a linear layer `output = input @ weight`. Returns
/// `#(grad_input, grad_weight)`. See `viva_tensor/nn/backward.linear_backward`.
pub fn linear_backward(
  grad_out: Tensor,
  input: Tensor,
  weight: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  nn_backward.linear_backward(grad_out, input, weight)
}

/// Backward for `matmul`. Returns `#(grad_a, grad_b)`. Same math as
/// `linear_backward`, exposed for the user-facing matmul.
/// See `viva_tensor/nn/backward.matmul_backward`.
pub fn matmul_backward(
  grad_out: Tensor,
  a: Tensor,
  b: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  nn_backward.matmul_backward(grad_out, a, b)
}

/// Backward for `layer_norm` over the last dimension. Requires the `mean`
/// and `variance` saved from the forward pass.
/// See `viva_tensor/nn/backward.layer_norm_backward`.
pub fn layer_norm_backward(
  grad_out: Tensor,
  input: Tensor,
  scale: Tensor,
  mean: Tensor,
  variance: Tensor,
  eps: Float,
) -> Result(#(Tensor, Tensor, Tensor), TensorError) {
  nn_backward.layer_norm_backward(grad_out, input, scale, mean, variance, eps)
}

/// Backward for `rms_norm` over the last dimension. Requires the `rms`
/// saved from the forward pass. `eps` is retained for signature symmetry.
/// See `viva_tensor/nn/backward.rms_norm_backward`.
pub fn rms_norm_backward(
  grad_out: Tensor,
  input: Tensor,
  scale: Tensor,
  rms: Tensor,
  eps: Float,
) -> Result(#(Tensor, Tensor), TensorError) {
  nn_backward.rms_norm_backward(grad_out, input, scale, rms, eps)
}

/// Backward for `softmax` along `axis`. Takes the softmax **output** of the
/// forward pass. See `viva_tensor/nn/backward.softmax_backward`.
pub fn softmax_backward(
  grad_out: Tensor,
  output: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  nn_backward.softmax_backward(grad_out, output, axis)
}

// --- Attention --------------------------------------------------------------

/// Multi-Head Attention layer. See `viva_tensor/nn/attention` for details.
pub type MultiHeadAttention =
  nn_attention.MultiHeadAttention

/// `softmax((Q @ K^T) / sqrt(d_k)) @ V`. See nn/attention docs.
pub fn scaled_dot_product_attention(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  mask: Option(Tensor),
  is_causal: Bool,
) -> Result(Tensor, TensorError) {
  nn_attention.scaled_dot_product_attention(q, k, v, mask, is_causal)
}

/// Initialize a Multi-Head Attention module with zero weights.
pub fn multi_head_attention_init(
  num_heads: Int,
  embed_dim: Int,
  use_bias: Bool,
) -> Result(MultiHeadAttention, TensorError) {
  nn_attention.multi_head_attention_init(num_heads, embed_dim, use_bias)
}

/// Multi-Head Attention forward pass.
pub fn multi_head_attention_forward(
  mha: MultiHeadAttention,
  q: Tensor,
  k: Tensor,
  v: Tensor,
  is_causal: Bool,
) -> Result(Tensor, TensorError) {
  nn_attention.multi_head_attention_forward(mha, q, k, v, is_causal)
}

/// Lower-triangular `[seq_len, seq_len]` mask of `1.0`s used by causal SDPA.
pub fn causal_mask(seq_len: Int) -> Tensor {
  nn_attention.causal_mask(seq_len)
}

// --- Recurrent cells (re-exports from viva_tensor/nn/rnn) ------------------

/// Vanilla Elman RNN cell parameters.
pub type RnnCell =
  nn_rnn.RnnCell

/// GRU cell parameters (reset/update/new gates stacked in weight rows).
pub type GruCell =
  nn_rnn.GruCell

/// LSTM cell parameters (input/forget/cell/output gates stacked).
pub type LstmCell =
  nn_rnn.LstmCell

/// Build an Elman RNN cell with Xavier-initialized weights and zero biases.
pub fn rnn_cell_init(input_size: Int, hidden_size: Int) -> RnnCell {
  nn_rnn.rnn_cell_init(input_size, hidden_size)
}

/// Build a GRU cell with Xavier-initialized stacked weights and zero biases.
pub fn gru_cell_init(input_size: Int, hidden_size: Int) -> GruCell {
  nn_rnn.gru_cell_init(input_size, hidden_size)
}

/// Build an LSTM cell with Xavier-initialized stacked weights and zero biases.
pub fn lstm_cell_init(input_size: Int, hidden_size: Int) -> LstmCell {
  nn_rnn.lstm_cell_init(input_size, hidden_size)
}

/// One Elman RNN time step: `h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)`.
pub fn rnn_cell_step(
  cell: RnnCell,
  input: Tensor,
  hidden: Tensor,
) -> Result(Tensor, TensorError) {
  nn_rnn.rnn_cell_step(cell, input, hidden)
}

/// One GRU time step (PyTorch `nn.GRUCell` convention).
pub fn gru_cell_step(
  cell: GruCell,
  input: Tensor,
  hidden: Tensor,
) -> Result(Tensor, TensorError) {
  nn_rnn.gru_cell_step(cell, input, hidden)
}

/// One LSTM time step. Returns `(new_hidden, new_cell_state)`.
pub fn lstm_cell_step(
  cell: LstmCell,
  input: Tensor,
  hidden: Tensor,
  cell_state: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  nn_rnn.lstm_cell_step(cell, input, hidden, cell_state)
}

/// Run an Elman RNN cell over a list of time steps.
pub fn rnn_sequence(
  cell: RnnCell,
  inputs: List(Tensor),
  initial_hidden: Tensor,
) -> Result(#(List(Tensor), Tensor), TensorError) {
  nn_rnn.rnn_sequence(cell, inputs, initial_hidden)
}

/// Run a GRU cell over a list of time steps.
pub fn gru_sequence(
  cell: GruCell,
  inputs: List(Tensor),
  initial_hidden: Tensor,
) -> Result(#(List(Tensor), Tensor), TensorError) {
  nn_rnn.gru_sequence(cell, inputs, initial_hidden)
}

/// Run an LSTM cell over a list of time steps. Returns
/// `(all_hidden_states, final_hidden, final_cell_state)`.
pub fn lstm_sequence(
  cell: LstmCell,
  inputs: List(Tensor),
  initial_hidden: Tensor,
  initial_cell: Tensor,
) -> Result(#(List(Tensor), Tensor, Tensor), TensorError) {
  nn_rnn.lstm_sequence(cell, inputs, initial_hidden, initial_cell)
}

// --- Data -------------------------------------------------------------------

/// Build an in-memory dataset from a list of labeled samples.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let x = t.from_list([1.0])
/// let y = t.from_list([0.0])
/// let _ds = t.dataset_from_samples([t.Sample(input: x, target: y)])
/// ```
pub fn dataset_from_samples(samples: List(Sample)) -> Dataset {
  dataloader.dataset_from_samples(samples)
}

/// Build a dataset from parallel input and target tensor lists.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let assert Ok(_ds) =
///   t.dataset_from_lists([t.from_list([1.0])], [t.from_list([0.0])])
/// ```
pub fn dataset_from_lists(
  inputs: List(Tensor),
  targets: List(Tensor),
) -> Result(Dataset, TensorError) {
  dataloader.dataset_from_lists(inputs, targets)
}

/// Number of samples in the dataset.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.dataset_len(t.dataset_from_samples([]))
/// ```
pub fn dataset_len(d: Dataset) -> Int {
  dataloader.dataset_len(d)
}

/// Fetch the i-th sample (zero-indexed; negative indices wrap).
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let ds =
///   t.dataset_from_samples([
///     t.Sample(input: t.from_list([1.0]), target: t.from_list([0.0])),
///   ])
/// let assert Ok(_) = t.dataset_get(ds, -1)
/// ```
pub fn dataset_get(d: Dataset, index: Int) -> Result(Sample, TensorError) {
  dataloader.dataset_get(d, index)
}

/// Create a new data loader.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.data_loader_new(t.dataset_from_samples([]), 32, True, False)
/// ```
pub fn data_loader_new(
  dataset: Dataset,
  batch_size: Int,
  shuffle: Bool,
  drop_last: Bool,
) -> DataLoader {
  dataloader.data_loader_new(dataset, batch_size, shuffle, drop_last)
}

/// Iterate the loader once, returning all batches.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let ds =
///   t.dataset_from_samples([
///     t.Sample(input: t.from_list([1.0]), target: t.from_list([0.0])),
///   ])
/// let loader = t.data_loader_new(ds, 1, False, False)
/// let assert Ok(_) = t.data_loader_batches(loader)
/// ```
pub fn data_loader_batches(
  loader: DataLoader,
) -> Result(List(Batch), TensorError) {
  dataloader.data_loader_batches(loader)
}

/// Total number of batches a single iteration will yield.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let loader = t.data_loader_new(t.dataset_from_samples([]), 4, False, False)
/// let _ = t.data_loader_len(loader)
/// ```
pub fn data_loader_len(loader: DataLoader) -> Int {
  dataloader.data_loader_len(loader)
}

// --- Learning-rate schedulers (re-export) -----------------------------------

/// Tag identifying which schedule a `Scheduler` is implementing.
pub type SchedulerKind =
  nn_scheduler.SchedulerKind

/// Scheduler state record. See `viva_tensor/nn/scheduler` for the formulas
/// behind each variant.
pub type Scheduler =
  nn_scheduler.Scheduler

/// StepLR: `lr = base_lr * gamma^floor(step / step_size)` — staircase decay.
pub fn step_lr(base_lr: Float, step_size: Int, gamma: Float) -> Scheduler {
  nn_scheduler.step_lr(base_lr, step_size, gamma)
}

/// CosineAnnealingLR: half-cosine from `base_lr` down to `eta_min` over
/// `t_max` steps.
pub fn cosine_annealing_lr(
  base_lr: Float,
  t_max: Int,
  eta_min: Float,
) -> Scheduler {
  nn_scheduler.cosine_annealing_lr(base_lr, t_max, eta_min)
}

/// LinearWarmup: linear ramp from 0 to `base_lr` over `warmup_steps`, then
/// constant `base_lr`.
pub fn linear_warmup(base_lr: Float, warmup_steps: Int) -> Scheduler {
  nn_scheduler.linear_warmup(base_lr, warmup_steps)
}

/// OneCycleLR: linear warmup `base_lr -> max_lr` for the first
/// `pct_start * total_steps` steps, then cosine anneal `max_lr -> base_lr`.
pub fn one_cycle_lr(
  base_lr: Float,
  max_lr: Float,
  total_steps: Int,
  pct_start: Float,
) -> Scheduler {
  nn_scheduler.one_cycle_lr(base_lr, max_lr, total_steps, pct_start)
}

/// ExponentialLR: `lr = base_lr * gamma^step` — smooth exponential decay.
pub fn exponential_lr(base_lr: Float, gamma: Float) -> Scheduler {
  nn_scheduler.exponential_lr(base_lr, gamma)
}

/// Advance the scheduler by one step and return the new learning rate.
pub fn scheduler_step(s: Scheduler) -> #(Scheduler, Float) {
  nn_scheduler.scheduler_step(s)
}

/// Compute the learning rate at the scheduler's current step without
/// advancing.
pub fn scheduler_lr(s: Scheduler) -> Float {
  nn_scheduler.scheduler_lr(s)
}

/// Apply the scheduler's next learning rate to an optimizer. Advances the
/// scheduler and returns the updated `(scheduler, optimizer)` pair.
pub fn apply_to_optimizer(
  s: Scheduler,
  opt: nn_optim.Optimizer,
) -> #(Scheduler, nn_optim.Optimizer) {
  nn_scheduler.apply_to_optimizer(s, opt)
}

// --- Text tokenizers (re-export) --------------------------------------------

/// Whitespace tokenizer (encoding against a pre-trained vocabulary).
pub type WhitespaceTokenizer =
  text_tokenizer.WhitespaceTokenizer

/// Character-level tokenizer over Unicode graphemes.
pub type CharTokenizer =
  text_tokenizer.CharTokenizer

/// BERT-style WordPiece tokenizer (encoding-only).
pub type WordPieceTokenizer =
  text_tokenizer.WordPieceTokenizer

/// Byte-pair encoding tokenizer driven by a pre-trained merge table.
pub type BpeTokenizer =
  text_tokenizer.BpeTokenizer

/// Build a `WhitespaceTokenizer` from an ordered vocabulary list.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.whitespace_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "hello"],
///   "[UNK]",
///   "[PAD]",
/// )
/// ```
pub fn whitespace_tokenizer_from_vocab(
  vocab: List(String),
  unk_token: String,
  pad_token: String,
) -> WhitespaceTokenizer {
  text_tokenizer.whitespace_tokenizer_from_vocab(vocab, unk_token, pad_token)
}

/// Encode text with a `WhitespaceTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.whitespace_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "hello"],
///   "[UNK]",
///   "[PAD]",
/// )
/// let _ = t.whitespace_encode(tok, "hello")
/// ```
pub fn whitespace_encode(
  tokenizer: WhitespaceTokenizer,
  text: String,
) -> List(Int) {
  text_tokenizer.whitespace_encode(tokenizer, text)
}

/// Decode ids with a `WhitespaceTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.whitespace_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "hello"],
///   "[UNK]",
///   "[PAD]",
/// )
/// let _ = t.whitespace_decode(tok, [2])
/// ```
pub fn whitespace_decode(
  tokenizer: WhitespaceTokenizer,
  ids: List(Int),
) -> String {
  text_tokenizer.whitespace_decode(tokenizer, ids)
}

/// Build a `CharTokenizer` from an alphabet.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.char_tokenizer_from_alphabet(["?", "a", "b"], "?")
/// ```
pub fn char_tokenizer_from_alphabet(
  alphabet: List(String),
  unk_token: String,
) -> CharTokenizer {
  text_tokenizer.char_tokenizer_from_alphabet(alphabet, unk_token)
}

/// Encode text with a `CharTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.char_tokenizer_from_alphabet(["?", "a", "b"], "?")
/// let _ = t.char_encode(tok, "ab")
/// ```
pub fn char_encode(tokenizer: CharTokenizer, text: String) -> List(Int) {
  text_tokenizer.char_encode(tokenizer, text)
}

/// Decode ids with a `CharTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.char_tokenizer_from_alphabet(["?", "a", "b"], "?")
/// let _ = t.char_decode(tok, [1, 2])
/// ```
pub fn char_decode(tokenizer: CharTokenizer, ids: List(Int)) -> String {
  text_tokenizer.char_decode(tokenizer, ids)
}

/// Build a `WordPieceTokenizer` from an ordered vocabulary list.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.word_piece_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"],
///   "[UNK]",
///   "[CLS]",
///   "[SEP]",
///   "[PAD]",
/// )
/// ```
pub fn word_piece_tokenizer_from_vocab(
  vocab: List(String),
  unk_token: String,
  cls_token: String,
  sep_token: String,
  pad_token: String,
) -> WordPieceTokenizer {
  text_tokenizer.word_piece_tokenizer_from_vocab(
    vocab,
    unk_token,
    cls_token,
    sep_token,
    pad_token,
  )
}

/// Encode text with a `WordPieceTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.word_piece_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"],
///   "[UNK]",
///   "[CLS]",
///   "[SEP]",
///   "[PAD]",
/// )
/// let _ = t.word_piece_encode(tok, "hello")
/// ```
pub fn word_piece_encode(
  tokenizer: WordPieceTokenizer,
  text: String,
) -> List(Int) {
  text_tokenizer.word_piece_encode(tokenizer, text)
}

/// Decode ids with a `WordPieceTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.word_piece_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello"],
///   "[UNK]",
///   "[CLS]",
///   "[SEP]",
///   "[PAD]",
/// )
/// let _ = t.word_piece_decode(tok, [2, 4, 3])
/// ```
pub fn word_piece_decode(
  tokenizer: WordPieceTokenizer,
  ids: List(Int),
) -> String {
  text_tokenizer.word_piece_decode(tokenizer, ids)
}

/// Build a `BpeTokenizer` from a vocab and pre-trained merges.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "l", "o", "lo"],
///   [#("l", "o")],
///   "?",
/// )
/// ```
pub fn bpe_tokenizer_from_vocab_and_merges(
  vocab: List(String),
  merges: List(#(String, String)),
  unk_token: String,
) -> BpeTokenizer {
  text_tokenizer.bpe_tokenizer_from_vocab_and_merges(vocab, merges, unk_token)
}

/// Encode text with a `BpeTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "l", "o", "lo"],
///   [#("l", "o")],
///   "?",
/// )
/// let _ = t.bpe_encode(tok, "lo")
/// ```
pub fn bpe_encode(tokenizer: BpeTokenizer, text: String) -> List(Int) {
  text_tokenizer.bpe_encode(tokenizer, text)
}

/// Decode ids with a `BpeTokenizer`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let tok = t.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "l", "o", "lo"],
///   [#("l", "o")],
///   "?",
/// )
/// let _ = t.bpe_decode(tok, [3])
/// ```
pub fn bpe_decode(tokenizer: BpeTokenizer, ids: List(Int)) -> String {
  text_tokenizer.bpe_decode(tokenizer, ids)
}

/// Convert a list of ids into a `[seq_len]` tensor of integer-valued floats.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.ids_to_tensor([1, 2, 3])
/// ```
pub fn ids_to_tensor(ids: List(Int)) -> Tensor {
  text_tokenizer.ids_to_tensor(ids)
}

/// Convert a `[seq_len]` integer-valued tensor back to a `List(Int)`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.tensor_to_ids(t.ids_to_tensor([1, 2, 3]))
/// ```
pub fn tensor_to_ids(tensor: Tensor) -> List(Int) {
  text_tokenizer.tensor_to_ids(tensor)
}

/// Pad or truncate a list of ids to `max_length` using `pad_id`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// let _ = t.pad_or_truncate([1, 2], 4, 0)
/// ```
pub fn pad_or_truncate(
  ids: List(Int),
  max_length: Int,
  pad_id: Int,
) -> List(Int) {
  text_tokenizer.pad_or_truncate(ids, max_length, pad_id)
}

// --- Classification / regression metrics ------------------------------------

/// Averaging strategy for multi-class precision / recall / F1.
pub type Average =
  metrics_classification.Average

/// Classification accuracy: `(1/N) * sum_i [pred_i == target_i]`.
pub fn accuracy(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  metrics_classification.accuracy(predictions, targets)
}

/// Confusion matrix `[num_classes, num_classes]` where `cm[true, pred]`
/// counts samples.
pub fn confusion_matrix(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(Tensor, TensorError) {
  metrics_classification.confusion_matrix(predictions, targets, num_classes)
}

/// Precision aggregated by the chosen `Average`.
pub fn precision(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
  average: Average,
) -> Result(Float, TensorError) {
  metrics_classification.precision(predictions, targets, num_classes, average)
}

/// Recall aggregated by the chosen `Average`.
pub fn recall(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
  average: Average,
) -> Result(Float, TensorError) {
  metrics_classification.recall(predictions, targets, num_classes, average)
}

/// F1-score aggregated by the chosen `Average`.
pub fn f1(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
  average: Average,
) -> Result(Float, TensorError) {
  metrics_classification.f1(predictions, targets, num_classes, average)
}

/// Top-K accuracy on 2D logits with 1D class-index targets.
pub fn top_k_accuracy(
  logits: Tensor,
  targets: Tensor,
  k: Int,
) -> Result(Float, TensorError) {
  metrics_classification.top_k_accuracy(logits, targets, k)
}

/// Per-class intersection-over-union.
pub fn iou_per_class(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(List(Float), TensorError) {
  metrics_classification.iou_per_class(predictions, targets, num_classes)
}

/// Mean of per-class IoU.
pub fn mean_iou(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(Float, TensorError) {
  metrics_classification.mean_iou(predictions, targets, num_classes)
}

/// Mean Absolute Error: `(1/N) * sum_i |pred_i - target_i|`.
pub fn mean_absolute_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  metrics_regression.mean_absolute_error(predictions, targets)
}

/// Mean Squared Error: `(1/N) * sum_i (pred_i - target_i)^2`.
pub fn mean_squared_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  metrics_regression.mean_squared_error(predictions, targets)
}

/// Root Mean Squared Error: `sqrt(MSE)`.
pub fn root_mean_squared_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  metrics_regression.root_mean_squared_error(predictions, targets)
}

/// Coefficient of determination: `1 - SS_res / SS_tot`.
pub fn r_squared(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  metrics_regression.r_squared(predictions, targets)
}

/// Mean Absolute Percentage Error: `(100/N) * sum_i |pred_i - target_i| / |target_i|`.
pub fn mean_absolute_percentage_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  metrics_regression.mean_absolute_percentage_error(predictions, targets)
}

// --- Parameter initialization (re-export) -----------------------------------
//
// Wrappers around `viva_tensor/nn/init`. The deterministic constructors
// (`zeros`, `ones`, `constant`, `identity`) collide with existing top-level
// names, so we prefix the init versions with `init_`. The random
// distributions and variance-scaled initializers keep their natural names
// since they are unique to this module.

/// `init.zeros` — all-zeros tensor. Same as `zeros`, exposed here for
/// API symmetry with the rest of `init_*`.
pub fn init_zeros(shape: List(Int)) -> Tensor {
  nn_init.zeros(shape)
}

/// `init.ones` — all-ones tensor. Use case: LayerNorm scale parameters.
pub fn init_ones(shape: List(Int)) -> Tensor {
  nn_init.ones(shape)
}

/// `init.constant` — constant-filled tensor. Equivalent to `fill`.
pub fn init_constant(shape: List(Int), value: Float) -> Tensor {
  nn_init.constant(shape, value)
}

/// `init.identity` — `[n, n]` identity matrix. Same as `identity`/`eye`,
/// exposed here for API symmetry.
pub fn init_identity(n: Int) -> Tensor {
  nn_init.identity(n)
}

/// Sample each element uniformly from `[low, high)`.
/// See `viva_tensor/nn/init.uniform`.
pub fn uniform(shape: List(Int), low: Float, high: Float) -> Tensor {
  nn_init.uniform(shape, low, high)
}

/// Sample each element from `N(mean, std^2)` via the Box-Muller transform.
/// See `viva_tensor/nn/init.normal`.
pub fn normal(shape: List(Int), mean: Float, std: Float) -> Tensor {
  nn_init.normal(shape, mean, std)
}

/// Sample each element from `N(mean, std^2)` truncated to `[a, b]`.
/// See `viva_tensor/nn/init.truncated_normal`.
pub fn truncated_normal(
  shape: List(Int),
  mean: Float,
  std: Float,
  a: Float,
  b: Float,
) -> Tensor {
  nn_init.truncated_normal(shape, mean, std, a, b)
}

/// Glorot uniform init: `U(-a, a)` with `a = sqrt(6 / (fan_in + fan_out))`.
pub fn xavier_uniform(fan_in: Int, fan_out: Int) -> Tensor {
  nn_init.xavier_uniform(fan_in, fan_out)
}

/// Glorot normal init: `N(0, std^2)` with `std = sqrt(2 / (fan_in + fan_out))`.
pub fn xavier_normal(fan_in: Int, fan_out: Int) -> Tensor {
  nn_init.xavier_normal(fan_in, fan_out)
}

/// He uniform init: `U(-bound, bound)` with `bound = gain * sqrt(3 / fan_in)`.
pub fn kaiming_uniform(fan_in: Int, fan_out: Int, gain: Float) -> Tensor {
  nn_init.kaiming_uniform(fan_in, fan_out, gain)
}

/// He normal init: `N(0, std^2)` with `std = gain * sqrt(1 / fan_in)`.
pub fn kaiming_normal(fan_in: Int, fan_out: Int, gain: Float) -> Tensor {
  nn_init.kaiming_normal(fan_in, fan_out, gain)
}

/// Orthogonal init via QR. Returns `[rows, cols]` with orthonormal columns
/// (or rows, when `rows < cols`). See `viva_tensor/nn/init.orthogonal`.
pub fn orthogonal(
  rows: Int,
  cols: Int,
  gain: Float,
) -> Result(Tensor, TensorError) {
  nn_init.orthogonal(rows, cols, gain)
}

/// `sqrt(2)` — gain for layers followed by ReLU.
pub fn relu_gain() -> Float {
  nn_init.relu_gain()
}

/// `sqrt(2 / (1 + slope^2))` — gain for layers followed by Leaky ReLU.
pub fn leaky_relu_gain(negative_slope: Float) -> Float {
  nn_init.leaky_relu_gain(negative_slope)
}

/// `5/3` — gain for layers followed by tanh.
pub fn tanh_gain() -> Float {
  nn_init.tanh_gain()
}

/// `1.0` — gain for layers followed by a linear activation.
pub fn linear_gain() -> Float {
  nn_init.linear_gain()
}

/// `1.0` — gain for layers followed by sigmoid.
pub fn sigmoid_gain() -> Float {
  nn_init.sigmoid_gain()
}

// --- Transformer building blocks (re-export) --------------------------------

/// Position-wise feed-forward sublayer used by `EncoderBlock` / `DecoderBlock`.
pub type FeedForward =
  nn_transformer.FeedForward

/// Activation tag for the FFN sublayer (`ReluAct` or `GeluAct`).
pub type Activation =
  nn_transformer.Activation

/// Transformer encoder block (pre-norm style, self-attention + FFN).
pub type EncoderBlock =
  nn_transformer.EncoderBlock

/// Transformer decoder block (pre-norm style, causal self-attn + cross-attn
/// + FFN).
pub type DecoderBlock =
  nn_transformer.DecoderBlock

/// Full Transformer model — a stack of `EncoderBlock`s followed by a stack
/// of `DecoderBlock`s.
pub type Transformer =
  nn_transformer.Transformer

/// Build a zero-weight `FeedForward` sublayer.
///
/// Forward: `activation(input @ w1 + b1) @ w2 + b2`.
pub fn feed_forward_init(
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> FeedForward {
  nn_transformer.feed_forward_init(embed_dim, hidden_dim, activation)
}

/// Run the FFN forward pass on `[seq_len, embed_dim]`-shaped input.
pub fn feed_forward_forward(
  ff: FeedForward,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  nn_transformer.feed_forward_forward(ff, input)
}

/// Build a zero-weight pre-norm encoder block.
///
/// Forward (per block):
/// ```
/// r1     = input + MHA(layer_norm(input), is_causal)
/// output = r1    + FFN(layer_norm(r1))
/// ```
pub fn encoder_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  activation: Activation,
) -> Result(EncoderBlock, TensorError) {
  nn_transformer.encoder_block_init(
    embed_dim,
    num_heads,
    ffn_hidden_dim,
    activation,
  )
}

/// Encoder block forward pass on `[seq_len, embed_dim]` input.
pub fn encoder_block_forward(
  block: EncoderBlock,
  input: Tensor,
  is_causal: Bool,
) -> Result(Tensor, TensorError) {
  nn_transformer.encoder_block_forward(block, input, is_causal)
}

/// Build a zero-weight pre-norm decoder block (causal self-attn + cross-attn
/// + FFN).
///
/// Forward (per block):
/// ```
/// r1     = input + MHA_self(layer_norm1(input), is_causal=True)
/// r2     = r1    + MHA_cross(layer_norm2(r1), memory, memory)
/// output = r2    + FFN(layer_norm3(r2))
/// ```
pub fn decoder_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  activation: Activation,
) -> Result(DecoderBlock, TensorError) {
  nn_transformer.decoder_block_init(
    embed_dim,
    num_heads,
    ffn_hidden_dim,
    activation,
  )
}

/// Decoder block forward pass. Input is `[tgt_seq_len, embed_dim]`,
/// `encoder_output` is `[src_seq_len, embed_dim]`.
pub fn decoder_block_forward(
  block: DecoderBlock,
  input: Tensor,
  encoder_output: Tensor,
) -> Result(Tensor, TensorError) {
  nn_transformer.decoder_block_forward(block, input, encoder_output)
}

/// Build a full encoder+decoder Transformer stack.
pub fn transformer_init(
  num_encoder_layers: Int,
  num_decoder_layers: Int,
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  activation: Activation,
) -> Result(Transformer, TensorError) {
  nn_transformer.transformer_init(
    num_encoder_layers,
    num_decoder_layers,
    embed_dim,
    num_heads,
    ffn_hidden_dim,
    activation,
  )
}

/// Run `src` through every encoder block in order.
pub fn transformer_encode(
  model: Transformer,
  src: Tensor,
) -> Result(Tensor, TensorError) {
  nn_transformer.transformer_encode(model, src)
}

/// Run `tgt` through every decoder block, attending to `memory` per layer.
pub fn transformer_decode(
  model: Transformer,
  tgt: Tensor,
  memory: Tensor,
) -> Result(Tensor, TensorError) {
  nn_transformer.transformer_decode(model, tgt, memory)
}

/// End-to-end forward: `transformer_decode(model, tgt,
/// transformer_encode(model, src))`.
pub fn transformer_forward(
  model: Transformer,
  src: Tensor,
  tgt: Tensor,
) -> Result(Tensor, TensorError) {
  nn_transformer.transformer_forward(model, src, tgt)
}

// --- Vision transforms ------------------------------------------------------

/// Resampling mode for `vision_resize`.
///
/// - `ResizeNearest`: nearest-neighbour, blocky and cheap.
/// - `ResizeBilinear`: linear interpolation along both spatial axes
///   (`align_corners=False`).
pub type ResizeMode =
  vision_transforms.ResizeMode

/// `ResizeMode.Nearest` re-export for ergonomic call sites.
pub const resize_nearest: ResizeMode = vision_transforms.Nearest

/// `ResizeMode.Bilinear` re-export for ergonomic call sites.
pub const resize_bilinear: ResizeMode = vision_transforms.Bilinear

/// Resize a CHW (`[C, H, W]`) or NCHW (`[B, C, H, W]`) image to
/// `[..., C, new_h, new_w]` using the requested resampling mode.
pub fn vision_resize(
  image: Tensor,
  new_h: Int,
  new_w: Int,
  mode: ResizeMode,
) -> Result(Tensor, TensorError) {
  vision_transforms.resize(image, new_h, new_w, mode)
}

/// Crop the centre `target_h x target_w` region of a CHW/NCHW image.
pub fn vision_center_crop(
  image: Tensor,
  target_h: Int,
  target_w: Int,
) -> Result(Tensor, TensorError) {
  vision_transforms.center_crop(image, target_h, target_w)
}

/// Crop a `target_h x target_w` window at a random top-left corner.
/// Non-deterministic.
pub fn vision_random_crop(
  image: Tensor,
  target_h: Int,
  target_w: Int,
) -> Result(Tensor, TensorError) {
  vision_transforms.random_crop(image, target_h, target_w)
}

/// Mirror the image along the width axis.
pub fn vision_horizontal_flip(image: Tensor) -> Result(Tensor, TensorError) {
  vision_transforms.horizontal_flip(image)
}

/// Mirror the image along the height axis.
pub fn vision_vertical_flip(image: Tensor) -> Result(Tensor, TensorError) {
  vision_transforms.vertical_flip(image)
}

/// Flip horizontally with probability `p`. Non-deterministic.
pub fn vision_random_horizontal_flip(
  image: Tensor,
  p: Float,
) -> Result(Tensor, TensorError) {
  vision_transforms.random_horizontal_flip(image, p)
}

/// Per-channel `(x - mean[c]) / std[c]` normalization.
pub fn vision_normalize(
  image: Tensor,
  mean: List(Float),
  std: List(Float),
) -> Result(Tensor, TensorError) {
  vision_transforms.normalize(image, mean, std)
}

/// Convert a 3-channel image to grayscale (ITU-R 601 luma).
pub fn vision_to_grayscale(
  image: Tensor,
  num_output_channels: Int,
) -> Result(Tensor, TensorError) {
  vision_transforms.to_grayscale(image, num_output_channels)
}

/// Multiply pixel values by `factor`, clamped to `[0, 1]`.
pub fn vision_adjust_brightness(
  image: Tensor,
  factor: Float,
) -> Result(Tensor, TensorError) {
  vision_transforms.adjust_brightness(image, factor)
}

/// Linearly interpolate each pixel toward its channel mean and clamp to
/// `[0, 1]`.
pub fn vision_adjust_contrast(
  image: Tensor,
  factor: Float,
) -> Result(Tensor, TensorError) {
  vision_transforms.adjust_contrast(image, factor)
}

/// HWC byte image (`[0..255]`) → CHW tensor in `[0, 1]`.
pub fn vision_to_tensor(
  byte_image: List(Int),
  height: Int,
  width: Int,
  channels: Int,
) -> Result(Tensor, TensorError) {
  vision_transforms.to_tensor(byte_image, height, width, channels)
}

/// CHW tensor in `[0, 1]` → HWC byte image (`[0..255]`).
pub fn vision_to_byte_image(image: Tensor) -> Result(List(Int), TensorError) {
  vision_transforms.to_byte_image(image)
}

/// Apply a list of transforms in order, threading the result through each
/// step. Bails on the first `Error`.
pub fn vision_compose(
  transforms: List(fn(Tensor) -> Result(Tensor, TensorError)),
  image: Tensor,
) -> Result(Tensor, TensorError) {
  vision_transforms.compose(transforms, image)
}

// --- HuggingFace SafeTensors loader (re-export) -----------------------------

/// Loader-local error type. See `viva_tensor/io/hf_loader.HfLoadError`.
pub type HfLoadError =
  hf_loader_io.HfLoadError

/// Structural config for `from_safetensors_file`. See
/// `viva_tensor/io/hf_loader.TransformerConfig`.
pub type TransformerConfig =
  hf_loader_io.TransformerConfig

/// Read a `.safetensors` file into a `Dict(String, Tensor)`, mapping I/O
/// failures into `HfLoadError.IoError`.
pub fn load_safetensors_dict(
  path: String,
) -> Result(Dict(String, Tensor), HfLoadError) {
  hf_loader_io.load_safetensors_dict(path)
}

/// Load an `Embedding` from `prefix <> ".weight"` (`[vocab_size,
/// embedding_dim]`).
pub fn load_embedding(
  weights: Dict(String, Tensor),
  prefix: String,
  vocab_size: Int,
  embedding_dim: Int,
) -> Result(Embedding, HfLoadError) {
  hf_loader_io.load_embedding(weights, prefix, vocab_size, embedding_dim)
}

/// Load a `LayerNorm` from `prefix <> ".weight"` (scale) and
/// `prefix <> ".bias"`, both `[num_features]`.
pub fn load_layer_norm(
  weights: Dict(String, Tensor),
  prefix: String,
  num_features: Int,
) -> Result(LayerNorm, HfLoadError) {
  hf_loader_io.load_layer_norm(weights, prefix, num_features)
}

/// Load a `MultiHeadAttention` from `q_proj`/`k_proj`/`v_proj`/`out_proj`
/// (weight + bias each) under the supplied `prefix`.
pub fn load_multi_head_attention(
  weights: Dict(String, Tensor),
  prefix: String,
  num_heads: Int,
  embed_dim: Int,
) -> Result(MultiHeadAttention, HfLoadError) {
  hf_loader_io.load_multi_head_attention(weights, prefix, num_heads, embed_dim)
}

/// Load a `FeedForward` from `linear1`/`linear2` (weight + bias each) under
/// the supplied `prefix`.
pub fn load_feed_forward(
  weights: Dict(String, Tensor),
  prefix: String,
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(FeedForward, HfLoadError) {
  hf_loader_io.load_feed_forward(
    weights,
    prefix,
    embed_dim,
    hidden_dim,
    activation,
  )
}

/// Load a single `EncoderBlock` (MHA + 2× LayerNorm + FFN) under
/// `prefix` (e.g. `"encoder.layers.0"`).
pub fn load_encoder_block(
  weights: Dict(String, Tensor),
  prefix: String,
  num_heads: Int,
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(EncoderBlock, HfLoadError) {
  hf_loader_io.load_encoder_block(
    weights,
    prefix,
    num_heads,
    embed_dim,
    hidden_dim,
    activation,
  )
}

/// Load a full `Transformer` (encoder stack + decoder stack) under the
/// conventional `encoder.layers.{i}` / `decoder.layers.{i}` prefixes.
pub fn load_transformer(
  weights: Dict(String, Tensor),
  num_enc_layers: Int,
  num_dec_layers: Int,
  embed_dim: Int,
  num_heads: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(Transformer, HfLoadError) {
  hf_loader_io.load_transformer(
    weights,
    num_enc_layers,
    num_dec_layers,
    embed_dim,
    num_heads,
    hidden_dim,
    activation,
  )
}

/// Read a `.safetensors` file then project it into a `Transformer` using
/// the dimensions in `config`.
pub fn from_safetensors_file(
  path: String,
  config: TransformerConfig,
) -> Result(Transformer, HfLoadError) {
  hf_loader_io.from_safetensors_file(path, config)
}
