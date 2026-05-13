//// CudaTensor - Persistent GPU Memory
////
//// Tensors that live on the GPU. Ideal for weights and heavy compute.
////
//// - **FP32 (CudaTensor)**: Standard precision. 40+ TFLOPS on RTX 4090.
//// - **FP16 (CudaTensor16)**: Low precision, high throughput using Tensor Cores. 330+ TFLOPS!
////
//// Data is uploaded once and stays on device.
//// Operations are launched asynchronously (mostly).

import gleam/result
import viva_tensor/core/error.{DimensionError, ShapeMismatch}
import viva_tensor/core/ffi
import viva_tensor/tensor

// =============================================================================
// FP32 CudaTensor
// =============================================================================

/// Reference to a tensor stored in GPU memory (FP32)
pub type CudaTensor =
  ffi.CudaTensorRef

/// Upload data to GPU (FP32)
pub fn new(data: List(Float), shape: List(Int)) -> Result(CudaTensor, String) {
  ffi.ct_from_list(data, shape)
}

/// Download data from GPU (FP32)
pub fn to_list(tensor: CudaTensor) -> Result(List(Float), String) {
  ffi.ct_to_list(tensor)
}

/// Get shape of tensor
pub fn shape(tensor: CudaTensor) -> Result(List(Int), String) {
  ffi.ct_shape(tensor)
}

/// Matrix Multiplication (FP32)
/// C = A @ B
pub fn matmul(
  a: CudaTensor,
  b: CudaTensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensor, String) {
  ffi.ct_matmul(a, b, m, n, k)
}

// =============================================================================
// FP16 CudaTensor (Tensor Cores)
// =============================================================================

/// Reference to a tensor stored in GPU memory (FP16)
pub type CudaTensor16 =
  ffi.CudaTensor16Ref

/// Check if FP16 Tensor Cores are available
pub fn fp16_available() -> Bool {
  ffi.ct16_available()
}

/// Upload data to GPU (converts f64 -> f16)
pub fn new16(
  data: List(Float),
  shape: List(Int),
) -> Result(CudaTensor16, String) {
  ffi.ct16_from_list(data, shape)
}

/// Download data from GPU (converts f16 -> f64)
pub fn to_list16(tensor: CudaTensor16) -> Result(List(Float), String) {
  ffi.ct16_to_list(tensor)
}

/// Get shape of FP16 tensor
pub fn shape16(tensor: CudaTensor16) -> Result(List(Int), String) {
  ffi.ct16_shape(tensor)
}

/// Matrix Multiplication (FP16 Tensor Cores)
/// C = A @ B
///
/// Uses HMMA (Half-precision Matrix Multiply Accumulate) instructions.
/// Expect massive speedups (up to 330 TFLOPS) if dimensions align with 16x16.
pub fn matmul16(
  a: CudaTensor16,
  b: CudaTensor16,
  m: Int,
  n: Int,
  k: Int,
) -> Result(CudaTensor, String) {
  ffi.ct16_matmul(a, b, m, n, k)
}

// =============================================================================
// RTX 4090-first planner
// =============================================================================

/// Backend selected by the RTX-first planner.
pub type AccelerationBackend {
  Rtx4090Fp16
  Rtx4090Fp32
  MklNative
  CpuFallback
}

/// Result of accelerated execution.
///
/// CUDA variants stay on the GPU. Use `to_cpu_tensor` at API boundaries.
pub type AcceleratedTensor {
  CudaFp16(ref: CudaTensor16, shape: List(Int), backend: AccelerationBackend)
  CudaFp32(ref: CudaTensor, shape: List(Int), backend: AccelerationBackend)
  Cpu(tensor: tensor.Tensor, backend: AccelerationBackend)
}

/// Workspace for persistent GPU buffers.
///
/// It centralizes the chosen GPU precision so callers can allocate reusable
/// output buffers without manually uploading zeros every time.
pub type GpuWorkspace {
  GpuWorkspace(backend: AccelerationBackend)
}

/// Persisted linear layer parameters.
///
/// `weight` has shape `[input_features, output_features]` and `bias` has shape
/// `[output_features]`.
pub type LinearLayer {
  LinearLayer(
    weight: AcceleratedTensor,
    bias: AcceleratedTensor,
    input_features: Int,
    output_features: Int,
    backend: AccelerationBackend,
  )
}

/// Create an RTX 4090 FP16 workspace.
pub fn gpu_workspace() -> Result(GpuWorkspace, tensor.TensorError) {
  case ffi.zig_is_loaded() && ffi.ct16_available() {
    True -> Ok(GpuWorkspace(backend: Rtx4090Fp16))
    False -> Error(DimensionError("CUDA FP16 backend is not available"))
  }
}

/// Allocate a reusable zero-filled output buffer in workspace memory.
pub fn workspace_zeros(
  workspace: GpuWorkspace,
  shape: List(Int),
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case workspace {
    GpuWorkspace(Rtx4090Fp16) -> to_rtx4090_fp16(tensor.zeros(shape))
    GpuWorkspace(Rtx4090Fp32) -> to_rtx4090_fp32(tensor.zeros(shape))
    GpuWorkspace(MklNative) -> tensor.native_zeros(shape) |> result.map(to_mkl)
    GpuWorkspace(CpuFallback) ->
      Ok(Cpu(tensor: tensor.zeros(shape), backend: CpuFallback))
  }
}

/// Move a tensor into workspace memory.
pub fn workspace_from_tensor(
  workspace: GpuWorkspace,
  t: tensor.Tensor,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case workspace {
    GpuWorkspace(Rtx4090Fp16) -> to_rtx4090_fp16(t)
    GpuWorkspace(Rtx4090Fp32) -> to_rtx4090_fp32(t)
    GpuWorkspace(MklNative) -> {
      let shape = tensor.shape(t)
      tensor.native_from_list(tensor.to_list(t), shape)
      |> result.map(to_mkl)
    }
    GpuWorkspace(CpuFallback) -> Ok(Cpu(tensor: t, backend: CpuFallback))
  }
}

/// Create a persisted FP16 linear layer on the RTX.
pub fn linear_layer_fp16(
  weight: tensor.Tensor,
  bias: tensor.Tensor,
) -> Result(LinearLayer, tensor.TensorError) {
  use workspace <- result.try(gpu_workspace())
  linear_layer(workspace, weight, bias)
}

/// Create a persisted linear layer in workspace memory.
pub fn linear_layer(
  workspace: GpuWorkspace,
  weight: tensor.Tensor,
  bias: tensor.Tensor,
) -> Result(LinearLayer, tensor.TensorError) {
  case tensor.shape(weight), tensor.shape(bias) {
    [input_features, output_features], [bias_features]
      if output_features == bias_features
    -> {
      use weight_acc <- result.try(workspace_from_tensor(workspace, weight))
      use bias_acc <- result.try(workspace_from_tensor(workspace, bias))
      Ok(LinearLayer(
        weight: weight_acc,
        bias: bias_acc,
        input_features: input_features,
        output_features: output_features,
        backend: workspace_backend(workspace),
      ))
    }

    [_input_features, output_features], [bias_features] ->
      Error(ShapeMismatch(expected: [output_features], got: [bias_features]))

    _, _ -> Error(DimensionError("Expected weight matrix and bias vector"))
  }
}

/// Allocate a reusable output buffer for a persisted linear layer.
pub fn linear_output(
  workspace: GpuWorkspace,
  layer: LinearLayer,
  batch_size: Int,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  workspace_zeros(workspace, [batch_size, linear_layer_output_features(layer)])
}

/// Run `out = relu(input @ layer.weight + layer.bias)`.
pub fn linear_relu_forward_into(
  out: AcceleratedTensor,
  input: AcceleratedTensor,
  layer: LinearLayer,
) -> Result(Nil, tensor.TensorError) {
  linear_forward_into(out, input, layer, "relu")
}

/// Run `out = gelu(input @ layer.weight + layer.bias)`.
pub fn linear_gelu_forward_into(
  out: AcceleratedTensor,
  input: AcceleratedTensor,
  layer: LinearLayer,
) -> Result(Nil, tensor.TensorError) {
  linear_forward_into(out, input, layer, "gelu")
}

/// Workspace backend.
pub fn workspace_backend(workspace: GpuWorkspace) -> AccelerationBackend {
  case workspace {
    GpuWorkspace(backend) -> backend
  }
}

/// Linear layer backend.
pub fn linear_layer_backend(layer: LinearLayer) -> AccelerationBackend {
  case layer {
    LinearLayer(_, _, _, _, backend) -> backend
  }
}

/// Linear layer input feature count.
pub fn linear_layer_input_features(layer: LinearLayer) -> Int {
  case layer {
    LinearLayer(_, _, input_features, _, _) -> input_features
  }
}

/// Linear layer output feature count.
pub fn linear_layer_output_features(layer: LinearLayer) -> Int {
  case layer {
    LinearLayer(_, _, _, output_features, _) -> output_features
  }
}

/// Move a CPU tensor to the best persistent backend: RTX 4090 FP16, RTX 4090
/// FP32, MKL/native CPU, then plain CPU.
pub fn to_accelerated(
  t: tensor.Tensor,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  let shape = tensor.shape(t)
  case to_rtx4090_fp16(t) {
    Ok(gpu) -> Ok(gpu)
    Error(_) -> {
      case to_rtx4090_fp32(t) {
        Ok(gpu) -> Ok(gpu)
        Error(_) -> Ok(to_mkl_or_cpu(t, shape))
      }
    }
  }
}

/// Upload a tensor to RTX 4090 FP16 memory and keep it there.
pub fn to_rtx4090_fp16(
  t: tensor.Tensor,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case ffi.zig_is_loaded() && ffi.ct16_available() {
    False -> Error(DimensionError("CUDA FP16 backend is not available"))
    True ->
      ffi.ct16_from_list(tensor.to_list(t), tensor.shape(t))
      |> result.map(fn(ref) {
        CudaFp16(ref: ref, shape: tensor.shape(t), backend: Rtx4090Fp16)
      })
      |> result.map_error(fn(reason) { DimensionError(reason) })
  }
}

/// Upload a tensor to RTX 4090 FP32 memory and keep it there.
pub fn to_rtx4090_fp32(
  t: tensor.Tensor,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case ffi.zig_is_loaded() {
    False -> Error(DimensionError("CUDA FP32 backend is not available"))
    True ->
      ffi.ct_from_list(tensor.to_list(t), tensor.shape(t))
      |> result.map(fn(ref) {
        CudaFp32(ref: ref, shape: tensor.shape(t), backend: Rtx4090Fp32)
      })
      |> result.map_error(fn(reason) { DimensionError(reason) })
  }
}

/// Matrix multiplication with priority: RTX 4090 FP16, RTX 4090 FP32, MKL, CPU.
pub fn matmul_auto(
  a: tensor.Tensor,
  b: tensor.Tensor,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [m, k], [k2, n] if k == k2 -> {
      case try_rtx4090_fp16(a, b, m, n, k) {
        Ok(result) -> Ok(result)
        Error(_) -> {
          case try_rtx4090_fp32(a, b, m, n, k) {
            Ok(result) -> Ok(result)
            Error(_) -> matmul_mkl_then_cpu(a, b, m, n, k)
          }
        }
      }
    }
    [_m, k], [k2, _n] -> Error(ShapeMismatch(expected: [k, -1], got: [k2, -1]))
    _, _ -> Error(DimensionError("Expected two matrices"))
  }
}

/// Matrix multiplication between persistent accelerated tensors.
///
/// Matching CUDA inputs stay on device. CPU/native inputs use the existing CPU
/// matmul path without forcing GPU uploads.
pub fn matmul_accelerated(
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case accelerated_shape(a), accelerated_shape(b) {
    [m, k], [k2, n] if k == k2 -> matmul_accelerated_checked(a, b, m, n, k)
    [_m, k], [k2, _n] -> Error(ShapeMismatch(expected: [k, -1], got: [k2, -1]))
    _, _ -> Error(DimensionError("Expected two matrices"))
  }
}

/// Write `out = a @ b` into a persistent accelerated output buffer.
///
/// CUDA inputs stay on the GPU and reuse `out` with no output allocation.
pub fn matmul_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(Nil, tensor.TensorError) {
  case accelerated_shape(out), accelerated_shape(a), accelerated_shape(b) {
    [m_out, n_out], [m, k], [k2, n] if k == k2 && m_out == m && n_out == n ->
      matmul_accelerated_into_checked(out, a, b, m, n, k)
    [m_out, n_out], [m, _k], [_k2, n] ->
      Error(ShapeMismatch(expected: [m, n], got: [m_out, n_out]))
    _, _, _ -> Error(DimensionError("Expected matrices"))
  }
}

/// Write `out = relu(a @ b)` using the FP16 Tensor Core fused epilogue.
pub fn matmul_relu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(Nil, tensor.TensorError) {
  fused_activation_accelerated_into(out, a, b, "relu")
}

/// Write `out = gelu(a @ b)` using the FP16 Tensor Core fused epilogue.
pub fn matmul_gelu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
) -> Result(Nil, tensor.TensorError) {
  fused_activation_accelerated_into(out, a, b, "gelu")
}

/// Write `out = relu(a @ b + bias)` using the FP16 Tensor Core fused epilogue.
pub fn linear_relu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  bias: AcceleratedTensor,
) -> Result(Nil, tensor.TensorError) {
  fused_linear_accelerated_into(out, a, b, bias, "relu")
}

/// Write `out = gelu(a @ b + bias)` using the FP16 Tensor Core fused epilogue.
pub fn linear_gelu_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  bias: AcceleratedTensor,
) -> Result(Nil, tensor.TensorError) {
  fused_linear_accelerated_into(out, a, b, bias, "gelu")
}

/// Download an accelerated tensor to a regular CPU tensor.
pub fn to_cpu_tensor(
  t: AcceleratedTensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  case t {
    Cpu(tensor, _) -> Ok(tensor)
    CudaFp16(ref, shape, _) ->
      ffi.ct16_to_list(ref)
      |> result.map(fn(data) { tensor.Tensor(data: data, shape: shape) })
      |> result.map_error(fn(reason) { DimensionError(reason) })
    CudaFp32(ref, shape, _) ->
      ffi.ct_to_list(ref)
      |> result.map(fn(data) { tensor.Tensor(data: data, shape: shape) })
      |> result.map_error(fn(reason) { DimensionError(reason) })
  }
}

/// Inspect which backend was selected.
pub fn backend(t: AcceleratedTensor) -> AccelerationBackend {
  case t {
    CudaFp16(_, _, backend) -> backend
    CudaFp32(_, _, backend) -> backend
    Cpu(_, backend) -> backend
  }
}

/// Shape of an accelerated tensor without forcing a download.
pub fn accelerated_shape(t: AcceleratedTensor) -> List(Int) {
  case t {
    CudaFp16(_, shape, _) -> shape
    CudaFp32(_, shape, _) -> shape
    Cpu(tensor, _) -> tensor.shape
  }
}

/// Wait for queued CUDA work to complete.
pub fn sync() -> Result(Nil, tensor.TensorError) {
  ffi.cuda_sync()
  |> result.map_error(fn(reason) { DimensionError(reason) })
}

fn matmul_accelerated_checked(
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case a, b {
    CudaFp16(a_ref, _, _), CudaFp16(b_ref, _, _) ->
      ffi.ct16_matmul(a_ref, b_ref, m, n, k)
      |> result.map(fn(out) {
        CudaFp32(ref: out, shape: [m, n], backend: Rtx4090Fp16)
      })
      |> result.map_error(fn(reason) { DimensionError(reason) })

    CudaFp32(a_ref, _, _), CudaFp32(b_ref, _, _) ->
      ffi.ct_matmul(a_ref, b_ref, m, n, k)
      |> result.map(fn(out) {
        CudaFp32(ref: out, shape: [m, n], backend: Rtx4090Fp32)
      })
      |> result.map_error(fn(reason) { DimensionError(reason) })

    Cpu(a_tensor, _), Cpu(b_tensor, _) ->
      tensor.matmul(a_tensor, b_tensor)
      |> result.map(fn(out) { Cpu(tensor: out, backend: backend(a)) })

    _, _ -> {
      use a_cpu <- result.try(to_cpu_tensor(a))
      use b_cpu <- result.try(to_cpu_tensor(b))
      matmul_auto(a_cpu, b_cpu)
    }
  }
}

fn matmul_accelerated_into_checked(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(Nil, tensor.TensorError) {
  case out, a, b {
    CudaFp16(out_ref, _, _), CudaFp16(a_ref, _, _), CudaFp16(b_ref, _, _) ->
      ffi.ct16_matmul_inplace(a_ref, b_ref, out_ref, m, n, k)
      |> result.map_error(fn(reason) { DimensionError(reason) })

    CudaFp32(out_ref, _, _), CudaFp32(a_ref, _, _), CudaFp32(b_ref, _, _) ->
      ffi.ct_matmul_inplace(a_ref, b_ref, out_ref, m, n, k)
      |> result.map_error(fn(reason) { DimensionError(reason) })

    Cpu(out_tensor, _), Cpu(a_tensor, _), Cpu(b_tensor, _) ->
      tensor.matmul_into(out_tensor, a_tensor, b_tensor)

    _, _, _ ->
      Error(DimensionError("Output, lhs, and rhs must use the same backend"))
  }
}

fn fused_activation_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  activation: String,
) -> Result(Nil, tensor.TensorError) {
  case accelerated_shape(out), accelerated_shape(a), accelerated_shape(b) {
    [m_out, n_out], [m, k], [k2, n] if k == k2 && m_out == m && n_out == n ->
      fused_activation_accelerated_into_checked(out, a, b, m, n, k, activation)
    [m_out, n_out], [m, _k], [_k2, n] ->
      Error(ShapeMismatch(expected: [m, n], got: [m_out, n_out]))
    _, _, _ -> Error(DimensionError("Expected matrices"))
  }
}

fn fused_activation_accelerated_into_checked(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  m: Int,
  n: Int,
  k: Int,
  activation: String,
) -> Result(Nil, tensor.TensorError) {
  case out, a, b {
    CudaFp16(out_ref, _, _), CudaFp16(a_ref, _, _), CudaFp16(b_ref, _, _) -> {
      case activation {
        "relu" -> ffi.ct16_matmul_fused_relu(a_ref, b_ref, out_ref, m, n, k)
        "gelu" -> ffi.ct16_matmul_fused_gelu(a_ref, b_ref, out_ref, m, n, k)
        _ -> Error("unsupported_activation")
      }
      |> result.map_error(fn(reason) { DimensionError(reason) })
    }
    _, _, _ ->
      Error(DimensionError(
        "Fused CUDA activation requires FP16 accelerated tensors",
      ))
  }
}

fn fused_linear_accelerated_into(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  bias: AcceleratedTensor,
  activation: String,
) -> Result(Nil, tensor.TensorError) {
  case
    accelerated_shape(out),
    accelerated_shape(a),
    accelerated_shape(b),
    accelerated_shape(bias)
  {
    [m_out, n_out], [m, k], [k2, n], [n_bias]
      if k == k2 && m_out == m && n_out == n && n_bias == n
    ->
      fused_linear_accelerated_into_checked(
        out,
        a,
        b,
        bias,
        m,
        n,
        k,
        activation,
      )

    [m_out, n_out], [m, _k], [_k2, n], [_n_bias] ->
      Error(ShapeMismatch(expected: [m, n], got: [m_out, n_out]))

    _, _, _, _ -> Error(DimensionError("Expected matrices and a bias vector"))
  }
}

fn fused_linear_accelerated_into_checked(
  out: AcceleratedTensor,
  a: AcceleratedTensor,
  b: AcceleratedTensor,
  bias: AcceleratedTensor,
  m: Int,
  n: Int,
  k: Int,
  activation: String,
) -> Result(Nil, tensor.TensorError) {
  case out, a, b, bias {
    CudaFp16(out_ref, _, _),
      CudaFp16(a_ref, _, _),
      CudaFp16(b_ref, _, _),
      CudaFp16(bias_ref, _, _)
    -> {
      case activation {
        "relu" -> ffi.ct16_linear_relu(a_ref, b_ref, bias_ref, out_ref, m, n, k)
        "gelu" -> ffi.ct16_linear_gelu(a_ref, b_ref, bias_ref, out_ref, m, n, k)
        _ -> Error("unsupported_activation")
      }
      |> result.map_error(fn(reason) { DimensionError(reason) })
    }

    Cpu(out_tensor, _), Cpu(a_tensor, _), Cpu(b_tensor, _), Cpu(bias_tensor, _)
    -> {
      case activation {
        "relu" ->
          tensor.linear_relu_into(out_tensor, a_tensor, b_tensor, bias_tensor)
        _ -> Error(DimensionError("CPU fused GELU is not implemented"))
      }
    }

    _, _, _, _ ->
      Error(DimensionError("Fused linear activation requires matching backends"))
  }
}

fn linear_forward_into(
  out: AcceleratedTensor,
  input: AcceleratedTensor,
  layer: LinearLayer,
  activation: String,
) -> Result(Nil, tensor.TensorError) {
  case layer {
    LinearLayer(weight, bias, input_features, output_features, _backend) -> {
      case accelerated_shape(input), accelerated_shape(out) {
        [batch_size, got_input], [out_batch, got_output]
          if got_input == input_features
          && out_batch == batch_size
          && got_output == output_features
        -> {
          case activation {
            "relu" -> linear_relu_accelerated_into(out, input, weight, bias)
            "gelu" -> linear_gelu_accelerated_into(out, input, weight, bias)
            _ -> Error(DimensionError("Unsupported activation"))
          }
        }

        [_batch_size, got_input], [_out_batch, _got_output] ->
          Error(ShapeMismatch(expected: [input_features], got: [got_input]))

        _, _ -> Error(DimensionError("Expected input and output matrices"))
      }
    }
  }
}

fn to_mkl(t: tensor.Tensor) -> AcceleratedTensor {
  Cpu(tensor: t, backend: MklNative)
}

fn try_rtx4090_fp16(
  a: tensor.Tensor,
  b: tensor.Tensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(AcceleratedTensor, Nil) {
  case ffi.zig_is_loaded() && ffi.ct16_available() {
    False -> Error(Nil)
    True -> {
      use a_gpu <- result.try(
        ffi.ct16_from_list(tensor.to_list(a), [m, k])
        |> result.map_error(fn(_) { Nil }),
      )
      use b_gpu <- result.try(
        ffi.ct16_from_list(tensor.to_list(b), [k, n])
        |> result.map_error(fn(_) { Nil }),
      )
      ffi.ct16_matmul(a_gpu, b_gpu, m, n, k)
      |> result.map(fn(out) {
        CudaFp32(ref: out, shape: [m, n], backend: Rtx4090Fp16)
      })
      |> result.map_error(fn(_) { Nil })
    }
  }
}

fn try_rtx4090_fp32(
  a: tensor.Tensor,
  b: tensor.Tensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(AcceleratedTensor, Nil) {
  case ffi.zig_is_loaded() {
    False -> Error(Nil)
    True -> {
      use a_gpu <- result.try(
        ffi.ct_from_list(tensor.to_list(a), [m, k])
        |> result.map_error(fn(_) { Nil }),
      )
      use b_gpu <- result.try(
        ffi.ct_from_list(tensor.to_list(b), [k, n])
        |> result.map_error(fn(_) { Nil }),
      )
      ffi.ct_matmul(a_gpu, b_gpu, m, n, k)
      |> result.map(fn(out) {
        CudaFp32(ref: out, shape: [m, n], backend: Rtx4090Fp32)
      })
      |> result.map_error(fn(_) { Nil })
    }
  }
}

fn matmul_mkl_then_cpu(
  a: tensor.Tensor,
  b: tensor.Tensor,
  m: Int,
  n: Int,
  k: Int,
) -> Result(AcceleratedTensor, tensor.TensorError) {
  case to_native(a, [m, k]), to_native(b, [k, n]) {
    Ok(a_native), Ok(b_native) ->
      tensor.matmul(a_native, b_native)
      |> result.map(fn(out) { Cpu(tensor: out, backend: MklNative) })
    _, _ ->
      tensor.matmul(a, b)
      |> result.map(fn(out) { Cpu(tensor: out, backend: CpuFallback) })
  }
}

fn to_native(t: tensor.Tensor, shape: List(Int)) -> Result(tensor.Tensor, Nil) {
  case tensor.native_ref(t) {
    Ok(_) -> Ok(t)
    Error(_) ->
      tensor.native_from_list(tensor.to_list(t), shape)
      |> result.map_error(fn(_) { Nil })
  }
}

fn to_mkl_or_cpu(t: tensor.Tensor, shape: List(Int)) -> AcceleratedTensor {
  case to_native(t, shape) {
    Ok(native) -> Cpu(tensor: native, backend: MklNative)
    Error(_) -> Cpu(tensor: t, backend: CpuFallback)
  }
}
