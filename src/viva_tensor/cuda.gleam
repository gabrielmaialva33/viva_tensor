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
  CudaFp32(ref: CudaTensor, shape: List(Int), backend: AccelerationBackend)
  Cpu(tensor: tensor.Tensor, backend: AccelerationBackend)
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

/// Download an accelerated tensor to a regular CPU tensor.
pub fn to_cpu_tensor(
  t: AcceleratedTensor,
) -> Result(tensor.Tensor, tensor.TensorError) {
  case t {
    Cpu(tensor, _) -> Ok(tensor)
    CudaFp32(ref, shape, _) ->
      ffi.ct_to_list(ref)
      |> result.map(fn(data) { tensor.Tensor(data: data, shape: shape) })
      |> result.map_error(fn(reason) { DimensionError(reason) })
  }
}

/// Inspect which backend was selected.
pub fn backend(t: AcceleratedTensor) -> AccelerationBackend {
  case t {
    CudaFp32(_, _, backend) -> backend
    Cpu(_, backend) -> backend
  }
}

/// Shape of an accelerated tensor without forcing a download.
pub fn accelerated_shape(t: AcceleratedTensor) -> List(Int) {
  case t {
    CudaFp32(_, shape, _) -> shape
    Cpu(tensor, _) -> tensor.shape
  }
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
