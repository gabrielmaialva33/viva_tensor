//// NIF Stubs for GPU Acceleration
////
//// This module provides FFI bindings for GPU-accelerated tensor operations.
//// Currently returns NotImplemented - will be implemented with Rust NIFs + cuDNN.
////
//// ## Planned NIF Operations
//// - `conv2d_cudnn` - cuDNN convolution
//// - `matmul_cublas` - cuBLAS matrix multiplication
//// - `pool2d_cudnn` - cuDNN pooling
////
//// ## Usage
//// When NIFs are available, they will be used automatically.
//// Falls back to Pure Gleam implementation otherwise.

import viva_tensor/tensor.{type Tensor, type TensorError}

/// NIF availability status
pub type NifStatus {
  Available(backend: String)
  NotAvailable(reason: String)
}

/// Check if GPU NIFs are available
pub fn check_nif_status() -> NifStatus {
  // TODO: Check for Rust NIF library
  NotAvailable(reason: "GPU NIFs not yet implemented. Using Pure Gleam.")
}

/// GPU-accelerated conv2d (stub)
/// Will use cuDNN when available
pub fn conv2d_gpu(
  _input: Tensor,
  _kernel: Tensor,
  _stride_h: Int,
  _stride_w: Int,
  _pad_h: Int,
  _pad_w: Int,
) -> Result(Tensor, TensorError) {
  // TODO: Implement with Rust NIF + cuDNN
  // For now, returns error to signal fallback to Pure Gleam
  Error(tensor.InvalidShape(reason: "GPU NIF not available"))
}

/// GPU-accelerated matmul (stub)
/// Will use cuBLAS when available
pub fn matmul_gpu(_a: Tensor, _b: Tensor) -> Result(Tensor, TensorError) {
  // TODO: Implement with Rust NIF + cuBLAS
  Error(tensor.InvalidShape(reason: "GPU NIF not available"))
}

/// GPU-accelerated max pooling (stub)
pub fn max_pool2d_gpu(
  _input: Tensor,
  _pool_h: Int,
  _pool_w: Int,
  _stride_h: Int,
  _stride_w: Int,
) -> Result(Tensor, TensorError) {
  // TODO: Implement with Rust NIF + cuDNN
  Error(tensor.InvalidShape(reason: "GPU NIF not available"))
}

/// GPU-accelerated avg pooling (stub)
pub fn avg_pool2d_gpu(
  _input: Tensor,
  _pool_h: Int,
  _pool_w: Int,
  _stride_h: Int,
  _stride_w: Int,
) -> Result(Tensor, TensorError) {
  // TODO: Implement with Rust NIF + cuDNN
  Error(tensor.InvalidShape(reason: "GPU NIF not available"))
}

/// Batch normalization (stub)
pub fn batch_norm_gpu(
  _input: Tensor,
  _gamma: Tensor,
  _beta: Tensor,
  _epsilon: Float,
) -> Result(Tensor, TensorError) {
  Error(tensor.InvalidShape(reason: "GPU NIF not available"))
}

// =============================================================================
// NIF IMPLEMENTATION PLAN (for future Rust implementation)
// =============================================================================
//
// 1. Create Rust crate with rustler
// 2. Link against:
//    - cuDNN (convolution, pooling, batch norm)
//    - cuBLAS (matmul, GEMM)
//    - CUDA Runtime (memory management)
//
// 3. Memory layout:
//    - Tensor data stays in GPU memory
//    - Lazy transfer to/from host
//    - Zero-copy where possible
//
// 4. Dispatch strategy:
//    - Check tensor size
//    - If size > threshold, use GPU
//    - Otherwise, use Pure Gleam (overhead not worth it)
//
// 5. Example Rust NIF signature:
//    ```rust
//    #[rustler::nif]
//    fn conv2d_cudnn(
//        input: Binary,
//        kernel: Binary,
//        input_shape: Vec<i64>,
//        kernel_shape: Vec<i64>,
//        stride: (i64, i64),
//        padding: (i64, i64),
//    ) -> Result<(Binary, Vec<i64>), String> {
//        // cuDNN implementation
//    }
//    ```
