//// SparseTensor - 2:4 Sparsity
////
//// Use cuSPARSELt to prune and compress weight matrices.
////
//// - **2:4 Structure**: For every block of 4 elements, 2 must be zero.
//// - **Compression**: Reduces memory usage by ~50% (1.78x practical).
//// - **Speedup**: Up to 2x theoretical (660 TFLOPS), 61% measured speedup vs dense.
////
//// Ideal for Large Language Model (LLM) weights.

import viva_tensor/core/ffi

/// Reference to a 2:4 structured sparse tensor (GPU)
pub type SparseTensor =
  ffi.SparseTensorRef

/// Check if cuSPARSELt is available
pub fn available() -> Bool {
  ffi.sparse_available()
}

/// Create SparseTensor from CudaTensor16 (Prune + Compress)
///
/// This operation is destructive: it prunes the smallest 2 values in every 4-element block.
/// The resulting sparse tensor is stored in a compressed format on the GPU.
pub fn from_cuda16(tensor: ffi.CudaTensor16Ref) -> Result(SparseTensor, String) {
  ffi.sparse_from_ct16(tensor)
}

/// Get shape of the original dense tensor [Rows, Cols]
pub fn shape(tensor: SparseTensor) -> Result(List(Int), String) {
  ffi.sparse_shape(tensor)
}

/// Get actual compression ratio (DenseBytes / SparseBytes)
pub fn compression_ratio(tensor: SparseTensor) -> Result(Float, String) {
  ffi.sparse_compression_ratio(tensor)
}

/// Sparse Matrix Multiplication (SpMM)
///
/// C = Sparse(A) @ Dense(B)
///
/// - `a_sparse`: Compressed weight matrix (2:4 sparse)
/// - `b_dense`: Dense activation matrix (FP16)
///
/// Returns dense FP16 result.
pub fn matmul(
  a_sparse: SparseTensor,
  b_dense: ffi.CudaTensor16Ref,
  m: Int,
  n: Int,
  k: Int,
) -> Result(ffi.CudaTensor16Ref, String) {
  ffi.sparse_matmul(a_sparse, b_dense, m, n, k)
}
