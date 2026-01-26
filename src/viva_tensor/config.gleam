//// Auto-Backend Selector - Smart tensor backend selection
////
//// Automatically chooses optimal backend (list vs strided) based on:
//// - Operation type (sequential, random access, matrix)
//// - Tensor size
//// - User configuration
////
//// Benchmarked on RTX 4090 + BEAM VM

import viva_tensor/tensor.{type Tensor, StridedTensor, Tensor}

// =============================================================================
// TYPES
// =============================================================================

/// Operation type for backend selection
pub type OperationType {
  /// Sequential operations - better with lists (dot, sum, reduce)
  Sequential
  /// Random access operations - better with strided (get, get2d, indexing)
  RandomAccess
  /// Matrix operations - strided for large, lists for small (matmul)
  MatrixOp
}

/// Configuration for automatic backend selection
pub type TensorConfig {
  TensorConfig(
    /// Minimum size to use strided for random access (default: 500)
    strided_threshold_random: Int,
    /// Minimum size to use strided for matmul (default: 64 = 8x8)
    strided_threshold_matmul: Int,
    /// Force strided for all operations (override)
    force_strided: Bool,
    /// Force list for all operations (override)
    force_list: Bool,
  )
}

// =============================================================================
// PRESETS
// =============================================================================

/// Default configuration based on benchmarks
pub fn default_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 500,
    strided_threshold_matmul: 64,
    force_strided: False,
    force_list: False,
  )
}

/// High-performance config (prefer strided for large tensors)
pub fn performance_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 100,
    strided_threshold_matmul: 32,
    force_strided: False,
    force_list: False,
  )
}

/// Memory-efficient config (prefer lists)
pub fn memory_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 5000,
    strided_threshold_matmul: 256,
    force_strided: False,
    force_list: False,
  )
}

/// GPU-optimized config (always strided for batched ops)
pub fn gpu_config() -> TensorConfig {
  TensorConfig(
    strided_threshold_random: 64,
    strided_threshold_matmul: 16,
    force_strided: False,
    force_list: False,
  )
}

// =============================================================================
// BACKEND SELECTION
// =============================================================================

/// Check if should use strided backend for given operation
pub fn should_use_strided(
  t: Tensor,
  op: OperationType,
  config: TensorConfig,
) -> Bool {
  case config.force_strided, config.force_list {
    True, _ -> True
    _, True -> False
    False, False -> {
      let tensor_size = tensor.size(t)

      case op {
        // Sequential ops (dot, sum) - NEVER use strided (0.7x slower)
        Sequential -> False

        // Random access (get, get2d) - strided if large enough
        RandomAccess -> tensor_size >= config.strided_threshold_random

        // Matrix ops - strided if matrices are large enough
        MatrixOp -> {
          let shape = get_tensor_shape(t)
          case shape {
            [rows, cols] -> rows * cols >= config.strided_threshold_matmul
            _ -> tensor_size >= config.strided_threshold_matmul
          }
        }
      }
    }
  }
}

/// Get shape from tensor (helper for pattern matching)
fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    StridedTensor(_, shape, _, _) -> shape
  }
}

/// Ensure tensor is in optimal format for operation
pub fn ensure_optimal(
  t: Tensor,
  op: OperationType,
  config: TensorConfig,
) -> Tensor {
  let use_strided = should_use_strided(t, op, config)

  case t, use_strided {
    // Already strided and should be strided - keep it
    StridedTensor(..), True -> t

    // Already list and should be list - keep it
    Tensor(..), False -> t

    // Need to convert to strided
    Tensor(..), True -> tensor.to_strided(t)

    // Need to convert to list
    StridedTensor(..), False -> tensor.to_contiguous(t)
  }
}

/// Auto-optimize tensor for matmul
pub fn for_matmul(t: Tensor) -> Tensor {
  ensure_optimal(t, MatrixOp, default_config())
}

/// Auto-optimize tensor for reduction operations
pub fn for_reduction(t: Tensor) -> Tensor {
  ensure_optimal(t, Sequential, default_config())
}

/// Auto-optimize tensor for indexing
pub fn for_indexing(t: Tensor) -> Tensor {
  ensure_optimal(t, RandomAccess, default_config())
}
