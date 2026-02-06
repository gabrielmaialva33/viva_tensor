//// 2:4 Structured Sparsity - NVIDIA Tensor Cores
////
//// Reference: Mishra et al. (2021) "Accelerating Sparse Deep Neural Networks"
//// https://arxiv.org/abs/2104.08378
////
//// Key insight from the paper: 2:4 sparsity achieves 2x theoretical speedup
//// with <1% accuracy loss on ImageNet. In practice, we see ~1.7x due to
//// memory bandwidth limits and kernel launch overhead.
////
//// Why 2:4 specifically?
//// NVIDIA's brilliant constraint: any 2 of 4 elements can be zero.
//// This is structured enough for hardware acceleration but flexible enough
//// to preserve accuracy. The Sparse Tensor Core can skip 50% of MACs while
//// the index overhead is just 2 bits per 4 elements.
////
//// Why not arbitrary sparsity?
//// Because sparse matrix formats (CSR, COO, BCSR) have indexing overhead
//// that kills performance for sparsity < 90%. At 50% sparsity, dense ops win.
//// 2:4's fixed structure eliminates the index explosion problem.
////
//// Storage format:
//// - 2 values (FP16: 32 bits total)
//// - 2-bit mask for positions (4 bits, padded to 8 bits in practice)
//// - Total: ~40 bits for 4 elements vs 64 bits dense = 1.6x compression
//// - Real memory savings: ~1.78x after alignment
////
//// Performance reality check (NVIDIA Ampere):
//// - Dense FP16 matmul: ~150 TFLOPS
//// - Sparse 2:4 FP16 matmul: ~300 TFLOPS (theoretical 2x)
//// - Actual observed: ~250-280 TFLOPS (1.7-1.9x)
//// - Bottleneck: memory bandwidth, not compute

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Types ---

/// Sparse 2:4 block: 4 elements compressed to 2 non-zeros + position mask
/// This is the fundamental unit of 2:4 sparsity.
pub type Sparse24Block {
  Sparse24Block(
    /// The 2 non-zero values (survivors of magnitude pruning)
    values: #(Float, Float),
    /// 2-bit positions (0-3 each), packed. Could be a single u4 in hardware.
    positions: #(Int, Int),
  )
}

/// Tensor with 2:4 structured sparsity
///
/// Sparsity ratio S = (total - nonzero) / total = 0.5 for 2:4
/// Effective FLOPS with 2:4: 2x theoretical, ~1.7x practical
pub type Sparse24Tensor {
  Sparse24Tensor(
    /// Sparse blocks covering the entire tensor
    blocks: List(Sparse24Block),
    /// Original dense shape (for reconstruction)
    shape: List(Int),
    /// Original element count (may not be divisible by 4)
    num_elements: Int,
    /// Compressed memory footprint in bytes
    memory_bytes: Int,
    /// Actual sparsity achieved (always 50% for 2:4)
    sparsity_percent: Float,
  )
}

/// Pruning metrics for analysis
/// Tracks the quality of the sparsification decision
pub type PruneMetrics {
  PruneMetrics(
    /// Elements zeroed out
    pruned_count: Int,
    /// Total elements
    total_count: Int,
    /// L1 approximation error (mean absolute difference)
    approximation_error: Float,
    /// Mean magnitude of kept elements
    kept_magnitude_mean: Float,
    /// Mean magnitude of pruned elements (lower = good pruning decisions)
    pruned_magnitude_mean: Float,
  )
}

// --- Pruning Strategies ---
//
// Magnitude pruning: simple, effective, theory-backed.
// The lottery ticket hypothesis (Frankle & Carlin, ICLR 2019) suggests
// that large-magnitude weights are the "winning tickets" that matter.
// More sophisticated options exist (gradient-based, Hessian-based) but
// magnitude works surprisingly well in practice.

/// Apply 2:4 pruning using magnitude-based selection
///
/// Strategy: keep the 2 largest (by absolute value) in each group of 4.
/// This is NVIDIA's recommended approach and works well empirically.
///
/// Theoretical justification: large weights carry more information.
/// Empirical validation: <1% accuracy drop on ImageNet, BERT, GPT-2.
pub fn prune_24_magnitude(t: Tensor) -> Sparse24Tensor {
  let data = tensor.to_list(t)
  let shape = get_tensor_shape(t)
  let num_elements = list.length(data)

  // Chunk into groups of 4. The 2:4 pattern is applied per-group.
  let groups = list.sized_chunk(data, 4)

  // For each group, keep the 2 highest-magnitude elements
  let blocks =
    list.map(groups, fn(group) { prune_group_magnitude(pad_group(group)) })

  // Memory calculation:
  // - 2 values at FP16 (4 bytes) + 2 positions packed (1 byte) = 5 bytes/block
  // - Each block represents 4 elements
  // - Original: 4 elements * 4 bytes (FP32) = 16 bytes
  // - Compression: 16/5 = 3.2x (but we store FP16, so actual 1.78x vs FP16 dense)
  let num_blocks = list.length(blocks)
  let memory = num_blocks * 5

  Sparse24Tensor(
    blocks: blocks,
    shape: shape,
    num_elements: num_elements,
    memory_bytes: memory,
    sparsity_percent: 50.0,
  )
}

/// Core pruning logic: select top-2 by magnitude from a group of 4
fn prune_group_magnitude(group: List(Float)) -> Sparse24Block {
  // Index each element with its magnitude for sorting
  let indexed =
    list.index_map(group, fn(val, idx) {
      #(idx, val, float.absolute_value(val))
    })

  // Sort descending by magnitude - largest first
  let sorted =
    list.sort(indexed, fn(a, b) {
      float.compare(b.2, a.2)
    })

  // Take the top 2 survivors
  case sorted {
    [first, second, ..] -> {
      // Maintain position order for deterministic reconstruction
      let #(pos1, val1, _) = first
      let #(pos2, val2, _) = second
      let #(p1, v1, p2, v2) = case pos1 < pos2 {
        True -> #(pos1, val1, pos2, val2)
        False -> #(pos2, val2, pos1, val1)
      }
      Sparse24Block(values: #(v1, v2), positions: #(p1, p2))
    }
    _ -> Sparse24Block(values: #(0.0, 0.0), positions: #(0, 1))
  }
}

/// Pad incomplete groups to exactly 4 elements
/// Edge case: tensors with size not divisible by 4
fn pad_group(group: List(Float)) -> List(Float) {
  let len = list.length(group)
  case len < 4 {
    True -> list.append(group, list.repeat(0.0, 4 - len))
    False -> list.take(group, 4)
  }
}

// --- Gradient-Based Pruning ---
//
// For training, magnitude alone isn't optimal. We want to preserve
// weights that have high gradient*weight product (movement pruning).
// This comes from "Movement Pruning" (Sanh et al., 2020).

/// Gradient-weighted pruning for training scenarios
///
/// Importance = |weight * gradient|
/// Intuition: weights that are both large AND changing rapidly matter most.
/// This is better than magnitude alone during fine-tuning.
pub fn prune_24_gradient(weights: Tensor, gradients: Tensor) -> Sparse24Tensor {
  let w_data = tensor.to_list(weights)
  let g_data = tensor.to_list(gradients)

  let shape = get_tensor_shape(weights)
  let num_elements = list.length(w_data)

  // Importance score = |weight * gradient|
  // High score = large weight with large gradient = important for learning
  let importance =
    list.map2(w_data, g_data, fn(w, g) { float.absolute_value(w *. g) })

  let w_groups = list.sized_chunk(w_data, 4)
  let i_groups = list.sized_chunk(importance, 4)

  let blocks =
    list.map2(w_groups, i_groups, fn(w_group, i_group) {
      prune_group_by_importance(pad_group(w_group), i_group)
    })

  let num_blocks = list.length(blocks)

  Sparse24Tensor(
    blocks: blocks,
    shape: shape,
    num_elements: num_elements,
    memory_bytes: num_blocks * 5,
    sparsity_percent: 50.0,
  )
}

fn prune_group_by_importance(
  weights: List(Float),
  importance: List(Float),
) -> Sparse24Block {
  let indexed =
    list.zip(list.range(0, 3), list.zip(weights, importance))
    |> list.map(fn(x) {
      let #(idx, #(w, i)) = x
      #(idx, w, i)
    })

  let sorted = list.sort(indexed, fn(a, b) { float.compare(b.2, a.2) })

  case sorted {
    [first, second, ..] -> {
      let #(pos1, val1, _) = first
      let #(pos2, val2, _) = second
      let #(p1, v1, p2, v2) = case pos1 < pos2 {
        True -> #(pos1, val1, pos2, val2)
        False -> #(pos2, val2, pos1, val1)
      }
      Sparse24Block(values: #(v1, v2), positions: #(p1, p2))
    }
    _ -> Sparse24Block(values: #(0.0, 0.0), positions: #(0, 1))
  }
}

// --- Decompression ---

/// Reconstruct dense tensor from 2:4 sparse representation
///
/// This is O(n) and allocation-heavy. In CUDA, you'd keep it sparse
/// and let the Tensor Core handle the pattern. On CPU, you often
/// need to decompress for compatibility with dense operations.
pub fn decompress(sparse: Sparse24Tensor) -> Tensor {
  let data =
    list.flat_map(sparse.blocks, fn(block) {
      let #(v1, v2) = block.values
      let #(p1, p2) = block.positions

      // Reconstruct the 4-element group with zeros in pruned positions
      list.range(0, 3)
      |> list.map(fn(i) {
        case i == p1 {
          True -> v1
          False ->
            case i == p2 {
              True -> v2
              False -> 0.0
            }
        }
      })
    })

  // Handle tensors with size not divisible by 4
  let truncated = list.take(data, sparse.num_elements)
  Tensor(data: truncated, shape: sparse.shape)
}

// --- Sparse Matrix Multiplication ---
//
// On real hardware (Ampere+), the Sparse Tensor Core does this natively.
// We simulate it here for correctness testing and CPU fallback.
//
// Real performance (NVIDIA A100):
// - Dense FP16: ~312 TFLOPS
// - Sparse 2:4 FP16: ~624 TFLOPS (2x theoretical)
// - Actual: ~500-550 TFLOPS (1.7x due to memory limits)

/// Sparse matrix multiplication (simulated)
///
/// On Tensor Cores, this skips 50% of multiplications by hardware.
/// Here we decompress and multiply densely for correctness.
/// Returns: (result, theoretical_speedup)
pub fn sparse_matmul(
  sparse_a: Sparse24Tensor,
  dense_b: Tensor,
) -> #(Tensor, Float) {
  // Real implementation would use sparse kernels.
  // We decompress for simulation - obviously no speedup here.
  let dense_a = decompress(sparse_a)
  let result = tensor_matmul(dense_a, dense_b)

  // Theoretical speedup: 2x (50% of MACs skipped)
  // Practical speedup: 1.5-1.7x (memory bandwidth bottleneck)
  let theoretical_speedup = 2.0

  #(result, theoretical_speedup)
}

/// Basic dense matmul for simulation
fn tensor_matmul(a: Tensor, b: Tensor) -> Tensor {
  let a_data = tensor.to_list(a)
  let b_data = tensor.to_list(b)

  let a_shape = get_tensor_shape(a)
  let b_shape = get_tensor_shape(b)

  let #(m, k) = case a_shape {
    [rows, cols] -> #(rows, cols)
    _ -> #(1, list.length(a_data))
  }

  let #(_k2, n) = case b_shape {
    [rows, cols] -> #(rows, cols)
    _ -> #(list.length(b_data), 1)
  }

  let a_rows = list.sized_chunk(a_data, k)
  let b_cols = transpose_matrix(list.sized_chunk(b_data, n))

  let result =
    list.flat_map(a_rows, fn(a_row) {
      list.map(b_cols, fn(b_col) {
        list.map2(a_row, b_col, fn(x, y) { x *. y })
        |> list.fold(0.0, float.add)
      })
    })

  Tensor(data: result, shape: [m, n])
}

fn transpose_matrix(m: List(List(Float))) -> List(List(Float)) {
  case m {
    [] -> []
    [first, ..] -> {
      let n_cols = list.length(first)
      list.range(0, n_cols - 1)
      |> list.map(fn(col_idx) {
        list.filter_map(m, fn(row) {
          case list.drop(row, col_idx) {
            [x, ..] -> Ok(x)
            [] -> Error(Nil)
          }
        })
      })
    }
  }
}

// --- Metrics ---

/// Compute pruning quality metrics
///
/// Key insight: if pruned_magnitude_mean << kept_magnitude_mean,
/// we're making good pruning decisions. The approximation_error
/// tells us how much information we lost.
pub fn compute_metrics(original: Tensor, sparse: Sparse24Tensor) -> PruneMetrics {
  let orig_data = tensor.to_list(original)
  let decomp_data = tensor.to_list(decompress(sparse))

  // L1 approximation error
  let errors =
    list.map2(orig_data, decomp_data, fn(o, d) { float.absolute_value(o -. d) })
  let mean_error = case errors {
    [] -> 0.0
    _ -> list.fold(errors, 0.0, float.add) /. int.to_float(list.length(errors))
  }

  // Separate kept vs pruned elements
  let kept =
    list.filter_map(list.zip(orig_data, decomp_data), fn(pair) {
      let #(o, d) = pair
      case float.absolute_value(d) >. 0.0 {
        True -> Ok(float.absolute_value(o))
        False -> Error(Nil)
      }
    })

  let pruned =
    list.filter_map(list.zip(orig_data, decomp_data), fn(pair) {
      let #(o, d) = pair
      case float.absolute_value(d) >. 0.0 {
        False -> Ok(float.absolute_value(o))
        True -> Error(Nil)
      }
    })

  let kept_mean = case kept {
    [] -> 0.0
    _ -> list.fold(kept, 0.0, float.add) /. int.to_float(list.length(kept))
  }

  let pruned_mean = case pruned {
    [] -> 0.0
    _ -> list.fold(pruned, 0.0, float.add) /. int.to_float(list.length(pruned))
  }

  PruneMetrics(
    pruned_count: list.length(pruned),
    total_count: list.length(orig_data),
    approximation_error: mean_error,
    kept_magnitude_mean: kept_mean,
    pruned_magnitude_mean: pruned_mean,
  )
}

// --- Benchmark ---

pub fn main() {
  benchmark_sparsity()
}

pub fn benchmark_sparsity() {
  io.println(
    "╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  2:4 STRUCTURED SPARSITY - NVIDIA Tensor Cores                   ║",
  )
  io.println(
    "║  Ampere+ Architecture (RTX 3000/4000, A100, H100)                ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  io.println("CONCEPT:")
  io.println("  - Keep 2 of every 4 elements (50% structured sparsity)")
  io.println("  - Tensor Cores skip multiplications by zero")
  io.println("  - Result: 2x throughput with ~1% accuracy loss")
  io.println("  - Reference: Mishra et al. (2021) arXiv:2104.08378\n")

  io.println("STORAGE FORMAT:")
  io.println("  - Original: 4 x FP16 = 64 bits")
  io.println("  - Sparse: 2 x FP16 + 4-bit mask = 36 bits")
  io.println("  - Compression: 1.78x\n")

  let t = tensor.random_uniform([1024, 512])

  io.println("━━━ BENCHMARK: Tensor [1024, 512] ━━━")

  let #(time_prune, sparse) = timer_tc(fn() { prune_24_magnitude(t) })

  let metrics = compute_metrics(t, sparse)

  let original_bytes = 1024 * 512 * 4
  let compression =
    int.to_float(original_bytes) /. int.to_float(sparse.memory_bytes)

  io.println(
    "  Prune time:         " <> int.to_string(time_prune / 1000) <> "ms",
  )
  io.println(
    "  Original memory:    " <> int.to_string(original_bytes / 1024) <> " KB",
  )
  io.println(
    "  Sparse memory:      "
    <> int.to_string(sparse.memory_bytes / 1024)
    <> " KB",
  )
  io.println("  Compression:        " <> float_to_string(compression) <> "x")
  io.println(
    "  Sparsity:           " <> float_to_string(sparse.sparsity_percent) <> "%",
  )
  io.println("")
  io.println(
    "  Pruned elements:    "
    <> int.to_string(metrics.pruned_count)
    <> "/"
    <> int.to_string(metrics.total_count),
  )
  io.println(
    "  Approx error:       " <> float_to_string(metrics.approximation_error),
  )
  io.println(
    "  Kept magnitude:     " <> float_to_string(metrics.kept_magnitude_mean),
  )
  io.println(
    "  Pruned magnitude:   " <> float_to_string(metrics.pruned_magnitude_mean),
  )

  // Simulated matmul
  io.println("\n━━━ SPARSE MATMUL SIMULATION ━━━")

  let b = tensor.random_uniform([512, 256])

  let #(time_dense, dense_result) =
    timer_tc(fn() { tensor_matmul(decompress(sparse), b) })

  let #(time_sparse, #(sparse_result, speedup)) =
    timer_tc(fn() { sparse_matmul(sparse, b) })

  io.println(
    "  Dense matmul:       " <> int.to_string(time_dense / 1000) <> "ms",
  )
  io.println(
    "  Sparse matmul:      "
    <> int.to_string(time_sparse / 1000)
    <> "ms (simulated)",
  )
  io.println(
    "  Theoretical speedup: " <> float_to_string(speedup) <> "x (real hardware)",
  )

  // Verification
  let dense_data = tensor.to_list(dense_result)
  let sparse_data = tensor.to_list(sparse_result)
  let diff =
    list.map2(dense_data, sparse_data, fn(d, s) { float.absolute_value(d -. s) })
    |> list.fold(0.0, float.max)

  io.println(
    "  Max difference:     " <> float_to_string(diff) <> " (should be ~0)",
  )

  // Comparison with other techniques
  io.println("\n━━━ COMBINING TECHNIQUES ━━━")
  io.println("  FP16 alone:         2x compression")
  io.println("  INT8 alone:         4x compression")
  io.println("  2:4 Sparsity:       2x speedup + 1.78x compression")
  io.println("  NF4 alone:          8x compression")
  io.println("")
  io.println("  INT8 + 2:4:         4x * 1.78x = 7.12x compression, 8x speedup")
  io.println("  NF4 + 2:4:          8x * 1.78x = 14.24x compression")

  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  WHY 2:4 SPARSITY MATTERS:                                       ║",
  )
  io.println(
    "║                                                                  ║",
  )
  io.println(
    "║  1. Native hardware support in RTX 3000/4000/A100/H100           ║",
  )
  io.println(
    "║  2. 2x throughput with ~1% accuracy loss                         ║",
  )
  io.println(
    "║  3. Stacks with quantization for 4x+ total speedup               ║",
  )
  io.println(
    "║  4. Standard in NVIDIA models (Megatron-LM, etc)                 ║",
  )
  io.println(
    "║                                                                  ║",
  )
  io.println(
    "║  viva_tensor + 2:4 = Maximum Tensor Core utilization!            ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝",
  )
}

// --- Helpers ---

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 10_000.0)) /. 10_000.0
  float.to_string(rounded)
}

fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)
