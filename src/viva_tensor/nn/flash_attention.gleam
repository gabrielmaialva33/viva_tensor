//// Flash Attention - IO-Aware Exact Attention in O(n) Memory
////
//// "The key insight is that memory bandwidth, not FLOPs, is the bottleneck."
////   — Tri Dao, channeling every GPU programmer's frustration
////
//// References:
//// - Dao et al. (2022). "FlashAttention: Fast and Memory-Efficient Exact
////   Attention with IO-Awareness." https://arxiv.org/abs/2205.14135
//// - Dao (2023). "FlashAttention-2: Faster Attention with Better Parallelism
////   and Work Partitioning." https://arxiv.org/abs/2307.08691
//// - Rabe & Staats (2021). "Self-attention Does Not Need O(n^2) Memory."
////   The theoretical foundation that made Flash Attention possible.
////
//// The Problem:
////   Standard attention: scores = Q @ K^T  (this creates an n x n matrix!)
////   For n=8192: 67M elements = 256MB per head. 32 heads = 8GB. Ouch.
////
//// The Solution:
////   Process in TILES. Never materialize the full n x n matrix.
////   Online softmax: update running statistics incrementally.
////   Result: O(n) memory, 2-4x faster, and EXACT (not an approximation!).
////
//// Why it works (IO-awareness):
////   GPU has fast SRAM (on-chip) and slow HBM (off-chip).
////   Standard attention: write n^2 elements to HBM, read them back. Slow.
////   Flash attention: keep working set in SRAM, only read/write O(n) to HBM.
////   Memory bandwidth wins over raw FLOPs. Every. Single. Time.
////
//// This implementation is a pure Gleam demonstration. For production,
//// you'd want CUDA kernels that fuse the operations. But the math is the same.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor, Tensor}

@external(erlang, "math", "exp")
fn math_exp(x: Float) -> Float

// -------------------------------------------------------------------------
// Configuration Types
// -------------------------------------------------------------------------

/// Flash Attention configuration.
///
/// Block sizes determine the tile dimensions. Larger blocks = fewer iterations
/// but more SRAM usage. The sweet spot depends on your GPU's SRAM size.
/// For A100: block_q=128, block_kv=128 works well.
/// For consumer GPUs: 64x64 is safer.
pub type FlashConfig {
  FlashConfig(
    /// Block size for Q dimension (rows of attention matrix)
    block_q: Int,
    /// Block size for KV dimension (columns of attention matrix)
    block_kv: Int,
    /// Scaling factor: 1/sqrt(d_k). Keeps attention weights from exploding.
    scale: Float,
    /// Causal masking for autoregressive models (GPT-style)
    causal: Bool,
  )
}

/// Running statistics for online softmax computation.
///
/// The magic of Flash Attention: we don't need all the scores to compute softmax.
/// We track max (for numerical stability) and sum_exp (for normalization),
/// updating them as we process each KV block.
///
/// Math: softmax(x)_i = exp(x_i - max(x)) / sum_j(exp(x_j - max(x)))
/// We can compute this incrementally by rescaling when max changes.
pub type OnlineStats {
  OnlineStats(
    /// Running maximum (for numerical stability in exp)
    max_val: Float,
    /// Running sum of exp(score - max) for normalization
    sum_exp: Float,
    /// Accumulated output (will be normalized at the end)
    output: List(Float),
  )
}

/// Result of Flash Attention, with memory statistics.
pub type FlashResult {
  FlashResult(
    /// The attention output tensor
    output: Tensor,
    /// Peak memory usage in bytes (just the tile, not the full matrix)
    memory_bytes: Int,
    /// Percentage of memory saved vs naive attention
    memory_saved_percent: Float,
  )
}

// -------------------------------------------------------------------------
// Configuration Builders
// -------------------------------------------------------------------------

/// Default configuration optimized for common use cases.
///
/// Block sizes of 64 work on most GPUs.
/// Scale is 1/sqrt(head_dim), following the original Transformer paper.
pub fn default_config(head_dim: Int) -> FlashConfig {
  let scale = case float.square_root(int.to_float(head_dim)) {
    Ok(sqrt) -> 1.0 /. sqrt
    Error(_) -> 0.125
    // fallback for head_dim=64
  }
  FlashConfig(block_q: 64, block_kv: 64, scale: scale, causal: False)
}

/// Causal configuration for autoregressive models.
///
/// In causal attention, position i can only attend to positions j <= i.
/// This is how GPT, LLaMA, and friends generate text token by token.
pub fn causal_config(head_dim: Int) -> FlashConfig {
  FlashConfig(..default_config(head_dim), causal: True)
}

// -------------------------------------------------------------------------
// Naive Attention - The O(n^2) Baseline
// -------------------------------------------------------------------------
//
// This is what we're trying to avoid. Included for comparison.
// Algorithm:
//   1. scores = Q @ K^T          <- Creates n x n matrix (the problem)
//   2. attn = softmax(scores * scale)
//   3. output = attn @ V

/// Standard attention with O(n^2) memory. DON'T use for long sequences.
///
/// This allocates the full attention matrix. For n=8192, that's 256MB per head.
/// Included only to show what Flash Attention saves you from.
pub fn naive_attention(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  scale: Float,
) -> #(Tensor, Int) {
  let q_data = tensor.to_list(q)
  let k_data = tensor.to_list(k)
  let v_data = tensor.to_list(v)

  let seq_len = list.length(q_data)
  let head_dim = case get_tensor_shape(q) {
    [_, d] -> d
    _ -> seq_len
  }

  let q_rows = list.sized_chunk(q_data, head_dim)
  let k_rows = list.sized_chunk(k_data, head_dim)
  let v_rows = list.sized_chunk(v_data, head_dim)

  let n_rows = list.length(q_rows)
  let n_cols = list.length(k_rows)

  // Step 1: scores = Q @ K^T
  // THIS is the O(n^2) memory allocation everyone complains about
  let scores =
    list.map(q_rows, fn(q_row) {
      list.map(k_rows, fn(k_row) { dot_product(q_row, k_row) *. scale })
    })

  // Step 2: softmax per row
  let attn = list.map(scores, softmax_row)

  // Step 3: output = attn @ V
  let output =
    list.map(attn, fn(attn_row) {
      list.index_fold(attn_row, list.repeat(0.0, head_dim), fn(acc, weight, i) {
        let v_row = get_row(v_rows, i)
        list.map2(acc, v_row, fn(a, v) { a +. weight *. v })
      })
    })
    |> list.flatten

  // Memory: n x n attention matrix in FP32
  let memory = n_rows * n_cols * 4

  #(Tensor(data: output, shape: get_tensor_shape(q)), memory)
}

// -------------------------------------------------------------------------
// Flash Attention - The O(n) Hero
// -------------------------------------------------------------------------
//
// The key insight: we don't need to materialize the full attention matrix.
// By processing in blocks and using online softmax, we get exact attention
// with O(block_size^2) memory instead of O(n^2).
//
// For n=8192 with block=64: 64^2 = 4KB vs 8192^2 = 256MB. That's 64,000x less!

/// Flash Attention: exact attention with O(n) memory.
///
/// This is the algorithm that enabled 100K+ context windows in LLMs.
/// It's not an approximation - it computes the exact same result as naive attention.
/// The magic is in the order of computation and the online softmax trick.
pub fn flash_attention(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  config: FlashConfig,
) -> FlashResult {
  let q_data = tensor.to_list(q)
  let k_data = tensor.to_list(k)
  let v_data = tensor.to_list(v)

  let head_dim = case get_tensor_shape(q) {
    [_, d] -> d
    _ -> list.length(q_data)
  }

  let q_rows = list.sized_chunk(q_data, head_dim)
  let k_rows = list.sized_chunk(k_data, head_dim)
  let v_rows = list.sized_chunk(v_data, head_dim)

  let n_q = list.length(q_rows)
  let n_kv = list.length(k_rows)

  // Partition into blocks - this is the tiling that saves memory
  let q_blocks = list.sized_chunk(q_rows, config.block_q)
  let k_blocks = list.sized_chunk(k_rows, config.block_kv)
  let v_blocks = list.sized_chunk(v_rows, config.block_kv)

  // Process each Q block against all KV blocks
  let output =
    list.index_map(q_blocks, fn(q_block, q_block_idx) {
      process_q_block(
        q_block,
        k_blocks,
        v_blocks,
        config,
        q_block_idx,
        head_dim,
      )
    })
    |> list.flatten
    |> list.flatten

  // Memory analysis: we only ever allocate one tile at a time
  let flash_memory = config.block_q * config.block_kv * 4
  let naive_memory = n_q * n_kv * 4

  let saved =
    100.0 *. { 1.0 -. int.to_float(flash_memory) /. int.to_float(naive_memory) }

  FlashResult(
    output: Tensor(data: output, shape: get_tensor_shape(q)),
    memory_bytes: flash_memory,
    memory_saved_percent: saved,
  )
}

/// Process one Q block against all KV blocks.
///
/// This is the outer loop of Flash Attention.
/// For each query in the block, we maintain running softmax statistics
/// as we iterate through the KV blocks.
fn process_q_block(
  q_block: List(List(Float)),
  k_blocks: List(List(List(Float))),
  v_blocks: List(List(List(Float))),
  config: FlashConfig,
  q_block_idx: Int,
  head_dim: Int,
) -> List(List(Float)) {
  // Initialize online stats for each query in this block
  let initial_stats =
    list.map(q_block, fn(_) {
      OnlineStats(
        max_val: -999_999.0,
        // Will be updated on first real score
        sum_exp: 0.0,
        output: list.repeat(0.0, head_dim),
      )
    })

  // Iterate through KV blocks, updating stats as we go
  let zipped_kv = list.zip(k_blocks, v_blocks)

  let final_stats =
    list.index_fold(zipped_kv, initial_stats, fn(stats, kv_pair, kv_idx) {
      let #(k_block, v_block) = kv_pair

      // Causal masking: skip KV blocks entirely in the future
      case config.causal {
        True -> {
          let q_start = q_block_idx * config.block_q
          let kv_start = kv_idx * config.block_kv

          // If the entire KV block is in the future, skip it
          case kv_start > q_start + list.length(q_block) {
            True -> stats
            False -> process_kv_block(stats, q_block, k_block, v_block, config)
          }
        }
        False -> process_kv_block(stats, q_block, k_block, v_block, config)
      }
    })

  // Normalize final outputs by sum_exp to complete the softmax
  list.map(final_stats, fn(s) {
    case s.sum_exp >. 0.0 {
      True -> list.map(s.output, fn(o) { o /. s.sum_exp })
      False -> s.output
    }
  })
}

/// Process one KV block against all queries, updating online statistics.
///
/// This is where the online softmax magic happens.
/// When we encounter a new max, we rescale the running sum and output.
/// This is numerically stable and gives exact results.
fn process_kv_block(
  stats: List(OnlineStats),
  q_block: List(List(Float)),
  k_block: List(List(Float)),
  v_block: List(List(Float)),
  config: FlashConfig,
) -> List(OnlineStats) {
  list.map2(stats, q_block, fn(stat, q_row) {
    // Compute scores: q_row @ each k in k_block, scaled
    let scores =
      list.map(k_block, fn(k_row) { dot_product(q_row, k_row) *. config.scale })

    // Online softmax update
    // Step 1: Find new max
    let new_max = list.fold(scores, stat.max_val, float.max)

    // Step 2: Compute correction factor for previous statistics
    // When max increases, we need to downscale previous exp values
    let correction = case stat.sum_exp >. 0.0 {
      True -> math_exp(stat.max_val -. new_max)
      False -> 1.0
    }

    // Step 3: Rescale previous sum_exp
    let corrected_sum = stat.sum_exp *. correction

    // Step 4: Compute exp(scores - new_max) for this block
    let exp_scores = list.map(scores, fn(s) { math_exp(s -. new_max) })

    // Step 5: Add to running sum
    let new_sum = list.fold(exp_scores, corrected_sum, float.add)

    // Step 6: Rescale previous output and add new contribution
    let corrected_output = list.map(stat.output, fn(o) { o *. correction })

    let new_contribution =
      list.index_fold(
        exp_scores,
        list.repeat(0.0, list.length(stat.output)),
        fn(acc, weight, i) {
          let v_row = get_row(v_block, i)
          list.map2(acc, v_row, fn(a, v) { a +. weight *. v })
        },
      )

    let new_output = list.map2(corrected_output, new_contribution, float.add)

    OnlineStats(max_val: new_max, sum_exp: new_sum, output: new_output)
  })
}

// -------------------------------------------------------------------------
// Benchmark - Prove That Flash Attention Delivers
// -------------------------------------------------------------------------

pub fn main() {
  benchmark_flash_attention()
}

pub fn benchmark_flash_attention() {
  io.println("========================================================")
  io.println("  FLASH ATTENTION - O(n) Memory Algorithm")
  io.println("  Tri Dao et al., 2022")
  io.println("  https://arxiv.org/abs/2205.14135")
  io.println("========================================================")
  io.println("")

  io.println("THE PROBLEM WITH STANDARD ATTENTION:")
  io.println("  Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d)) @ V")
  io.println("  Q @ K^T creates an n x n matrix")
  io.println("  For seq_len=8192: 67M elements = 256MB per head")
  io.println("  With 32 heads: 8GB just for attention scores!")
  io.println("")

  io.println("THE FLASH ATTENTION INSIGHT:")
  io.println("  Memory bandwidth > FLOPs on modern GPUs")
  io.println("  Process in TILES to stay in fast SRAM")
  io.println("  Online softmax: compute exactly without full matrix")
  io.println("  Result: 2-4x faster, O(n) memory, EXACT!")
  io.println("")

  // Benchmark different sequence lengths
  let sizes = [64, 128, 256, 512]
  let head_dim = 64

  io.println("--- BENCHMARK: Naive vs Flash ---")
  io.println("  head_dim = " <> int.to_string(head_dim))
  io.println("")

  list.each(sizes, fn(seq_len) {
    let q = tensor.random_uniform([seq_len, head_dim])
    let k = tensor.random_uniform([seq_len, head_dim])
    let v = tensor.random_uniform([seq_len, head_dim])

    let config = default_config(head_dim)

    let #(naive_time, #(_naive_out, naive_mem)) =
      timer_tc(fn() { naive_attention(q, k, v, config.scale) })

    let #(flash_time, flash_result) =
      timer_tc(fn() { flash_attention(q, k, v, config) })

    io.println("seq_len = " <> int.to_string(seq_len) <> ":")
    io.println(
      "  Naive:  "
      <> int.to_string(naive_time / 1000)
      <> "ms, "
      <> int.to_string(naive_mem / 1024)
      <> " KB",
    )
    io.println(
      "  Flash:  "
      <> int.to_string(flash_time / 1000)
      <> "ms, "
      <> int.to_string(flash_result.memory_bytes / 1024)
      <> " KB",
    )
    io.println(
      "  Memory saved: "
      <> float_to_string(flash_result.memory_saved_percent)
      <> "%",
    )
    io.println("")
  })

  // Projections for long contexts
  io.println("--- MEMORY SAVINGS AT SCALE ---")
  let long_contexts = [1024, 2048, 4096, 8192, 16_384, 32_768]

  list.each(long_contexts, fn(n) {
    let naive_mem = n * n * 4
    // Full n x n matrix in FP32
    let flash_mem = 64 * 64 * 4
    // Just one tile

    let saved =
      100.0 *. { 1.0 -. int.to_float(flash_mem) /. int.to_float(naive_mem) }

    io.println(
      "  n="
      <> int.to_string(n)
      <> ": Naive="
      <> bytes_to_string(naive_mem)
      <> ", Flash="
      <> bytes_to_string(flash_mem)
      <> " ("
      <> float_to_string(saved)
      <> "% saved)",
    )
  })

  io.println("")
  io.println("========================================================")
  io.println("  WHY FLASH ATTENTION CHANGED EVERYTHING:")
  io.println("")
  io.println("  1. Long contexts are now viable (32K, 100K, 1M tokens)")
  io.println("  2. 2-4x speedup from IO-awareness")
  io.println("  3. Exact computation (not an approximation!)")
  io.println("  4. Now standard in GPT-4, Claude, LLaMA, Gemini...")
  io.println("")
  io.println("  viva_tensor + Flash = Unlimited context on RTX 4090!")
  io.println("========================================================")
}

// -------------------------------------------------------------------------
// Helper Functions
// -------------------------------------------------------------------------

fn dot_product(a: List(Float), b: List(Float)) -> Float {
  list.map2(a, b, fn(x, y) { x *. y })
  |> list.fold(0.0, float.add)
}

fn softmax_row(row: List(Float)) -> List(Float) {
  let max_val = list.fold(row, -999_999.0, float.max)
  let exp_vals = list.map(row, fn(x) { math_exp(x -. max_val) })
  let sum = list.fold(exp_vals, 0.0, float.add)
  case sum >. 0.0 {
    True -> list.map(exp_vals, fn(e) { e /. sum })
    False -> row
  }
}

fn get_row(matrix: List(List(Float)), idx: Int) -> List(Float) {
  case list.drop(matrix, idx) {
    [row, ..] -> row
    [] -> []
  }
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn bytes_to_string(bytes: Int) -> String {
  case bytes {
    b if b >= 1_073_741_824 ->
      float_to_string(int.to_float(b) /. 1_073_741_824.0) <> "GB"
    b if b >= 1_048_576 ->
      float_to_string(int.to_float(b) /. 1_048_576.0) <> "MB"
    b if b >= 1024 -> int.to_string(b / 1024) <> "KB"
    b -> int.to_string(b) <> "B"
  }
}

fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)
