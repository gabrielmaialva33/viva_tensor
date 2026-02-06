//// AWQ (Activation-aware Weight Quantization)
////
//// Reference: Lin et al. (2024) - "AWQ: Activation-aware Weight Quantization
//// for LLM Compression and Acceleration"
//// MLSys 2024 BEST PAPER AWARD
//// https://arxiv.org/abs/2306.00978
////
//// --- The Key Insight (worth repeating) ---
//// Only ~1% of weights are "salient" - and they matter 10x more than the rest.
//// But here's the twist: you identify them by looking at ACTIVATIONS, not weights.
//// High activation magnitude = that channel matters = protect those weights.
////
//// --- The Genius ---
//// Don't modify the quantization algorithm. Modify the weights BEFORE quantizing.
//// Scale salient channels UP by s, then scale activations DOWN by 1/s.
//// Mathematically equivalent: W*X = (sW)*(X/s)
//// But now the important weights have more precision after quantization.
////
//// --- Compression Math ---
//// Same as NF4/INT4: 32/4 = 8x theoretical, ~7.7x effective
//// The magic is in the QUALITY, not the ratio.
//// AWQ achieves NF4-level compression with FP16-level accuracy.
////
//// --- Why AWQ Won MLSys 2024 ---
//// 1. Simple insight, huge impact
//// 2. Zero runtime overhead (transform is pre-computed)
//// 3. Works with ANY quantization method (INT4, NF4, whatever)
//// 4. State-of-the-art on LLaMA, OPT, BLOOM benchmarks
////
//// Implementation based on: MIT-HAN Lab + AutoAWQ

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor, Tensor}

// --- AWQ Types ---

/// AWQ configuration
pub type AWQConfig {
  AWQConfig(
    /// Quantization bits (4 is standard, 3 is aggressive)
    bits: Int,
    /// Group size for per-group scaling (128 is typical)
    /// Smaller = more accurate, larger = more compressed
    group_size: Int,
    /// Alpha exponent for scaling: scale = activation_stat ^ alpha
    /// 0.5 is empirically optimal (sqrt of activation magnitude)
    /// Higher alpha = more aggressive protection of salient channels
    alpha: Float,
    /// Use zero-point (asymmetric quantization)
    /// Opinion: Skip it. The cache-miss overhead isn't worth the accuracy gain.
    zero_point: Bool,
  )
}

/// Computed AWQ scales for weight transformation
pub type AWQScales {
  AWQScales(
    /// Per-channel scale factors: multiply weights by these before quantizing
    weight_scales: List(Float),
    /// Original activation statistics (for debugging/analysis)
    activation_stats: List(Float),
    /// Alpha used in computation
    alpha: Float,
  )
}

/// AWQ-quantized tensor
pub type AWQTensor {
  AWQTensor(
    /// Quantized weights (INT4 values)
    quantized_weights: List(Int),
    /// AWQ channel scales (the secret sauce)
    awq_scales: AWQScales,
    /// Per-group quantization scales
    quant_scales: List(Float),
    /// Zero-points if asymmetric (usually empty)
    zero_points: List(Int),
    /// Original shape
    shape: List(Int),
    /// Memory in bytes
    memory_bytes: Int,
  )
}

/// Default AWQ config (matches AutoAWQ defaults)
pub fn default_config() -> AWQConfig {
  AWQConfig(bits: 4, group_size: 128, alpha: 0.5, zero_point: False)
}

// --- Calibration: The Critical Step ---

/// Collect activation statistics from calibration data
///
/// This is THE critical step. Bad calibration = bad quantization.
/// Use 128-512 samples from your actual inference distribution.
/// More samples = more stable statistics, diminishing returns after 256.
///
/// Returns: mean absolute activation per channel
pub fn collect_activation_stats(
  activations_batch: List(List(Float)),
) -> List(Float) {
  case activations_batch {
    [] -> []
    [first, ..] -> {
      let num_channels = list.length(first)

      // Initialize per-channel accumulators
      let initial = list.repeat(0.0, num_channels)

      // Accumulate |activation| per channel across all samples
      let sums =
        list.fold(activations_batch, initial, fn(acc, activation) {
          list.map2(acc, activation, fn(sum, act) {
            sum +. float.absolute_value(act)
          })
        })

      // Mean absolute activation per channel
      let num_samples = int.to_float(list.length(activations_batch))
      list.map(sums, fn(sum) { sum /. num_samples })
    }
  }
}

// --- AWQ Scaling: The Core Algorithm ---

/// Compute AWQ scales from activation statistics
///
/// Formula: scale[i] = activation_stat[i] ^ alpha
///
/// Why alpha = 0.5 (sqrt)?
/// - Too low (0.1): Not enough protection for salient channels
/// - Too high (0.9): Over-protection, wastes precision on outliers
/// - 0.5: Empirically optimal across LLaMA, OPT, BLOOM
pub fn compute_awq_scales(
  activation_stats: List(Float),
  alpha: Float,
) -> AWQScales {
  let weight_scales =
    list.map(activation_stats, fn(stat) {
      // Avoid scale of zero (would lose the channel entirely)
      let safe_stat = case stat >. 0.0 {
        True -> stat
        False -> 1.0
      }
      float_power(safe_stat, alpha)
    })

  AWQScales(
    weight_scales: weight_scales,
    activation_stats: activation_stats,
    alpha: alpha,
  )
}

/// Apply equivalent transformation to weights: W' = W * diag(s)
/// This scales salient channels UP before quantization
pub fn apply_weight_transform(
  weights: List(List(Float)),
  scales: AWQScales,
) -> List(List(Float)) {
  list.map(weights, fn(row) {
    list.map2(row, scales.weight_scales, fn(w, s) { w *. s })
  })
}

/// Apply inverse transformation to activations: X' = X * diag(1/s)
/// This compensates for the weight scaling at runtime
/// Note: In production, fuse this into the previous layer's output
pub fn apply_activation_transform(
  activations: List(Float),
  scales: AWQScales,
) -> List(Float) {
  list.map2(activations, scales.weight_scales, fn(x, s) {
    case s >. 0.0 {
      True -> x /. s
      False -> x
    }
  })
}

// --- Full AWQ Pipeline ---

/// Complete AWQ quantization pipeline
///
/// Steps:
/// 1. Collect activation statistics (calibration)
/// 2. Compute per-channel AWQ scales
/// 3. Transform weights (scale up salient channels)
/// 4. Quantize transformed weights
///
/// At inference:
/// - Use quantized weights directly
/// - Apply inverse activation transform (fused into previous layer)
pub fn quantize_awq(
  weights: Tensor,
  calibration_data: List(List(Float)),
  config: AWQConfig,
) -> AWQTensor {
  let weight_data = tensor.to_list(weights)
  let shape = get_tensor_shape(weights)

  // Assume weights is [out_features, in_features]
  let #(_out_features, in_features) = case shape {
    [o, i] -> #(o, i)
    _ -> #(1, list.length(weight_data))
  }

  // Reshape to matrix
  let weight_matrix = list.sized_chunk(weight_data, in_features)

  // Step 1: Calibration - collect activation statistics
  let activation_stats = collect_activation_stats(calibration_data)

  // Step 2: Compute AWQ scales (the magic)
  let awq_scales = compute_awq_scales(activation_stats, config.alpha)

  // Step 3: Transform weights (scale salient channels UP)
  let transformed_weights = apply_weight_transform(weight_matrix, awq_scales)

  // Step 4: Standard symmetric quantization on transformed weights
  let flat_transformed = list.flatten(transformed_weights)
  let #(quantized, quant_scales) =
    symmetric_group_quantize(flat_transformed, config.bits, config.group_size)

  // Memory calculation:
  // - bits per value for quantized data
  // - FP16 per group for quant scales
  // - FP16 per channel for AWQ scales
  let num_elements = list.length(flat_transformed)
  let num_groups = { num_elements + config.group_size - 1 } / config.group_size
  let data_bytes = { num_elements * config.bits + 7 } / 8
  let scale_bytes = num_groups * 2
  let awq_scale_bytes = in_features * 2
  let memory = data_bytes + scale_bytes + awq_scale_bytes

  AWQTensor(
    quantized_weights: quantized,
    awq_scales: awq_scales,
    quant_scales: quant_scales,
    zero_points: [],
    shape: shape,
    memory_bytes: memory,
  )
}

/// Symmetric per-group quantization
/// Why symmetric? Because asymmetric zero-points are a cache-miss nightmare.
fn symmetric_group_quantize(
  values: List(Float),
  bits: Int,
  group_size: Int,
) -> #(List(Int), List(Float)) {
  let qmax =
    float.power(2.0, int.to_float(bits - 1))
    |> float_result_to_float(128.0)
    |> fn(x) { x -. 1.0 }

  let groups = list.sized_chunk(values, group_size)

  let #(quantized_groups, scales) =
    list.fold(groups, #([], []), fn(acc, group) {
      let #(q_acc, s_acc) = acc

      // Per-group absmax
      let max_abs =
        group
        |> list.map(float.absolute_value)
        |> list.fold(0.0, float.max)

      let scale = case max_abs >. 0.0 {
        True -> qmax /. max_abs
        False -> 1.0
      }

      // Quantize
      let quantized =
        list.map(group, fn(v) {
          let scaled = v *. scale
          let clamped = float.clamp(scaled, -1.0 *. qmax, qmax)
          float.round(clamped)
        })

      #(list.append(q_acc, quantized), [scale, ..s_acc])
    })

  #(quantized_groups, list.reverse(scales))
}

// --- Dequantization ---

/// Dequantize AWQ tensor back to FP32
/// Note: Must also undo the AWQ weight transform
pub fn dequantize_awq(awq: AWQTensor) -> Tensor {
  let group_size = case awq.quant_scales {
    [] -> list.length(awq.quantized_weights)
    _ -> list.length(awq.quantized_weights) / list.length(awq.quant_scales)
  }

  let groups = list.sized_chunk(awq.quantized_weights, group_size)

  // Step 1: Dequantize per group
  let dequantized =
    list.index_map(groups, fn(group, idx) {
      let scale = get_at_index_float(awq.quant_scales, idx, 1.0)
      list.map(group, fn(q) { int.to_float(q) /. scale })
    })
    |> list.flatten

  // Step 2: Undo AWQ transform (divide by AWQ scales)
  let in_features = case awq.shape {
    [_, i] -> i
    _ -> 1
  }

  let weight_matrix = list.sized_chunk(dequantized, in_features)

  let restored =
    list.map(weight_matrix, fn(row) {
      list.map2(row, awq.awq_scales.weight_scales, fn(w, s) {
        case s >. 0.0 {
          True -> w /. s
          False -> w
        }
      })
    })
    |> list.flatten

  Tensor(data: restored, shape: awq.shape)
}

// --- Saliency Analysis ---

/// Identify the most salient channels (top-k by activation magnitude)
///
/// Key insight: Only ~1% of channels are truly salient.
/// But they contribute ~10% of the output magnitude.
/// Protecting them is the key to AWQ's success.
pub fn identify_salient_channels(
  activation_stats: List(Float),
  top_percent: Float,
) -> List(Int) {
  let n = list.length(activation_stats)
  let k =
    float.round(int.to_float(n) *. top_percent /. 100.0)
    |> int.max(1)

  // Sort by magnitude (descending) and take top-k indices
  activation_stats
  |> list.index_map(fn(stat, idx) { #(idx, stat) })
  |> list.sort(fn(a, b) { float.compare(b.1, a.1) })
  |> list.take(k)
  |> list.map(fn(pair) { pair.0 })
}

// --- Benchmark ---

pub fn main() {
  benchmark_awq()
}

pub fn benchmark_awq() {
  io.println(
    "=====================================================================",
  )
  io.println("  AWQ - Lin et al. (2024) MLSys Best Paper")
  io.println("  The key insight: 1% of weights matter 10x more")
  io.println(
    "=====================================================================\n",
  )

  io.println("--- The Algorithm ---")
  io.println("  1. Collect activation statistics (calibration)")
  io.println("  2. Identify salient channels (high activation = important)")
  io.println("  3. Scale salient weights UP before quantizing")
  io.println("  4. Scale activations DOWN at runtime (mathematically equivalent)")
  io.println("  Result: Protected channels get more quantization precision")
  io.println("")

  // Simulate [512, 256] weight matrix
  let weights = tensor.random_uniform([512, 256])

  // Simulate calibration data: 100 samples x 256 features
  let calibration_data =
    list.range(1, 100)
    |> list.map(fn(_) {
      tensor.random_uniform([256])
      |> tensor.to_list
    })

  let config = default_config()

  io.println("--- Calibration ---")
  let activation_stats = collect_activation_stats(calibration_data)
  io.println("  Samples: 100")
  io.println("  Features: 256")

  // Saliency analysis
  let salient_channels = identify_salient_channels(activation_stats, 1.0)
  io.println(
    "  Salient channels (top 1%): " <> int.to_string(list.length(salient_channels)),
  )

  io.println("  Top 5 most salient:")
  salient_channels
  |> list.take(5)
  |> list.each(fn(idx) {
    let stat = get_at_index_float(activation_stats, idx, 0.0)
    io.println(
      "    Channel " <> int.to_string(idx) <> ": " <> float_to_string(stat),
    )
  })

  io.println("\n--- AWQ Quantization ---")
  let #(time_awq, awq_tensor) =
    timer_tc(fn() { quantize_awq(weights, calibration_data, config) })

  let original_bytes = 512 * 256 * 4
  let ratio =
    int.to_float(original_bytes) /. int.to_float(awq_tensor.memory_bytes)

  io.println("  Time:        " <> int.to_string(time_awq / 1000) <> "ms")
  io.println("  Original:    " <> int.to_string(original_bytes / 1024) <> " KB")
  io.println(
    "  Compressed:  " <> int.to_string(awq_tensor.memory_bytes / 1024) <> " KB",
  )
  io.println("  Compression: " <> float_to_string(ratio) <> "x")

  // Error analysis
  io.println("\n--- Error Analysis ---")
  let decompressed = dequantize_awq(awq_tensor)
  let orig_data = tensor.to_list(weights)
  let decomp_data = tensor.to_list(decompressed)

  let errors =
    list.map2(orig_data, decomp_data, fn(o, d) { float.absolute_value(o -. d) })

  let mean_error = case errors {
    [] -> 0.0
    _ -> list.fold(errors, 0.0, float.add) /. int.to_float(list.length(errors))
  }

  let max_error = list.fold(errors, 0.0, float.max)

  io.println("  Mean error: " <> float_to_string(mean_error))
  io.println("  Max error:  " <> float_to_string(max_error))

  io.println("\n--- Why AWQ Beats Standard Quantization ---")
  io.println("  Standard: All channels quantized equally")
  io.println("  AWQ: Salient channels get more precision")
  io.println("  Same compression ratio, MUCH lower perplexity")

  io.println(
    "\n=====================================================================",
  )
  io.println("  AWQ IN PRODUCTION")
  io.println("")
  io.println("  LLaMA-7B:")
  io.println("    - FP16: 14GB")
  io.println("    - AWQ-4bit: 3.5GB (fits on RTX 3060!)")
  io.println("    - Perplexity loss: <0.5%")
  io.println("")
  io.println("  LLaMA-70B:")
  io.println("    - FP16: 140GB (needs 8x A100)")
  io.println("    - AWQ-4bit: 35GB (fits on single A100!)")
  io.println("    - Perplexity loss: <1%")
  io.println("")
  io.println("  Zero runtime overhead - transform is pre-computed.")
  io.println(
    "=====================================================================",
  )
}

// --- Helper Functions ---

fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

fn get_at_index_float(lst: List(Float), idx: Int, default: Float) -> Float {
  case list.drop(lst, idx) {
    [first, ..] -> first
    [] -> default
  }
}

fn float_power(base: Float, exp: Float) -> Float {
  case float.power(base, exp) {
    Ok(result) -> result
    Error(_) -> 1.0
  }
}

fn float_result_to_float(r: Result(Float, a), default: Float) -> Float {
  case r {
    Ok(v) -> v
    Error(_) -> default
  }
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 10_000.0)) /. 10_000.0
  float.to_string(rounded)
}

@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)
