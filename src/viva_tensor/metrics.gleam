//// Advanced Metrics for Quantization
////
//// Provides reconstruction-error and signal-quality metrics used by
//// quantization experiments. Fallible `try_*` functions preserve shape,
//// materialization, and empty-input errors; infallible wrappers remain for
//// compatibility with benchmark-style code.

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import gleam/string
import gleam_community/maths
import viva_tensor/core/error.{DimensionError, InvalidShape, ShapeMismatch}
import viva_tensor/tensor.{type Tensor, Tensor}

// ============================================================================
// TYPES
// ============================================================================

/// Complete quantization metrics
pub type QuantMetrics {
  QuantMetrics(
    /// Mean Squared Error
    mse: Float,
    /// Mean Absolute Error
    mae: Float,
    /// Root Mean Squared Error
    rmse: Float,
    /// Cosine Similarity (1.0 = perfect)
    cosine_sim: Float,
    /// Signal-to-Noise Ratio (dB)
    snr_db: Float,
    /// Signal-to-Quantization-Noise Ratio (dB)
    sqnr_db: Float,
    /// Max absolute error
    max_error: Float,
    /// 99th percentile of error
    p99_error: Float,
    /// Percentage of values with error > 1%
    outlier_pct: Float,
  )
}

/// Per-layer metrics (for LLMs)
pub type LayerMetrics {
  LayerMetrics(
    layer_name: String,
    metrics: QuantMetrics,
    sensitivity: Float,
    // How sensitive this layer is
  )
}

// ============================================================================
// BASIC METRICS
// ============================================================================

/// MSE - Mean Squared Error
pub fn try_mse(
  original: Tensor,
  quantized: Tensor,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair

  let squared_errors =
    list.map2(orig, quant, fn(o, q) {
      let diff = o -. q
      diff *. diff
    })

  mean(squared_errors)
}

/// MSE - Mean Squared Error
pub fn mse(original: Tensor, quantized: Tensor) -> Float {
  try_mse(original, quantized)
  |> result.unwrap(0.0)
}

/// MAE - Mean Absolute Error
pub fn try_mae(
  original: Tensor,
  quantized: Tensor,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair

  let abs_errors =
    list.map2(orig, quant, fn(o, q) { float.absolute_value(o -. q) })

  mean(abs_errors)
}

/// MAE - Mean Absolute Error
pub fn mae(original: Tensor, quantized: Tensor) -> Float {
  try_mae(original, quantized)
  |> result.unwrap(0.0)
}

/// RMSE - Root Mean Squared Error
pub fn try_rmse(
  original: Tensor,
  quantized: Tensor,
) -> Result(Float, tensor.TensorError) {
  use mse_val <- result.try(try_mse(original, quantized))

  case float.square_root(mse_val) {
    Ok(sqrt) -> Ok(sqrt)
    Error(_) -> Error(DimensionError("RMSE received a negative MSE"))
  }
}

/// RMSE - Root Mean Squared Error
pub fn rmse(original: Tensor, quantized: Tensor) -> Float {
  try_rmse(original, quantized)
  |> result.unwrap(0.0)
}

/// Cosine Similarity - measures direction, not magnitude
/// 1.0 = identical vectors, 0.0 = orthogonal, -1.0 = opposite
pub fn try_cosine_similarity(
  original: Tensor,
  quantized: Tensor,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair

  case maths.cosine_similarity(list.zip(orig, quant)) {
    Ok(value) -> Ok(value)
    Error(_) ->
      Error(DimensionError("Cosine similarity requires non-zero norm tensors"))
  }
}

/// Cosine Similarity - measures direction, not magnitude
/// 1.0 = identical vectors, 0.0 = orthogonal, -1.0 = opposite
pub fn cosine_similarity(original: Tensor, quantized: Tensor) -> Float {
  try_cosine_similarity(original, quantized)
  |> result.unwrap(0.0)
}

/// SNR - Signal-to-Noise Ratio in dB
/// SNR = 10 * log10(signal_power / noise_power)
pub fn try_snr_db(
  original: Tensor,
  quantized: Tensor,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair

  use signal_power <- result.try(mean(list.map(orig, fn(x) { x *. x })))

  use noise_power <- result.try(
    list.map2(orig, quant, fn(o, q) {
      let diff = o -. q
      diff *. diff
    })
    |> mean,
  )

  case noise_power >. 0.0 {
    True -> Ok(10.0 *. log10(signal_power /. noise_power))
    False -> Ok(100.0)
  }
}

/// SNR - Signal-to-Noise Ratio in dB
/// SNR = 10 * log10(signal_power / noise_power)
pub fn snr_db(original: Tensor, quantized: Tensor) -> Float {
  try_snr_db(original, quantized)
  |> result.unwrap(0.0)
}

/// SQNR - Signal-to-Quantization-Noise Ratio
/// Theoretical for N bits: SQNR = 6.02 * N + 1.76 dB
pub fn theoretical_sqnr(bits: Int) -> Float {
  6.02 *. int.to_float(bits) +. 1.76
}

/// Max Error - worst case
pub fn try_max_error(
  original: Tensor,
  quantized: Tensor,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair

  Ok(
    list.map2(orig, quant, fn(o, q) { float.absolute_value(o -. q) })
    |> list.fold(0.0, float.max),
  )
}

/// Max Error - worst case
pub fn max_error(original: Tensor, quantized: Tensor) -> Float {
  try_max_error(original, quantized)
  |> result.unwrap(0.0)
}

// ============================================================================
// ADVANCED METRICS
// ============================================================================

/// Error percentile (approximated via sorting)
pub fn error_percentile(
  original: Tensor,
  quantized: Tensor,
  percentile: Float,
) -> Float {
  try_error_percentile(original, quantized, percentile)
  |> result.unwrap(0.0)
}

/// Error percentile (approximated via sorting)
pub fn try_error_percentile(
  original: Tensor,
  quantized: Tensor,
  percentile: Float,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair
  use Nil <- result.try(validate_percentile(percentile))

  let errors =
    list.map2(orig, quant, fn(o, q) { float.absolute_value(o -. q) })
    |> list.sort(float.compare)

  case maths.percentile(errors, float.round(percentile)) {
    Ok(value) -> Ok(value)
    Error(_) ->
      Error(InvalidShape("Cannot compute percentile for empty tensors"))
  }
}

/// Percentage of outliers (error > threshold)
pub fn outlier_percentage(
  original: Tensor,
  quantized: Tensor,
  threshold: Float,
) -> Float {
  try_outlier_percentage(original, quantized, threshold)
  |> result.unwrap(0.0)
}

/// Percentage of outliers (error > threshold)
pub fn try_outlier_percentage(
  original: Tensor,
  quantized: Tensor,
  threshold: Float,
) -> Result(Float, tensor.TensorError) {
  use pair <- result.try(metric_data(original, quantized))
  let #(orig, quant) = pair

  let errors = list.map2(orig, quant, fn(o, q) { float.absolute_value(o -. q) })

  let outliers = list.filter(errors, fn(e) { e >. threshold })
  let n = list.length(errors)

  Ok(100.0 *. int.to_float(list.length(outliers)) /. int.to_float(n))
}

// ============================================================================
// COMPLETE METRICS
// ============================================================================

/// Computes all metrics at once
pub fn try_compute_all(
  original: Tensor,
  quantized: Tensor,
) -> Result(QuantMetrics, tensor.TensorError) {
  use mse_val <- result.try(try_mse(original, quantized))
  use mae_val <- result.try(try_mae(original, quantized))
  use rmse_val <- result.try(try_rmse(original, quantized))
  use cosine_val <- result.try(try_cosine_similarity(original, quantized))
  use snr_val <- result.try(try_snr_db(original, quantized))
  let sqnr_val = snr_val
  use max_err <- result.try(try_max_error(original, quantized))
  use p99 <- result.try(try_error_percentile(original, quantized, 99.0))
  use outliers <- result.try(try_outlier_percentage(original, quantized, 0.01))

  Ok(QuantMetrics(
    mse: mse_val,
    mae: mae_val,
    rmse: rmse_val,
    cosine_sim: cosine_val,
    snr_db: snr_val,
    sqnr_db: sqnr_val,
    max_error: max_err,
    p99_error: p99,
    outlier_pct: outliers,
  ))
}

/// Computes all metrics at once
pub fn compute_all(original: Tensor, quantized: Tensor) -> QuantMetrics {
  try_compute_all(original, quantized)
  |> result.unwrap(QuantMetrics(
    mse: 0.0,
    mae: 0.0,
    rmse: 0.0,
    cosine_sim: 0.0,
    snr_db: 0.0,
    sqnr_db: 0.0,
    max_error: 0.0,
    p99_error: 0.0,
    outlier_pct: 0.0,
  ))
}

// ============================================================================
// SALIENCY - AWQ Insight
// ============================================================================

/// Computes weight saliency based on activations
/// Salience(w) = Var(activation) * w²
pub fn compute_saliency(
  weights: Tensor,
  activations: List(List(Float)),
) -> List(Float) {
  let w_data = tensor.to_list(weights)

  // Compute variance of activations per channel
  let activation_vars = case activations {
    [] -> list.repeat(1.0, list.length(w_data))
    [first, ..] -> {
      let n_channels = list.length(first)
      let n_samples = int.to_float(list.length(activations))

      // Mean per channel
      let means =
        list.repeat(0.0, n_channels)
        |> list.index_fold(activations, _, fn(acc, acts, _) {
          list.map2(acc, acts, fn(a, act) { a +. act })
        })
        |> list.map(fn(s) { s /. n_samples })

      // Variance per channel
      list.index_fold(
        activations,
        list.repeat(0.0, n_channels),
        fn(acc, acts, _) {
          list.index_map(acc, fn(a, i) {
            let mean = get_at(means, i) |> result_or(0.0)
            let act = get_at(acts, i) |> result_or(0.0)
            let diff = act -. mean
            a +. diff *. diff
          })
        },
      )
      |> list.map(fn(v) { v /. n_samples })
    }
  }

  // Pad variances to match weights
  let padded_vars = pad_or_truncate(activation_vars, list.length(w_data), 1.0)

  // Saliency = var * w²
  list.map2(padded_vars, w_data, fn(var, w) { var *. w *. w })
}

/// Identifies top K% of salient weights
pub fn find_salient_weights(
  saliency: List(Float),
  top_pct: Float,
) -> List(Int) {
  // Index + saliency pairs
  let indexed = list.index_map(saliency, fn(s, i) { #(i, s) })

  // Sort by saliency descending
  let sorted =
    list.sort(indexed, fn(a, b) {
      float.compare(b.1, a.1)
      // descending
    })

  // Take top K%
  let n = list.length(saliency)
  let k =
    float.round(int.to_float(n) *. top_pct /. 100.0)
    |> int.max(1)

  list.take(sorted, k)
  |> list.map(fn(pair) { pair.0 })
}

// ============================================================================
// BENCHMARK
// ============================================================================

pub fn main() {
  benchmark_metrics()
}

pub fn benchmark_metrics() {
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║          QUANTIZATION METRICS - BENCHMARK                     ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Create test tensor
  let original = tensor.random_normal([1024], 0.0, 1.0)

  // Simulate quantization with different noise levels
  let small_noise = add_noise(original, 0.01)
  let medium_noise = add_noise(original, 0.05)
  let large_noise = add_noise(original, 0.1)

  io.println("Original: 1024 floats, mean=0, std=1")
  io.println("")

  io.println("┌────────────────┬────────────┬────────────┬────────────┐")
  io.println("│ Metric         │ Noise 1%   │ Noise 5%   │ Noise 10%  │")
  io.println("├────────────────┼────────────┼────────────┼────────────┤")

  let m1 = compute_all(original, small_noise)
  let m2 = compute_all(original, medium_noise)
  let m3 = compute_all(original, large_noise)

  io.println(
    "│ MSE            │ "
    <> pad_float(m1.mse)
    <> " │ "
    <> pad_float(m2.mse)
    <> " │ "
    <> pad_float(m3.mse)
    <> " │",
  )
  io.println(
    "│ MAE            │ "
    <> pad_float(m1.mae)
    <> " │ "
    <> pad_float(m2.mae)
    <> " │ "
    <> pad_float(m3.mae)
    <> " │",
  )
  io.println(
    "│ RMSE           │ "
    <> pad_float(m1.rmse)
    <> " │ "
    <> pad_float(m2.rmse)
    <> " │ "
    <> pad_float(m3.rmse)
    <> " │",
  )
  io.println(
    "│ Cosine Sim     │ "
    <> pad_float(m1.cosine_sim)
    <> " │ "
    <> pad_float(m2.cosine_sim)
    <> " │ "
    <> pad_float(m3.cosine_sim)
    <> " │",
  )
  io.println(
    "│ SNR (dB)       │ "
    <> pad_float(m1.snr_db)
    <> " │ "
    <> pad_float(m2.snr_db)
    <> " │ "
    <> pad_float(m3.snr_db)
    <> " │",
  )
  io.println(
    "│ Max Error      │ "
    <> pad_float(m1.max_error)
    <> " │ "
    <> pad_float(m2.max_error)
    <> " │ "
    <> pad_float(m3.max_error)
    <> " │",
  )
  io.println(
    "│ P99 Error      │ "
    <> pad_float(m1.p99_error)
    <> " │ "
    <> pad_float(m2.p99_error)
    <> " │ "
    <> pad_float(m3.p99_error)
    <> " │",
  )
  io.println("└────────────────┴────────────┴────────────┴────────────┘")

  io.println("")
  io.println("Theoretical SQNR:")
  io.println("  INT8 (8 bits): " <> float_to_str(theoretical_sqnr(8)) <> " dB")
  io.println("  INT4 (4 bits): " <> float_to_str(theoretical_sqnr(4)) <> " dB")
  io.println("  INT2 (2 bits): " <> float_to_str(theoretical_sqnr(2)) <> " dB")

  io.println("")
}

// ============================================================================
// HELPERS
// ============================================================================

fn metric_data(
  original: Tensor,
  quantized: Tensor,
) -> Result(#(List(Float), List(Float)), tensor.TensorError) {
  case original.shape == quantized.shape {
    False -> Error(ShapeMismatch(original.shape, quantized.shape))
    True -> {
      use orig <- result.try(tensor.try_to_list(original))
      use quant <- result.try(tensor.try_to_list(quantized))

      case orig {
        [] -> Error(InvalidShape("Metrics require at least one element"))
        _ -> Ok(#(orig, quant))
      }
    }
  }
}

fn mean(values: List(Float)) -> Result(Float, tensor.TensorError) {
  case maths.mean(values) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(InvalidShape("Cannot compute mean of an empty list"))
  }
}

fn validate_percentile(percentile: Float) -> Result(Nil, tensor.TensorError) {
  case percentile >=. 0.0 && percentile <=. 100.0 {
    True -> Ok(Nil)
    False -> Error(DimensionError("Percentile must be between 0 and 100"))
  }
}

fn add_noise(t: Tensor, noise_level: Float) -> Tensor {
  let data = tensor.to_list(t)
  let noisy =
    list.index_map(data, fn(x, i) {
      // Pseudo-random noise based on index
      let noise = int.to_float(i % 100 - 50) /. 50.0 *. noise_level
      x +. noise
    })
  Tensor(data: noisy, shape: get_tensor_shape(t))
}

fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
    tensor.NativeTensor(_, shape) -> shape
  }
}

fn log10(x: Float) -> Float {
  maths.logarithm_10(x)
  |> result.unwrap(0.0)
}

fn float_to_str(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn pad_float(f: Float) -> String {
  let s = float_to_str(f)
  let len = string.length(s)
  let padding = 10 - len
  case padding > 0 {
    True -> s <> string.repeat(" ", padding)
    False -> string.slice(s, 0, 10)
  }
}

fn get_at(list: List(a), index: Int) -> Result(a, Nil) {
  list
  |> list.drop(index)
  |> list.first
}

fn result_or(r: Result(a, e), default: a) -> a {
  case r {
    Ok(v) -> v
    Error(_) -> default
  }
}

fn pad_or_truncate(lst: List(a), target_len: Int, default: a) -> List(a) {
  let current_len = list.length(lst)
  case current_len >= target_len {
    True -> list.take(lst, target_len)
    False -> lst |> list.append(list.repeat(default, target_len - current_len))
  }
}
