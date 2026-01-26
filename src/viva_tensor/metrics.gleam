//// Métricas Avançadas para Quantização
////
//// Baseado na análise do Qwen3-235B sobre algoritmos state-of-the-art:
//// - MSE (Mean Squared Error)
//// - MAE (Mean Absolute Error)
//// - Cosine Similarity
//// - SNR (Signal-to-Noise Ratio)
//// - SQNR (Signal-to-Quantization-Noise Ratio)
//// - Perplexity Delta (para LLMs)
////
//// INSIGHTS DO QWEN3:
//// 1. AWQ: Proteger 1% dos pesos salientes reduz erro drasticamente
//// 2. NF4: Quantis não-uniformes (distribuição normal) > uniformes
//// 3. GPTQ: Ponderar erro pelo Hessian melhora precisão
//// 4. Flash Attention: Online softmax com shifting evita overflow

import gleam/float
import gleam/int
import gleam/list
import gleam/io
import gleam/string
import viva_tensor/tensor.{type Tensor, Tensor}

// ============================================================================
// TIPOS
// ============================================================================

/// Métricas completas de quantização
pub type QuantMetrics {
  QuantMetrics(
    /// Mean Squared Error
    mse: Float,
    /// Mean Absolute Error
    mae: Float,
    /// Root Mean Squared Error
    rmse: Float,
    /// Cosine Similarity (1.0 = perfeito)
    cosine_sim: Float,
    /// Signal-to-Noise Ratio (dB)
    snr_db: Float,
    /// Signal-to-Quantization-Noise Ratio (dB)
    sqnr_db: Float,
    /// Max absolute error
    max_error: Float,
    /// Percentil 99 do erro
    p99_error: Float,
    /// Porcentagem de valores com erro > 1%
    outlier_pct: Float,
  )
}

/// Métricas por camada (para LLMs)
pub type LayerMetrics {
  LayerMetrics(
    layer_name: String,
    metrics: QuantMetrics,
    sensitivity: Float,  // Quão sensível é essa camada
  )
}

// ============================================================================
// MÉTRICAS BÁSICAS
// ============================================================================

/// MSE - Mean Squared Error
pub fn mse(original: Tensor, quantized: Tensor) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  let squared_errors = list.map2(orig, quant, fn(o, q) {
    let diff = o -. q
    diff *. diff
  })

  list.fold(squared_errors, 0.0, fn(acc, x) { acc +. x })
    /. int.to_float(list.length(squared_errors))
}

/// MAE - Mean Absolute Error
pub fn mae(original: Tensor, quantized: Tensor) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  let abs_errors = list.map2(orig, quant, fn(o, q) {
    float.absolute_value(o -. q)
  })

  list.fold(abs_errors, 0.0, fn(acc, x) { acc +. x })
    /. int.to_float(list.length(abs_errors))
}

/// RMSE - Root Mean Squared Error
pub fn rmse(original: Tensor, quantized: Tensor) -> Float {
  let mse_val = mse(original, quantized)
  case float.square_root(mse_val) {
    Ok(sqrt) -> sqrt
    Error(_) -> 0.0
  }
}

/// Cosine Similarity - mede direção, não magnitude
/// 1.0 = vetores idênticos, 0.0 = ortogonais, -1.0 = opostos
pub fn cosine_similarity(original: Tensor, quantized: Tensor) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  // dot product
  let dot = list.map2(orig, quant, fn(o, q) { o *. q })
    |> list.fold(0.0, fn(acc, x) { acc +. x })

  // norms
  let norm_orig = orig
    |> list.map(fn(x) { x *. x })
    |> list.fold(0.0, fn(acc, x) { acc +. x })
    |> float.square_root

  let norm_quant = quant
    |> list.map(fn(x) { x *. x })
    |> list.fold(0.0, fn(acc, x) { acc +. x })
    |> float.square_root

  case norm_orig, norm_quant {
    Ok(no), Ok(nq) if no >. 0.0 && nq >. 0.0 -> dot /. { no *. nq }
    _, _ -> 0.0
  }
}

/// SNR - Signal-to-Noise Ratio em dB
/// SNR = 10 * log10(signal_power / noise_power)
pub fn snr_db(original: Tensor, quantized: Tensor) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  // Signal power = mean(x²)
  let signal_power = orig
    |> list.map(fn(x) { x *. x })
    |> list.fold(0.0, fn(acc, x) { acc +. x })
    |> fn(sum) { sum /. int.to_float(list.length(orig)) }

  // Noise power = mean((x - x')²)
  let noise_power = list.map2(orig, quant, fn(o, q) {
    let diff = o -. q
    diff *. diff
  })
    |> list.fold(0.0, fn(acc, x) { acc +. x })
    |> fn(sum) { sum /. int.to_float(list.length(orig)) }

  // Evita divisão por zero
  case noise_power >. 0.0 {
    True -> 10.0 *. log10(signal_power /. noise_power)
    False -> 100.0  // Sem ruído = SNR infinito, capped
  }
}

/// SQNR - Signal-to-Quantization-Noise Ratio
/// Teórico para N bits: SQNR = 6.02 * N + 1.76 dB
pub fn theoretical_sqnr(bits: Int) -> Float {
  6.02 *. int.to_float(bits) +. 1.76
}

/// Max Error - pior caso
pub fn max_error(original: Tensor, quantized: Tensor) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  list.map2(orig, quant, fn(o, q) {
    float.absolute_value(o -. q)
  })
    |> list.fold(0.0, float.max)
}

// ============================================================================
// MÉTRICAS AVANÇADAS
// ============================================================================

/// Percentil do erro (aproximado via sorting)
pub fn error_percentile(original: Tensor, quantized: Tensor, percentile: Float) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  let errors = list.map2(orig, quant, fn(o, q) {
    float.absolute_value(o -. q)
  })
    |> list.sort(float.compare)

  let n = list.length(errors)
  let idx = float.round(int.to_float(n) *. percentile /. 100.0)
  let safe_idx = int.min(idx, n - 1) |> int.max(0)

  list.drop(errors, safe_idx)
    |> list.first
    |> fn(r) { case r { Ok(v) -> v Error(_) -> 0.0 } }
}

/// Porcentagem de outliers (erro > threshold)
pub fn outlier_percentage(original: Tensor, quantized: Tensor, threshold: Float) -> Float {
  let orig = tensor.to_list(original)
  let quant = tensor.to_list(quantized)

  let errors = list.map2(orig, quant, fn(o, q) {
    float.absolute_value(o -. q)
  })

  let outliers = list.filter(errors, fn(e) { e >. threshold })
  let n = list.length(errors)

  100.0 *. int.to_float(list.length(outliers)) /. int.to_float(n)
}

// ============================================================================
// MÉTRICAS COMPLETAS
// ============================================================================

/// Computa todas as métricas de uma vez
pub fn compute_all(original: Tensor, quantized: Tensor) -> QuantMetrics {
  let mse_val = mse(original, quantized)
  let mae_val = mae(original, quantized)
  let rmse_val = case float.square_root(mse_val) {
    Ok(sqrt) -> sqrt
    Error(_) -> 0.0
  }
  let cosine_val = cosine_similarity(original, quantized)
  let snr_val = snr_db(original, quantized)
  let sqnr_val = snr_val  // Para quantização, SNR ≈ SQNR
  let max_err = max_error(original, quantized)
  let p99 = error_percentile(original, quantized, 99.0)
  let outliers = outlier_percentage(original, quantized, 0.01)

  QuantMetrics(
    mse: mse_val,
    mae: mae_val,
    rmse: rmse_val,
    cosine_sim: cosine_val,
    snr_db: snr_val,
    sqnr_db: sqnr_val,
    max_error: max_err,
    p99_error: p99,
    outlier_pct: outliers,
  )
}

// ============================================================================
// SALIENCY - Insight do AWQ
// ============================================================================

/// Computa saliência de pesos baseado em ativações
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
      let means = list.repeat(0.0, n_channels)
        |> list.index_fold(activations, _, fn(acc, acts, _) {
          list.map2(acc, acts, fn(a, act) { a +. act })
        })
        |> list.map(fn(s) { s /. n_samples })

      // Variance per channel
      list.index_fold(activations, list.repeat(0.0, n_channels), fn(acc, acts, _) {
        list.index_map(acc, fn(a, i) {
          let mean = get_at(means, i) |> result_or(0.0)
          let act = get_at(acts, i) |> result_or(0.0)
          let diff = act -. mean
          a +. diff *. diff
        })
      })
        |> list.map(fn(v) { v /. n_samples })
    }
  }

  // Pad variances to match weights
  let padded_vars = pad_or_truncate(activation_vars, list.length(w_data), 1.0)

  // Saliency = var * w²
  list.map2(padded_vars, w_data, fn(var, w) {
    var *. w *. w
  })
}

/// Identifica top K% de pesos salientes
pub fn find_salient_weights(saliency: List(Float), top_pct: Float) -> List(Int) {
  // Index + saliency pairs
  let indexed = list.index_map(saliency, fn(s, i) { #(i, s) })

  // Sort by saliency descending
  let sorted = list.sort(indexed, fn(a, b) {
    float.compare(b.1, a.1)  // descending
  })

  // Take top K%
  let n = list.length(saliency)
  let k = float.round(int.to_float(n) *. top_pct /. 100.0)
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
  io.println("╔═══════════════════════════════════════════════════════════════╗")
  io.println("║          MÉTRICAS DE QUANTIZAÇÃO - BENCHMARK                  ║")
  io.println("╚═══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Criar tensor de teste
  let original = tensor.random_normal([1024], 0.0, 1.0)

  // Simular quantização com diferentes níveis de ruído
  let small_noise = add_noise(original, 0.01)
  let medium_noise = add_noise(original, 0.05)
  let large_noise = add_noise(original, 0.10)

  io.println("Original: 1024 floats, mean=0, std=1")
  io.println("")

  io.println("┌────────────────┬────────────┬────────────┬────────────┐")
  io.println("│ Métrica        │ Ruído 1%   │ Ruído 5%   │ Ruído 10%  │")
  io.println("├────────────────┼────────────┼────────────┼────────────┤")

  let m1 = compute_all(original, small_noise)
  let m2 = compute_all(original, medium_noise)
  let m3 = compute_all(original, large_noise)

  io.println("│ MSE            │ " <> pad_float(m1.mse) <> " │ " <> pad_float(m2.mse) <> " │ " <> pad_float(m3.mse) <> " │")
  io.println("│ MAE            │ " <> pad_float(m1.mae) <> " │ " <> pad_float(m2.mae) <> " │ " <> pad_float(m3.mae) <> " │")
  io.println("│ RMSE           │ " <> pad_float(m1.rmse) <> " │ " <> pad_float(m2.rmse) <> " │ " <> pad_float(m3.rmse) <> " │")
  io.println("│ Cosine Sim     │ " <> pad_float(m1.cosine_sim) <> " │ " <> pad_float(m2.cosine_sim) <> " │ " <> pad_float(m3.cosine_sim) <> " │")
  io.println("│ SNR (dB)       │ " <> pad_float(m1.snr_db) <> " │ " <> pad_float(m2.snr_db) <> " │ " <> pad_float(m3.snr_db) <> " │")
  io.println("│ Max Error      │ " <> pad_float(m1.max_error) <> " │ " <> pad_float(m2.max_error) <> " │ " <> pad_float(m3.max_error) <> " │")
  io.println("│ P99 Error      │ " <> pad_float(m1.p99_error) <> " │ " <> pad_float(m2.p99_error) <> " │ " <> pad_float(m3.p99_error) <> " │")
  io.println("└────────────────┴────────────┴────────────┴────────────┘")

  io.println("")
  io.println("Teórico SQNR:")
  io.println("  INT8 (8 bits): " <> float_to_str(theoretical_sqnr(8)) <> " dB")
  io.println("  INT4 (4 bits): " <> float_to_str(theoretical_sqnr(4)) <> " dB")
  io.println("  INT2 (2 bits): " <> float_to_str(theoretical_sqnr(2)) <> " dB")

  io.println("")
}

// ============================================================================
// HELPERS
// ============================================================================

fn add_noise(t: Tensor, noise_level: Float) -> Tensor {
  let data = tensor.to_list(t)
  let noisy = list.index_map(data, fn(x, i) {
    // Pseudo-random noise baseado no índice
    let noise = int.to_float(i % 100 - 50) /. 50.0 *. noise_level
    x +. noise
  })
  Tensor(data: noisy, shape: get_tensor_shape(t))
}

fn get_tensor_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

fn log10(x: Float) -> Float {
  // log10(x) = ln(x) / ln(10)
  case x >. 0.0 {
    True -> {
      // Aproximação simples
      let ln_10 = 2.302585093
      approximate_ln(x) /. ln_10
    }
    False -> 0.0
  }
}

fn approximate_ln(x: Float) -> Float {
  // ln(x) ≈ 2 * sum((y-1)/(y+1))^(2n+1) / (2n+1)
  // onde y = x
  // Simplificado para range comum
  case x {
    v if v <. 0.001 -> -7.0
    v if v <. 0.01 -> -4.6
    v if v <. 0.1 -> -2.3
    v if v <. 1.0 -> v -. 1.0  // Aproximação linear perto de 1
    v if v <. 10.0 -> { v -. 1.0 } /. v *. 2.0
    v if v <. 100.0 -> 2.3 +. { v /. 10.0 -. 1.0 } /. { v /. 10.0 }
    _ -> 4.6
  }
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
