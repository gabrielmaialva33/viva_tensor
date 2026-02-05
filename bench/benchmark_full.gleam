//// Benchmark Completo com Métricas Avançadas
////
//// Compara SQNR real vs teórico para INT8, NF4, AWQ
//// Baseado em papers 2024-2026: NVFP4, TWEO, FALQON

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string

import viva_tensor/awq
import viva_tensor/compression
import viva_tensor/metrics
import viva_tensor/nf4
import viva_tensor/tensor.{type Tensor}

// ============================================================================
// MAIN BENCHMARK
// ============================================================================

pub fn main() {
  run_full_benchmark()
}

pub fn run_full_benchmark() {
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║          viva_tensor - BENCHMARK COMPLETO COM MÉTRICAS AVANÇADAS          ║",
  )
  io.println(
    "║                   SQNR Real vs Teórico | Papers 2024-2026                 ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Criar tensor de teste - simula pesos de rede neural (distribuição normal)
  let test_tensor = tensor.random_normal([512, 512], 0.0, 0.3)
  let original_bytes = 512 * 512 * 4
  // FP32

  io.println("━━━ CONFIGURAÇÃO DO TESTE ━━━")
  io.println("  Tensor: [512, 512] = 262,144 elementos")
  io.println("  Distribuição: Normal(0, 0.3) - típico de pesos NN")
  io.println(
    "  Original: " <> int.to_string(original_bytes / 1024) <> " KB (FP32)",
  )
  io.println("")

  // Benchmark INT8
  benchmark_int8(test_tensor, original_bytes)

  // Benchmark NF4
  benchmark_nf4_full(test_tensor, original_bytes)

  // Benchmark AWQ
  benchmark_awq_full(test_tensor, original_bytes)

  // Tabela comparativa final
  print_comparison_table(test_tensor, original_bytes)

  io.println("")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
  io.println("                        BENCHMARK CONCLUÍDO!")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
}

// ============================================================================
// INT8 BENCHMARK
// ============================================================================

fn benchmark_int8(original: Tensor, original_bytes: Int) {
  io.println(
    "┌─────────────────────────────────────────────────────────────────────────┐",
  )
  io.println(
    "│ INT8 QUANTIZATION - Symmetric Per-Tensor                               │",
  )
  io.println(
    "└─────────────────────────────────────────────────────────────────────────┘",
  )

  let quantized = compression.quantize_int8(original)
  let recovered = compression.dequantize(quantized)

  // Métricas completas
  let quant_metrics = metrics.compute_all(original, recovered)

  // SQNR teórico para 8 bits
  let theoretical_sqnr = metrics.theoretical_sqnr(8)

  io.println("")
  io.println("  ┌─────────────────────────────────────────────┐")
  io.println("  │ MÉTRICAS DE QUALIDADE                       │")
  io.println("  ├─────────────────────────────────────────────┤")
  io.println(
    "  │ MSE:              " <> pad_float(quant_metrics.mse) <> "          │",
  )
  io.println(
    "  │ MAE:              " <> pad_float(quant_metrics.mae) <> "          │",
  )
  io.println(
    "  │ RMSE:             " <> pad_float(quant_metrics.rmse) <> "          │",
  )
  io.println(
    "  │ Cosine Sim:       "
    <> pad_float(quant_metrics.cosine_sim)
    <> "          │",
  )
  io.println(
    "  │ Max Error:        "
    <> pad_float(quant_metrics.max_error)
    <> "          │",
  )
  io.println(
    "  │ P99 Error:        "
    <> pad_float(quant_metrics.p99_error)
    <> "          │",
  )
  io.println("  ├─────────────────────────────────────────────┤")
  io.println(
    "  │ SNR (Real):       " <> pad_float(quant_metrics.snr_db) <> " dB       │",
  )
  io.println(
    "  │ SQNR (Teórico):   " <> pad_float(theoretical_sqnr) <> " dB       │",
  )
  io.println(
    "  │ Gap:              "
    <> pad_float(theoretical_sqnr -. quant_metrics.snr_db)
    <> " dB       │",
  )
  io.println("  └─────────────────────────────────────────────┘")

  let ratio =
    int.to_float(original_bytes) /. int.to_float(quantized.memory_bytes)
  io.println("")
  io.println("  Compressão: " <> float_to_str(ratio) <> "x")
  io.println(
    "  Memória: " <> int.to_string(quantized.memory_bytes / 1024) <> " KB",
  )
  io.println("")
}

// ============================================================================
// NF4 BENCHMARK
// ============================================================================

fn benchmark_nf4_full(original: Tensor, original_bytes: Int) {
  io.println(
    "┌─────────────────────────────────────────────────────────────────────────┐",
  )
  io.println(
    "│ NF4 QUANTIZATION - QLoRA Style (Normal Distribution Quantiles)         │",
  )
  io.println(
    "└─────────────────────────────────────────────────────────────────────────┘",
  )

  let config = nf4.default_config()
  let quantized = nf4.quantize(original, config)
  let recovered = nf4.dequantize(quantized)

  // Métricas completas
  let quant_metrics = metrics.compute_all(original, recovered)

  // SQNR teórico para 4 bits
  let theoretical_sqnr = metrics.theoretical_sqnr(4)

  io.println("")
  io.println("  ┌─────────────────────────────────────────────┐")
  io.println("  │ MÉTRICAS DE QUALIDADE                       │")
  io.println("  ├─────────────────────────────────────────────┤")
  io.println(
    "  │ MSE:              " <> pad_float(quant_metrics.mse) <> "          │",
  )
  io.println(
    "  │ MAE:              " <> pad_float(quant_metrics.mae) <> "          │",
  )
  io.println(
    "  │ RMSE:             " <> pad_float(quant_metrics.rmse) <> "          │",
  )
  io.println(
    "  │ Cosine Sim:       "
    <> pad_float(quant_metrics.cosine_sim)
    <> "          │",
  )
  io.println(
    "  │ Max Error:        "
    <> pad_float(quant_metrics.max_error)
    <> "          │",
  )
  io.println(
    "  │ P99 Error:        "
    <> pad_float(quant_metrics.p99_error)
    <> "          │",
  )
  io.println("  ├─────────────────────────────────────────────┤")
  io.println(
    "  │ SNR (Real):       " <> pad_float(quant_metrics.snr_db) <> " dB       │",
  )
  io.println(
    "  │ SQNR (Teórico):   " <> pad_float(theoretical_sqnr) <> " dB       │",
  )
  io.println(
    "  │ Gap:              "
    <> pad_float(theoretical_sqnr -. quant_metrics.snr_db)
    <> " dB       │",
  )
  io.println("  └─────────────────────────────────────────────┘")

  io.println("")
  io.println(
    "  Compressão: " <> float_to_str(quantized.compression_ratio) <> "x",
  )
  io.println(
    "  Memória: " <> int.to_string(quantized.memory_bytes / 1024) <> " KB",
  )

  // Double Quantization
  io.println("")
  io.println("  ─── Double Quantization (NF4 + DQ) ───")
  let dq = nf4.double_quantize(original, config)
  let dq_ratio = int.to_float(original_bytes) /. int.to_float(dq.memory_bytes)
  io.println("  Compressão DQ: " <> float_to_str(dq_ratio) <> "x")
  io.println("  Memória DQ: " <> int.to_string(dq.memory_bytes / 1024) <> " KB")
  io.println("")
}

// ============================================================================
// AWQ BENCHMARK
// ============================================================================

fn benchmark_awq_full(original: Tensor, original_bytes: Int) {
  io.println(
    "┌─────────────────────────────────────────────────────────────────────────┐",
  )
  io.println(
    "│ AWQ QUANTIZATION - Activation-aware (MLSys 2024 Best Paper)            │",
  )
  io.println(
    "└─────────────────────────────────────────────────────────────────────────┘",
  )

  // Gera dados de calibração simulados
  let calibration_data = generate_calibration_data(64, 512)

  let config = awq.default_config()
  let quantized = awq.quantize_awq(original, calibration_data, config)
  let recovered = awq.dequantize_awq(quantized)

  // Métricas completas
  let quant_metrics = metrics.compute_all(original, recovered)

  // SQNR teórico para 4 bits (AWQ usa 4-bit por padrão)
  let theoretical_sqnr = metrics.theoretical_sqnr(4)

  io.println("")
  io.println("  ┌─────────────────────────────────────────────┐")
  io.println("  │ MÉTRICAS DE QUALIDADE                       │")
  io.println("  ├─────────────────────────────────────────────┤")
  io.println(
    "  │ MSE:              " <> pad_float(quant_metrics.mse) <> "          │",
  )
  io.println(
    "  │ MAE:              " <> pad_float(quant_metrics.mae) <> "          │",
  )
  io.println(
    "  │ RMSE:             " <> pad_float(quant_metrics.rmse) <> "          │",
  )
  io.println(
    "  │ Cosine Sim:       "
    <> pad_float(quant_metrics.cosine_sim)
    <> "          │",
  )
  io.println(
    "  │ Max Error:        "
    <> pad_float(quant_metrics.max_error)
    <> "          │",
  )
  io.println(
    "  │ P99 Error:        "
    <> pad_float(quant_metrics.p99_error)
    <> "          │",
  )
  io.println("  ├─────────────────────────────────────────────┤")
  io.println(
    "  │ SNR (Real):       " <> pad_float(quant_metrics.snr_db) <> " dB       │",
  )
  io.println(
    "  │ SQNR (Teórico):   " <> pad_float(theoretical_sqnr) <> " dB       │",
  )
  io.println(
    "  │ Gap:              "
    <> pad_float(theoretical_sqnr -. quant_metrics.snr_db)
    <> " dB       │",
  )
  io.println("  └─────────────────────────────────────────────┘")

  let ratio =
    int.to_float(original_bytes) /. int.to_float(quantized.memory_bytes)
  io.println("")
  io.println("  Compressão: " <> float_to_str(ratio) <> "x")
  io.println(
    "  Memória: " <> int.to_string(quantized.memory_bytes / 1024) <> " KB",
  )

  // Análise de saliência
  io.println("")
  io.println("  ─── Análise de Saliência (AWQ Insight) ───")
  let activation_stats = awq.collect_activation_stats(calibration_data)
  let salient = awq.identify_salient_channels(activation_stats, 1.0)
  io.println(
    "  Canais salientes (top 1%): " <> int.to_string(list.length(salient)),
  )
  io.println("  Esses ~1% dominam o erro de quantização!")
  io.println("")
}

// ============================================================================
// TABELA COMPARATIVA
// ============================================================================

fn print_comparison_table(original: Tensor, original_bytes: Int) {
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║                    TABELA COMPARATIVA FINAL                               ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Coleta métricas de cada método
  let int8_q = compression.quantize_int8(original)
  let int8_r = compression.dequantize(int8_q)
  let int8_m = metrics.compute_all(original, int8_r)

  let nf4_config = nf4.default_config()
  let nf4_q = nf4.quantize(original, nf4_config)
  let nf4_r = nf4.dequantize(nf4_q)
  let nf4_m = metrics.compute_all(original, nf4_r)

  let calib = generate_calibration_data(64, 512)
  let awq_config = awq.default_config()
  let awq_q = awq.quantize_awq(original, calib, awq_config)
  let awq_r = awq.dequantize_awq(awq_q)
  let awq_m = metrics.compute_all(original, awq_r)

  // SQNR teóricos
  let sqnr_8 = metrics.theoretical_sqnr(8)
  let sqnr_4 = metrics.theoretical_sqnr(4)

  // Ratios
  let int8_ratio =
    int.to_float(original_bytes) /. int.to_float(int8_q.memory_bytes)
  let nf4_ratio = nf4_q.compression_ratio
  let awq_ratio =
    int.to_float(original_bytes) /. int.to_float(awq_q.memory_bytes)

  io.println(
    "┌────────────┬───────────┬──────────┬───────────┬───────────┬──────────┐",
  )
  io.println(
    "│ Método     │ Compres.  │ SNR Real │ SNR Teór. │ Gap       │ Cosine   │",
  )
  io.println(
    "├────────────┼───────────┼──────────┼───────────┼───────────┼──────────┤",
  )

  // INT8
  let int8_gap = sqnr_8 -. int8_m.snr_db
  io.println(
    "│ INT8       │ "
    <> pad_ratio(int8_ratio)
    <> "x   │ "
    <> pad_snr(int8_m.snr_db)
    <> " │ "
    <> pad_snr(sqnr_8)
    <> "  │ "
    <> pad_gap(int8_gap)
    <> "  │ "
    <> pad_cos(int8_m.cosine_sim)
    <> " │",
  )

  // NF4
  let nf4_gap = sqnr_4 -. nf4_m.snr_db
  io.println(
    "│ NF4        │ "
    <> pad_ratio(nf4_ratio)
    <> "x   │ "
    <> pad_snr(nf4_m.snr_db)
    <> " │ "
    <> pad_snr(sqnr_4)
    <> "  │ "
    <> pad_gap(nf4_gap)
    <> "  │ "
    <> pad_cos(nf4_m.cosine_sim)
    <> " │",
  )

  // AWQ
  let awq_gap = sqnr_4 -. awq_m.snr_db
  io.println(
    "│ AWQ        │ "
    <> pad_ratio(awq_ratio)
    <> "x   │ "
    <> pad_snr(awq_m.snr_db)
    <> " │ "
    <> pad_snr(sqnr_4)
    <> "  │ "
    <> pad_gap(awq_gap)
    <> "  │ "
    <> pad_cos(awq_m.cosine_sim)
    <> " │",
  )

  io.println(
    "└────────────┴───────────┴──────────┴───────────┴───────────┴──────────┘",
  )

  io.println("")
  io.println("LEGENDA:")
  io.println("  - Compres.: Taxa de compressão (maior = melhor)")
  io.println("  - SNR Real: Signal-to-Noise Ratio medido (maior = melhor)")
  io.println("  - SNR Teór.: SQNR teórico = 6.02*N + 1.76 dB")
  io.println("  - Gap: Diferença entre teórico e real (menor = melhor)")
  io.println("  - Cosine: Similaridade cosseno (1.0 = perfeito)")

  io.println("")
  io.println("INSIGHTS:")
  io.println("  - INT8: Gap pequeno = quantização eficiente")
  io.println("  - NF4: Melhor que Q4 uniforme por usar quantis normais")
  io.println("  - AWQ: Foca em canais salientes para minimizar erro")
}

// ============================================================================
// HELPERS
// ============================================================================

fn generate_calibration_data(
  num_samples: Int,
  features: Int,
) -> List(List(Float)) {
  list.range(1, num_samples)
  |> list.map(fn(_) {
    tensor.random_uniform([features])
    |> tensor.to_list
  })
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

fn pad_ratio(f: Float) -> String {
  let s = float_to_str(f)
  let len = string.length(s)
  case len < 4 {
    True -> string.repeat(" ", 4 - len) <> s
    False -> s
  }
}

fn pad_snr(f: Float) -> String {
  let s = float_to_str(f)
  let len = string.length(s)
  case len < 6 {
    True -> string.repeat(" ", 6 - len) <> s
    False -> string.slice(s, 0, 6)
  }
}

fn pad_gap(f: Float) -> String {
  let s = float_to_str(f)
  let len = string.length(s)
  case len < 6 {
    True -> string.repeat(" ", 6 - len) <> s
    False -> string.slice(s, 0, 6)
  }
}

fn pad_cos(f: Float) -> String {
  let s = float_to_str(f)
  let len = string.length(s)
  case len < 6 {
    True -> s <> string.repeat(" ", 6 - len)
    False -> string.slice(s, 0, 6)
  }
}
