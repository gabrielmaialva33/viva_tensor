//// viva_tensor Demo - Demonstração completa da biblioteca
////
//// Roda com: gleam run -m viva_tensor/demo

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string

import viva_tensor/awq
import viva_tensor/compression
import viva_tensor/flash_attention
import viva_tensor/nf4
import viva_tensor/sparsity
import viva_tensor/tensor.{type Tensor, Tensor}

// ============================================================================
// MAIN
// ============================================================================

pub fn main() {
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║          viva_tensor - Pure Gleam Tensor Library              ║",
  )
  io.println(
    "║                       DEMO COMPLETA                           ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // 1. Operações básicas
  demo_basic_ops()

  // 2. Quantização INT8
  demo_int8_quantization()

  // 3. NF4 (QLoRA style)
  demo_nf4_quantization()

  // 4. AWQ (Activation-aware)
  demo_awq_quantization()

  // 5. Flash Attention
  demo_flash_attention()

  // 6. 2:4 Sparsity
  demo_sparsity()

  // 7. Combinando técnicas
  demo_combined()

  io.println("")
  io.println("═══════════════════════════════════════════════════════════════")
  io.println("                    DEMO CONCLUÍDA!")
  io.println("═══════════════════════════════════════════════════════════════")
}

// ============================================================================
// 1. OPERAÇÕES BÁSICAS
// ============================================================================

fn demo_basic_ops() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 1. OPERAÇÕES BÁSICAS DE TENSORS                            │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  // Criar tensors
  let a = tensor.zeros([2, 3])
  let b = tensor.ones([2, 3])
  let c = tensor.random_uniform([2, 3])

  io.println("  zeros([2,3]): " <> tensor_preview(a))
  io.println("  ones([2,3]):  " <> tensor_preview(b))
  io.println("  random([2,3]): " <> tensor_preview(c))

  // Operações
  let sum_result = tensor.add(a, b)
  io.println("  zeros + ones: " <> result_tensor_preview(sum_result))

  let scaled = tensor.scale(b, 5.0)
  io.println("  ones * 5.0:   " <> tensor_preview(scaled))

  // Matmul
  let mat_a = tensor.random_uniform([3, 4])
  let mat_b = tensor.random_uniform([4, 2])
  let matmul_result = tensor.matmul(mat_a, mat_b)
  io.println("  matmul([3,4], [4,2]): " <> result_shape_str(matmul_result))

  // Stats
  let random_data = tensor.random_normal([100], 0.0, 1.0)
  io.println(
    "  random_normal: mean="
    <> float_to_str(tensor.mean(random_data))
    <> ", std="
    <> float_to_str(tensor.std(random_data)),
  )

  io.println("")
}

// ============================================================================
// 2. INT8 QUANTIZATION
// ============================================================================

fn demo_int8_quantization() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 2. INT8 QUANTIZATION (4x compression)                      │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  // Criar tensor de pesos simulando uma layer de rede neural
  let weights = tensor.random_normal([256, 256], 0.0, 0.5)
  let original_size = 256 * 256 * 4
  // FP32 = 4 bytes

  io.println(
    "  Original: "
    <> int.to_string(256 * 256)
    <> " floats = "
    <> format_bytes(original_size),
  )

  // Quantizar
  let quantized = compression.quantize_int8(weights)
  io.println(
    "  Quantizado: " <> int.to_string(quantized.memory_bytes) <> " bytes",
  )

  let compression_ratio =
    int.to_float(original_size) /. int.to_float(quantized.memory_bytes)
  io.println("  Compressão: " <> float_to_str(compression_ratio) <> "x")

  // Dequantizar e medir erro
  let recovered = compression.dequantize(quantized)
  let error = compute_error(weights, recovered)
  io.println("  Erro médio: " <> float_to_str(error *. 100.0) <> "%")

  io.println("")
}

// ============================================================================
// 3. NF4 QUANTIZATION
// ============================================================================

fn demo_nf4_quantization() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 3. NF4 QUANTIZATION - QLoRA Style (7.5x compression)       │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  let weights = tensor.random_normal([512, 512], 0.0, 0.3)
  let original_size = 512 * 512 * 4

  io.println("  Original: " <> format_bytes(original_size))

  // NF4 quantização
  let config = nf4.default_config()
  let quantized = nf4.quantize(weights, config)

  io.println(
    "  NF4 quantizado: " <> int.to_string(quantized.memory_bytes) <> " bytes",
  )
  io.println(
    "  Compressão: " <> float_to_str(quantized.compression_ratio) <> "x",
  )

  // Dequantizar
  let recovered = nf4.dequantize(quantized)
  let error = compute_error(weights, recovered)
  io.println("  Erro médio: " <> float_to_str(error *. 100.0) <> "%")

  // Double quantization
  let dq_quantized = nf4.double_quantize(weights, config)
  let dq_ratio =
    int.to_float(original_size) /. int.to_float(dq_quantized.memory_bytes)
  io.println("  NF4+DQ: " <> float_to_str(dq_ratio) <> "x compressão")

  io.println("")
}

// ============================================================================
// 4. AWQ QUANTIZATION
// ============================================================================

fn demo_awq_quantization() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 4. AWQ - Activation-aware (MLSys 2024 Best Paper)          │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  // Weights como Tensor - 256x256 matrix
  let weights = tensor.random_normal([256, 256], 0.0, 0.3)

  // Calibration data como List(List(Float)) - 64 samples, 256 features
  let activations_tensor = tensor.random_uniform([64, 256])
  let calibration_data = tensor_to_matrix(activations_tensor, 256)

  let original_size = 256 * 256 * 4

  io.println("  Weights: [256, 256] = " <> format_bytes(original_size))
  io.println("  Calibration: [64, 256] (activations batch)")

  // AWQ quantização
  let config = awq.default_config()
  let quantized = awq.quantize_awq(weights, calibration_data, config)

  io.println(
    "  AWQ quantizado: " <> int.to_string(quantized.memory_bytes) <> " bytes",
  )

  let compression_ratio =
    int.to_float(original_size) /. int.to_float(quantized.memory_bytes)
  io.println("  Compressão: " <> float_to_str(compression_ratio) <> "x")

  // Dequantizar
  let recovered = awq.dequantize_awq(quantized)
  let error = compute_error(weights, recovered)
  io.println("  Erro médio: " <> float_to_str(error *. 100.0) <> "%")

  io.println("")
}

// ============================================================================
// 5. FLASH ATTENTION
// ============================================================================

fn demo_flash_attention() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 5. FLASH ATTENTION - O(n) Memory                           │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  // Simular Q, K, V para attention
  let seq_len = 128
  let head_dim = 64

  let q = tensor.random_normal([seq_len, head_dim], 0.0, 0.1)
  let k = tensor.random_normal([seq_len, head_dim], 0.0, 0.1)
  let v = tensor.random_normal([seq_len, head_dim], 0.0, 0.1)

  io.println("  Sequence length: " <> int.to_string(seq_len))
  io.println("  Head dimension: " <> int.to_string(head_dim))

  // Naive attention (O(n²) memory)
  let #(naive_output, naive_mem) =
    flash_attention.naive_attention(q, k, v, 0.125)
  io.println("  Naive attention memory: " <> format_bytes(naive_mem))

  // Flash attention (O(n) memory)
  let config = flash_attention.default_config(head_dim)
  let flash_result = flash_attention.flash_attention(q, k, v, config)
  io.println(
    "  Flash attention memory: " <> format_bytes(flash_result.memory_bytes),
  )
  io.println(
    "  Memory saved: " <> float_to_str(flash_result.memory_saved_percent) <> "%",
  )

  // Verificar que outputs são similares
  let diff = compute_error(naive_output, flash_result.output)
  io.println("  Output difference: " <> float_to_str(diff *. 100.0) <> "%")

  io.println("")
}

// ============================================================================
// 6. 2:4 SPARSITY
// ============================================================================

fn demo_sparsity() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 6. 2:4 STRUCTURED SPARSITY - Tensor Cores                  │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  // Criar tensor para sparsity
  let weights = tensor.random_normal([256, 256], 0.0, 0.5)
  let original_size = 256 * 256 * 4

  io.println("  Original: [256, 256] = " <> format_bytes(original_size))

  // Aplicar 2:4 sparsity
  let sparse = sparsity.prune_24_magnitude(weights)

  io.println("  Sparse memory: " <> format_bytes(sparse.memory_bytes))
  io.println("  Sparsity: " <> float_to_str(sparse.sparsity_percent) <> "%")

  let compression_ratio =
    int.to_float(original_size) /. int.to_float(sparse.memory_bytes)
  io.println("  Compression: " <> float_to_str(compression_ratio) <> "x")

  // Decompress e medir erro
  let recovered = sparsity.decompress(sparse)
  let error = compute_error(weights, recovered)
  io.println("  Approximation error: " <> float_to_str(error))

  // Sparse matmul
  let dense_b = tensor.random_normal([256, 64], 0.0, 0.5)
  let #(_result, speedup) = sparsity.sparse_matmul(sparse, dense_b)
  io.println("  Theoretical speedup: " <> float_to_str(speedup) <> "x")

  io.println("")
}

// ============================================================================
// 7. COMBINANDO TÉCNICAS
// ============================================================================

fn demo_combined() {
  io.println("┌─────────────────────────────────────────────────────────────┐")
  io.println("│ 7. MEMORY MULTIPLICATION - Combining Techniques            │")
  io.println("└─────────────────────────────────────────────────────────────┘")

  // Simular um LLM de 7B params
  let params = 7_000_000_000
  let fp16_size = params * 2
  // 14GB

  io.println("  Model: 7B parameters")
  io.println(
    "  FP16 size: " <> int.to_string(fp16_size / 1_000_000_000) <> "GB",
  )

  io.println("")
  io.println("  ┌─────────────────────────────────────────────────────┐")
  io.println("  │ Technique         │ Size     │ Fits RTX 4090 24GB  │")
  io.println("  ├───────────────────┼──────────┼─────────────────────┤")
  io.println("  │ FP16              │ 14GB     │ [x] Tight           │")
  io.println("  │ INT8              │ 7GB      │ [x] + KV Cache      │")
  io.println("  │ NF4               │ 3.5GB    │ [x] + Batch=32      │")
  io.println("  │ NF4 + 2:4 Sparse  │ 1.75GB   │ [x] Multiple models!│")
  io.println("  └─────────────────────────────────────────────────────┘")

  // RTX 4090 memory multiplication
  let vram = 24
  // GB
  io.println("")
  io.println("  RTX 4090 24GB VRAM can effectively hold:")
  io.println(
    "    - FP16:           " <> int.to_string(vram / 14 * 7) <> "B params",
  )
  io.println(
    "    - INT8:           " <> int.to_string(vram / 7 * 7) <> "B params",
  )
  io.println("    - NF4:            " <> int.to_string(vram * 2) <> "B params")
  io.println("    - NF4 + Sparsity: " <> int.to_string(vram * 4) <> "B params")

  io.println("")
}

// ============================================================================
// HELPERS
// ============================================================================

fn tensor_preview(t: Tensor) -> String {
  let data = tensor.to_list(t)
  let preview =
    list.take(data, 5)
    |> list.map(float_to_str)
    |> string.join(", ")
  "[" <> preview <> "...]"
}

fn result_tensor_preview(r: Result(Tensor, tensor.TensorError)) -> String {
  case r {
    Ok(t) -> tensor_preview(t)
    Error(_) -> "Error"
  }
}

fn result_shape_str(r: Result(Tensor, tensor.TensorError)) -> String {
  case r {
    Ok(t) -> shape_to_string(get_shape(t))
    Error(_) -> "Error"
  }
}

fn get_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

fn shape_to_string(shape: List(Int)) -> String {
  "[" <> list.map(shape, int.to_string) |> string.join(", ") <> "]"
}

fn float_to_str(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn format_bytes(bytes: Int) -> String {
  case bytes {
    b if b >= 1_073_741_824 ->
      float_to_str(int.to_float(b) /. 1_073_741_824.0) <> "GB"
    b if b >= 1_048_576 -> float_to_str(int.to_float(b) /. 1_048_576.0) <> "MB"
    b if b >= 1024 -> int.to_string(b / 1024) <> "KB"
    b -> int.to_string(b) <> "B"
  }
}

fn compute_error(original: Tensor, recovered: Tensor) -> Float {
  let orig_data = tensor.to_list(original)
  let rec_data = tensor.to_list(recovered)

  let diffs =
    list.map2(orig_data, rec_data, fn(a, b) { float.absolute_value(a -. b) })

  let sum = list.fold(diffs, 0.0, fn(acc, x) { acc +. x })
  sum /. int.to_float(list.length(diffs))
}

/// Convert tensor to matrix (List of List)
fn tensor_to_matrix(t: Tensor, cols: Int) -> List(List(Float)) {
  let data = tensor.to_list(t)
  list.sized_chunk(data, cols)
}
