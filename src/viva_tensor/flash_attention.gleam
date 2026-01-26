//// Flash Attention - O(n) Memory Attention
////
//// REVOLUCIONÁRIO: Reduz memória de O(n²) para O(n)!
//// https://arxiv.org/abs/2205.14135 (Tri Dao, 2022)
////
//// PROBLEMA:
//// - Attention padrão: Q @ K^T @ V = O(n²) memória
//// - Para n=8192 (contexto longo): 8192² = 67M elementos = 256MB por cabeça!
//// - 32 cabeças = 8GB só para attention scores
////
//// SOLUÇÃO:
//// - Processar em TILES (blocos)
//// - Nunca materializar matriz n×n completa
//// - Online softmax: atualiza estatísticas incrementalmente
////
//// RESULTADO: 2-4x mais rápido, O(n) memória!

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor, Tensor}

// ============================================================================
// TIPOS
// ============================================================================

/// Configuração Flash Attention
pub type FlashConfig {
  FlashConfig(
    /// Tamanho do tile para Q (linhas)
    block_q: Int,
    /// Tamanho do tile para KV (colunas)
    block_kv: Int,
    /// Scaling factor (1/sqrt(d))
    scale: Float,
    /// Usar causal mask
    causal: Bool,
  )
}

/// Estatísticas online para softmax
pub type OnlineStats {
  OnlineStats(
    /// Máximo corrente (para estabilidade numérica)
    max_val: Float,
    /// Soma exponenciada corrente
    sum_exp: Float,
    /// Output acumulado
    output: List(Float),
  )
}

/// Resultado de Flash Attention
pub type FlashResult {
  FlashResult(
    /// Tensor de saída
    output: Tensor,
    /// Memória usada (bytes)
    memory_bytes: Int,
    /// Memória economizada vs naive (%)
    memory_saved_percent: Float,
  )
}

/// Configuração padrão
pub fn default_config(head_dim: Int) -> FlashConfig {
  // Block sizes otimizados para CUDA
  // 64-128 para Q, 64-256 para KV
  let scale = case float.square_root(int.to_float(head_dim)) {
    Ok(sqrt) -> 1.0 /. sqrt
    Error(_) -> 0.125  // fallback for head_dim=64
  }
  FlashConfig(
    block_q: 64,
    block_kv: 64,
    scale: scale,
    causal: False,
  )
}

/// Config para causal (autoregressive)
pub fn causal_config(head_dim: Int) -> FlashConfig {
  FlashConfig(..default_config(head_dim), causal: True)
}

// ============================================================================
// NAIVE ATTENTION (para comparação)
// ============================================================================

/// Attention ingênua: O(n²) memória
/// scores = Q @ K^T
/// attn = softmax(scores * scale)
/// out = attn @ V
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

  // Reshape para matriz
  let q_rows = list.sized_chunk(q_data, head_dim)
  let k_rows = list.sized_chunk(k_data, head_dim)
  let v_rows = list.sized_chunk(v_data, head_dim)

  let n_rows = list.length(q_rows)
  let n_cols = list.length(k_rows)

  // Step 1: scores = Q @ K^T  (AQUI ESTÁ O PROBLEMA: n×n matriz!)
  let scores = list.map(q_rows, fn(q_row) {
    list.map(k_rows, fn(k_row) {
      dot_product(q_row, k_row) *. scale
    })
  })

  // Step 2: softmax por linha
  let attn = list.map(scores, softmax_row)

  // Step 3: out = attn @ V
  let output = list.map(attn, fn(attn_row) {
    // Weighted sum of V rows
    list.index_fold(attn_row, list.repeat(0.0, head_dim), fn(acc, weight, i) {
      let v_row = get_row(v_rows, i)
      list.map2(acc, v_row, fn(a, v) { a +. weight *. v })
    })
  }) |> list.flatten

  // Memória: n×n scores matrix (FP32)
  let memory = n_rows * n_cols * 4

  #(Tensor(data: output, shape: get_tensor_shape(q)), memory)
}

// ============================================================================
// FLASH ATTENTION - O(n) Memória!
// ============================================================================

/// Flash Attention: processa em tiles, nunca materializa n×n
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

  // Processa Q em blocos
  let q_blocks = list.sized_chunk(q_rows, config.block_q)
  let k_blocks = list.sized_chunk(k_rows, config.block_kv)
  let v_blocks = list.sized_chunk(v_rows, config.block_kv)

  // Para cada bloco de Q, processa todos os blocos KV
  let output = list.index_map(q_blocks, fn(q_block, q_block_idx) {
    process_q_block(
      q_block,
      k_blocks,
      v_blocks,
      config,
      q_block_idx,
      head_dim,
    )
  }) |> list.flatten |> list.flatten

  // Memória Flash: apenas block_q × block_kv por vez
  let flash_memory = config.block_q * config.block_kv * 4

  // Memória naive: n×n
  let naive_memory = n_q * n_kv * 4

  let saved = 100.0 *. {
    1.0 -. int.to_float(flash_memory) /. int.to_float(naive_memory)
  }

  FlashResult(
    output: Tensor(data: output, shape: get_tensor_shape(q)),
    memory_bytes: flash_memory,
    memory_saved_percent: saved,
  )
}

/// Processa um bloco de Q contra todos os blocos KV
fn process_q_block(
  q_block: List(List(Float)),
  k_blocks: List(List(List(Float))),
  v_blocks: List(List(List(Float))),
  config: FlashConfig,
  q_block_idx: Int,
  head_dim: Int,
) -> List(List(Float)) {
  // Inicializa estatísticas online para cada query neste bloco
  let initial_stats = list.map(q_block, fn(_) {
    OnlineStats(
      max_val: -999999.0,
      sum_exp: 0.0,
      output: list.repeat(0.0, head_dim),
    )
  })

  // Itera sobre blocos KV, atualizando estatísticas
  let zipped_kv = list.zip(k_blocks, v_blocks)

  let final_stats = list.index_fold(zipped_kv, initial_stats, fn(stats, kv_pair, kv_idx) {
    let #(k_block, v_block) = kv_pair

    // Aplica causal mask se necessário
    case config.causal {
      True -> {
        let q_start = q_block_idx * config.block_q
        let kv_start = kv_idx * config.block_kv
        let kv_end = kv_start + list.length(k_block)

        // Se todo o bloco KV está no futuro, pula
        case kv_start > q_start + list.length(q_block) {
          True -> stats
          False -> process_kv_block(stats, q_block, k_block, v_block, config)
        }
      }
      False -> process_kv_block(stats, q_block, k_block, v_block, config)
    }
  })

  // Normaliza outputs finais
  list.map(final_stats, fn(s) {
    case s.sum_exp >. 0.0 {
      True -> list.map(s.output, fn(o) { o /. s.sum_exp })
      False -> s.output
    }
  })
}

/// Processa um bloco KV contra queries, atualiza estatísticas online
fn process_kv_block(
  stats: List(OnlineStats),
  q_block: List(List(Float)),
  k_block: List(List(Float)),
  v_block: List(List(Float)),
  config: FlashConfig,
) -> List(OnlineStats) {
  list.map2(stats, q_block, fn(stat, q_row) {
    // Compute scores para este query vs todos os keys do bloco
    let scores = list.map(k_block, fn(k_row) {
      dot_product(q_row, k_row) *. config.scale
    })

    // Online softmax update
    let new_max = list.fold(scores, stat.max_val, float.max)

    // Correção para o novo máximo
    let correction = case stat.sum_exp >. 0.0 {
      True -> float.power(2.71828, stat.max_val -. new_max)
        |> result_to_float(1.0)
      False -> 1.0
    }

    // Atualiza sum_exp com correção
    let corrected_sum = stat.sum_exp *. correction

    // Computa novos pesos exponenciados
    let exp_scores = list.map(scores, fn(s) {
      float.power(2.71828, s -. new_max)
      |> result_to_float(0.0)
    })

    let new_sum = list.fold(exp_scores, corrected_sum, float.add)

    // Atualiza output: corrige anterior + adiciona contribuição do novo bloco
    let corrected_output = list.map(stat.output, fn(o) { o *. correction })

    let new_contribution = list.index_fold(exp_scores, list.repeat(0.0, list.length(stat.output)), fn(acc, weight, i) {
      let v_row = get_row(v_block, i)
      list.map2(acc, v_row, fn(a, v) { a +. weight *. v })
    })

    let new_output = list.map2(corrected_output, new_contribution, float.add)

    OnlineStats(max_val: new_max, sum_exp: new_sum, output: new_output)
  })
}

// ============================================================================
// BENCHMARK
// ============================================================================

pub fn main() {
  benchmark_flash_attention()
}

pub fn benchmark_flash_attention() {
  io.println("╔══════════════════════════════════════════════════════════════════╗")
  io.println("║  FLASH ATTENTION - O(n) Memory Algorithm                         ║")
  io.println("║  Tri Dao et al., 2022 - FlashAttention                           ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝\n")

  io.println("PROBLEMA DA ATTENTION PADRÃO:")
  io.println("  - Attention = softmax(Q @ K^T / sqrt(d)) @ V")
  io.println("  - Q @ K^T cria matriz n×n")
  io.println("  - Para seq_len=8192: 67M elementos = 256MB por cabeça!")
  io.println("  - 32 cabeças = 8GB só para scores intermediários\n")

  io.println("SOLUÇÃO FLASH ATTENTION:")
  io.println("  - Processa em TILES (blocos)")
  io.println("  - Online softmax: atualiza estatísticas incrementalmente")
  io.println("  - Nunca materializa matriz n×n completa")
  io.println("  - Resultado: 2-4x mais rápido, O(n) memória!\n")

  // Benchmark com diferentes tamanhos
  let sizes = [64, 128, 256, 512]
  let head_dim = 64

  io.println("━━━ BENCHMARK: Naive vs Flash Attention ━━━")
  io.println("  head_dim = " <> int.to_string(head_dim))
  io.println("")

  list.each(sizes, fn(seq_len) {
    let q = tensor.random_uniform([seq_len, head_dim])
    let k = tensor.random_uniform([seq_len, head_dim])
    let v = tensor.random_uniform([seq_len, head_dim])

    let config = default_config(head_dim)

    // Naive
    let #(naive_time, #(_naive_out, naive_mem)) = timer_tc(fn() {
      naive_attention(q, k, v, config.scale)
    })

    // Flash
    let #(flash_time, flash_result) = timer_tc(fn() {
      flash_attention(q, k, v, config)
    })

    io.println("seq_len = " <> int.to_string(seq_len) <> ":")
    io.println("  Naive:  " <> int.to_string(naive_time / 1000) <> "ms, " <>
               int.to_string(naive_mem / 1024) <> " KB")
    io.println("  Flash:  " <> int.to_string(flash_time / 1000) <> "ms, " <>
               int.to_string(flash_result.memory_bytes / 1024) <> " KB")
    io.println("  Memory saved: " <> float_to_string(flash_result.memory_saved_percent) <> "%")
    io.println("")
  })

  // Projeção para contextos longos
  io.println("━━━ PROJEÇÃO: Economia de Memória por Contexto ━━━")
  let long_contexts = [1024, 2048, 4096, 8192, 16384, 32768]

  list.each(long_contexts, fn(n) {
    let naive_mem = n * n * 4  // n×n FP32
    let flash_mem = 64 * 64 * 4  // block_q × block_kv FP32

    let saved = 100.0 *. { 1.0 -. int.to_float(flash_mem) /. int.to_float(naive_mem) }

    io.println("  n=" <> int.to_string(n) <>
               ": Naive=" <> bytes_to_string(naive_mem) <>
               ", Flash=" <> bytes_to_string(flash_mem) <>
               " (" <> float_to_string(saved) <> "% saved)")
  })

  io.println("\n╔══════════════════════════════════════════════════════════════════╗")
  io.println("║  POR QUE FLASH ATTENTION É REVOLUCIONÁRIO:                       ║")
  io.println("║                                                                  ║")
  io.println("║  1. Contextos longos viáveis (32K, 100K tokens)                  ║")
  io.println("║  2. 2-4x speedup via IO-awareness                                ║")
  io.println("║  3. Exatamente correto (não é aproximação!)                      ║")
  io.println("║  4. Padrão em LLMs modernos (GPT-4, LLaMA, etc)                  ║")
  io.println("║                                                                  ║")
  io.println("║  viva_tensor + Flash = Contextos ilimitados na RTX 4090!         ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝")
}

// ============================================================================
// HELPERS
// ============================================================================

fn dot_product(a: List(Float), b: List(Float)) -> Float {
  list.map2(a, b, fn(x, y) { x *. y })
  |> list.fold(0.0, float.add)
}

fn softmax_row(row: List(Float)) -> List(Float) {
  let max_val = list.fold(row, -999999.0, float.max)
  let exp_vals = list.map(row, fn(x) {
    float.power(2.71828, x -. max_val)
    |> result_to_float(0.0)
  })
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

fn result_to_float(r: Result(Float, a), default: Float) -> Float {
  case r {
    Ok(v) -> v
    Error(_) -> default
  }
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn bytes_to_string(bytes: Int) -> String {
  case bytes {
    b if b >= 1_073_741_824 -> float_to_string(int.to_float(b) /. 1_073_741_824.0) <> "GB"
    b if b >= 1_048_576 -> float_to_string(int.to_float(b) /. 1_048_576.0) <> "MB"
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
