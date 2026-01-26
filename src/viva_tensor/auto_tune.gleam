//// Auto-Tuning System - Self-Optimizing Tensor Library
////
//// Inspirado pelas pesquisas do HuggingChat + ggml + Candle
////
//// Features:
//// 1. GPU Auto-Detection (detecta VRAM, load, capabilities)
//// 2. Adaptive Quantization (int8 para inference, fp32 para training)
//// 3. Zero-Copy entre Gleam e Rust NIFs
//// 4. Auto Batch Size Optimizer (aprende o melhor batch para o hardware)
////
//// Target: RTX 4090 24GB VRAM + 32GB RAM

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor.{type Tensor}

// ============================================================================
// TIPOS - HARDWARE DETECTION
// ============================================================================

/// Device type detectado
pub type Device {
  Cuda(gpu_id: Int, vram_gb: Float)
  Metal(device_id: Int)
  Cpu(cores: Int)
}

/// Hardware profile completo
pub type HardwareProfile {
  HardwareProfile(
    device: Device,
    total_vram_gb: Float,
    available_vram_gb: Float,
    total_ram_gb: Float,
    gpu_load_pct: Float,
    optimal_batch_size: Int,
  )
}

/// Modo de quantização
pub type QuantMode {
  Inference
  // INT8 - 4x menos memória, mais rápido
  Training
  // FP32 - precisão total
  Adaptive
  // Escolhe automaticamente
}

/// Estado do auto-tuner
pub type AutoTuner {
  AutoTuner(
    hardware: HardwareProfile,
    quant_mode: QuantMode,
    history: List(BatchResult),
    current_batch_size: Int,
  )
}

/// Resultado de uma execução de batch
pub type BatchResult {
  BatchResult(batch_size: Int, duration_ms: Float, throughput: Float)
}

// ============================================================================
// DETECÇÃO DE HARDWARE
// ============================================================================

/// Detecta hardware disponível
pub fn detect_hardware() -> HardwareProfile {
  // Por enquanto retorna perfil do GATO-PC
  // Em produção: chamaria NIF para gfxinfo/nvml
  HardwareProfile(
    device: Cuda(gpu_id: 0, vram_gb: 24.0),
    total_vram_gb: 24.0,
    available_vram_gb: 20.0,
    // 4GB reservado para sistema
    total_ram_gb: 32.0,
    gpu_load_pct: 0.0,
    optimal_batch_size: 32,
  )
}

/// Cria auto-tuner para CPU-only
pub fn detect_cpu_only() -> HardwareProfile {
  HardwareProfile(
    device: Cpu(cores: 16),
    total_vram_gb: 0.0,
    available_vram_gb: 0.0,
    total_ram_gb: 32.0,
    gpu_load_pct: 0.0,
    optimal_batch_size: 8,
  )
}

// ============================================================================
// AUTO-TUNER
// ============================================================================

/// Cria novo auto-tuner
pub fn new() -> AutoTuner {
  let hardware = detect_hardware()
  let batch_size = get_default_batch_size(hardware.device)

  AutoTuner(
    hardware: hardware,
    quant_mode: Adaptive,
    history: [],
    current_batch_size: batch_size,
  )
}

/// Batch size padrão baseado no device
fn get_default_batch_size(device: Device) -> Int {
  case device {
    Cuda(_, vram_gb) -> {
      // Escala com VRAM disponível
      case vram_gb >=. 20.0 {
        True -> 64
        // RTX 4090 com bastante VRAM
        False ->
          case vram_gb >=. 10.0 {
            True -> 32
            // RTX 3080/4080
            False -> 16
            // GPUs menores
          }
      }
    }
    Metal(_) -> 16
    // Apple Silicon é eficiente com batches menores
    Cpu(_) -> 8
    // BEAM CPU-only
  }
}

/// Registra resultado de execução e otimiza
pub fn profile(
  tuner: AutoTuner,
  batch_size: Int,
  duration_ms: Float,
) -> AutoTuner {
  let throughput = int.to_float(batch_size) /. duration_ms

  let result =
    BatchResult(
      batch_size: batch_size,
      duration_ms: duration_ms,
      throughput: throughput,
    )

  // Adiciona ao histórico (mantém últimos 20)
  let new_history = case list.length(tuner.history) >= 20 {
    True -> [result, ..list.take(tuner.history, 19)]
    False -> [result, ..tuner.history]
  }

  // Encontra batch size ótimo
  let optimal = find_optimal_batch_size(new_history, tuner.hardware)

  AutoTuner(..tuner, history: new_history, current_batch_size: optimal)
}

/// Encontra batch size com maior throughput
fn find_optimal_batch_size(
  history: List(BatchResult),
  hw: HardwareProfile,
) -> Int {
  case history {
    [] -> get_default_batch_size(hw.device)
    _ -> {
      // Agrupa por batch size e calcula média de throughput
      let best =
        history
        |> list.fold(BatchResult(0, 0.0, 0.0), fn(acc, r) {
          case r.throughput >. acc.throughput {
            True -> r
            False -> acc
          }
        })

      // Ajusta para memória disponível
      adjust_for_memory(best.batch_size, hw)
    }
  }
}

/// Ajusta batch size para memória disponível
fn adjust_for_memory(batch_size: Int, hw: HardwareProfile) -> Int {
  let tensor_size_mb = 2.0
  // ~2MB por tensor 512d fp32
  let batch_memory_gb = int.to_float(batch_size) *. tensor_size_mb /. 1024.0

  // Deixa 20% de headroom
  let max_memory = hw.available_vram_gb *. 0.8

  case batch_memory_gb >. max_memory {
    True -> {
      let adjusted =
        float.round(max_memory /. tensor_size_mb *. 1024.0)
        |> int.max(1)
      adjusted
    }
    False -> batch_size
  }
}

// ============================================================================
// QUANTIZAÇÃO ADAPTATIVA
// ============================================================================

/// Contexto de quantização
pub type QuantContext {
  QuantContext(mode: QuantMode, scales: List(Float))
}

/// Cria contexto de quantização
pub fn new_quant_context(mode: QuantMode) -> QuantContext {
  QuantContext(mode: mode, scales: [])
}

/// Decide modo de quantização baseado na operação
pub fn should_quantize(ctx: QuantContext, is_inference: Bool) -> Bool {
  case ctx.mode {
    Inference -> True
    Training -> False
    Adaptive -> is_inference
  }
}

/// Calcula escala para quantização absmax (int8)
pub fn compute_scale(tensor: Tensor) -> Float {
  let max_val = tensor.max(tensor)
  case max_val >. 0.0 {
    True -> 127.0 /. max_val
    False -> 1.0
  }
}

// ============================================================================
// MEMORY PRESSURE HANDLING
// ============================================================================

/// Verifica pressão de memória
pub fn check_memory_pressure(hw: HardwareProfile) -> MemoryPressure {
  let usage_pct = 1.0 -. { hw.available_vram_gb /. hw.total_vram_gb }

  case usage_pct {
    p if p >=. 0.9 -> Critical
    p if p >=. 0.7 -> High
    p if p >=. 0.5 -> Medium
    _ -> Low
  }
}

pub type MemoryPressure {
  Low
  Medium
  High
  Critical
}

/// Estratégia baseada em pressão de memória
pub fn get_memory_strategy(pressure: MemoryPressure) -> MemoryStrategy {
  case pressure {
    Critical ->
      MemoryStrategy(
        batch_size_mult: 0.25,
        quant_mode: Inference,
        gc_aggressive: True,
      )
    High ->
      MemoryStrategy(
        batch_size_mult: 0.5,
        quant_mode: Inference,
        gc_aggressive: True,
      )
    Medium ->
      MemoryStrategy(
        batch_size_mult: 0.75,
        quant_mode: Adaptive,
        gc_aggressive: False,
      )
    Low ->
      MemoryStrategy(
        batch_size_mult: 1.0,
        quant_mode: Training,
        gc_aggressive: False,
      )
  }
}

pub type MemoryStrategy {
  MemoryStrategy(
    batch_size_mult: Float,
    quant_mode: QuantMode,
    gc_aggressive: Bool,
  )
}

// ============================================================================
// BENCHMARK E PROFILE
// ============================================================================

/// Roda profile completo do hardware
pub fn run_hardware_profile() {
  io.println(
    "╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  AUTO-TUNE HARDWARE PROFILE                                     ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  let hw = detect_hardware()

  io.println("HARDWARE DETECTADO:")
  print_device(hw.device)
  io.println("  VRAM Total: " <> float_to_string(hw.total_vram_gb) <> " GB")
  io.println(
    "  VRAM Disponível: " <> float_to_string(hw.available_vram_gb) <> " GB",
  )
  io.println("  RAM Total: " <> float_to_string(hw.total_ram_gb) <> " GB")
  io.println("  GPU Load: " <> float_to_string(hw.gpu_load_pct) <> "%")
  io.println("  Batch Size Ótimo: " <> int.to_string(hw.optimal_batch_size))

  let pressure = check_memory_pressure(hw)
  io.println("\nMEMORY PRESSURE: " <> pressure_to_string(pressure))

  let strategy = get_memory_strategy(pressure)
  io.println("ESTRATÉGIA:")
  io.println("  Batch Mult: " <> float_to_string(strategy.batch_size_mult))
  io.println("  Quant Mode: " <> quant_mode_to_string(strategy.quant_mode))
  io.println("  GC Agressivo: " <> bool_to_string(strategy.gc_aggressive))

  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  RECOMENDAÇÕES PARA RTX 4090 24GB + 32GB RAM:                   ║",
  )
  io.println(
    "║                                                                  ║",
  )
  io.println(
    "║  1. Batch Size: 64 (pode ir até 128 com INT8)                   ║",
  )
  io.println(
    "║  2. Quantização: INT8 para inference (4x menos VRAM)            ║",
  )
  io.println(
    "║  3. Memory Pool: Pre-alocar 20GB para tensores                  ║",
  )
  io.println(
    "║  4. Zero-Copy: Usar Binary refs entre Gleam e Rust              ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝",
  )
}

pub fn main() {
  run_hardware_profile()
}

// ============================================================================
// HELPERS
// ============================================================================

fn print_device(device: Device) {
  case device {
    Cuda(id, vram) ->
      io.println(
        "  Device: CUDA GPU #"
        <> int.to_string(id)
        <> " ("
        <> float_to_string(vram)
        <> "GB)",
      )
    Metal(id) -> io.println("  Device: Metal #" <> int.to_string(id))
    Cpu(cores) ->
      io.println("  Device: CPU (" <> int.to_string(cores) <> " cores)")
  }
}

fn pressure_to_string(p: MemoryPressure) -> String {
  case p {
    Low -> "LOW (tudo ok)"
    Medium -> "MEDIUM (monitorar)"
    High -> "HIGH (reduzir batch)"
    Critical -> "CRITICAL (emergency mode)"
  }
}

fn quant_mode_to_string(m: QuantMode) -> String {
  case m {
    Inference -> "INT8 (inference)"
    Training -> "FP32 (training)"
    Adaptive -> "ADAPTIVE"
  }
}

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "Sim"
    False -> "Não"
  }
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}
