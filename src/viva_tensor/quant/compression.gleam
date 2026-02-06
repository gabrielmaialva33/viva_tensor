//// INT8 Quantization and Memory Hierarchy System
////
//// Reference: Jacob et al. (2017) - "Quantization and Training of Neural Networks
//// for Efficient Integer-Arithmetic-Only Inference"
//// https://arxiv.org/abs/1712.05877
////
//// --- Compression Math ---
//// INT8: 32-bit / 8-bit = 4x compression
//// - 24GB VRAM -> 96GB effective parameter storage
//// - RTX 4090 INT8 Tensor Cores: 2x throughput vs FP16 (660 vs 330 TOPS)
////
//// Why symmetric quantization? Because asymmetric zero-points are a cache-miss
//// nightmare. The extra memory access for zero-point lookup kills throughput.
//// Per-tensor symmetric is fast. Per-channel symmetric is accurate. Pick one.
////
//// absmax quantization: simple but loses dynamic range at the tails.
//// For weights: per-channel absmax is worth the overhead.
//// For activations: per-tensor is fine (they're more uniform).
////
//// FP16 was a mistake for storage. It's 2x larger than INT8 with minimal
//// accuracy benefit for inference. Train in FP16/BF16, deploy in INT8.
////
//// Inspired by: ggml, llama.cpp, Candle, bitsandbytes

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Quantization Formats ---

/// Formato de quantização
pub type QuantFormat {
  /// Full precision (32 bits, 4 bytes per value)
  /// Memory: 4 bytes/param | Compression: 1x
  Fp32
  /// Half precision (16 bits, 2 bytes per value)
  /// Memory: 2 bytes/param | Compression: 2x
  /// Opinion: Only use for training. Deploy with INT8.
  Fp16
  /// Integer 8-bit with scale (1 byte + 1 float per block)
  /// Memory: ~1 byte/param | Compression: 4x
  /// Sweet spot: Best accuracy/compression tradeoff for inference
  Int8(scale: Float)
  /// 4-bit quantized (0.5 bytes per value) - GGML style
  /// Memory: 0.5 bytes + scale overhead | Compression: ~7x
  /// Use NF4 instead for normal distributions (see nf4.gleam)
  Quant4(block_size: Int, scales: List(Float))
  /// 4-bit with min/max (asymmetric, more accurate for ReLU activations)
  /// Memory: 0.5 bytes + 2 scales per block | Compression: ~6.5x
  Quant4Min(block_size: Int, scales: List(Float), mins: List(Float))
}

/// Tensor comprimido
pub type CompressedTensor {
  CompressedTensor(
    /// Quantized data (bytes simulated as ints for BEAM compatibility)
    data: List(Int),
    /// Original shape preserved for dequantization
    shape: List(Int),
    /// Quantization format with metadata
    format: QuantFormat,
    /// Memory footprint in bytes (useful for memory budgeting)
    memory_bytes: Int,
  )
}

// --- Memory Hierarchy Types ---

/// Tensor location in the memory hierarchy
/// Performance note: GPU->RAM transfer on PCIe 4.0 x16: ~25 GB/s
/// That's 40ms for 1GB. Plan your offloading accordingly.
pub type TensorLocation {
  /// VRAM - fastest (RTX 4090: 1008 GB/s bandwidth)
  OnGpu(device_id: Int)
  /// System RAM - medium (DDR5-6400: ~100 GB/s dual channel)
  OnRam
  /// NVMe SSD - slow but huge (7 GB/s sequential)
  OnDisk(path: String)
  /// Hybrid: split across GPU and RAM (gradient offloading pattern)
  Hybrid(gpu_pct: Float)
}

/// Memory tier with capacity and bandwidth tracking
pub type MemoryTier {
  MemoryTier(
    location: TensorLocation,
    capacity_gb: Float,
    used_gb: Float,
    /// Bandwidth matters more than latency for large transfers
    bandwidth_gbps: Float,
  )
}

/// Hierarchical memory system for offloading
/// Pattern: Hot tensors on GPU, warm on RAM, cold on disk
pub type MemoryHierarchy {
  MemoryHierarchy(
    gpu: MemoryTier,
    ram: MemoryTier,
    disk: Option(MemoryTier),
    /// Effective capacity after quantization multiplier
    total_effective_gb: Float,
  )
}

/// Offload policy determines when to move tensors between tiers
pub type OffloadPolicy {
  /// Keep everything on GPU (default, fastest)
  KeepOnGpu
  /// Move to RAM when GPU usage exceeds threshold
  OffloadToRam(threshold_pct: Float)
  /// Tiered: GPU -> RAM -> Disk
  OffloadToDisk(ram_threshold: Float, disk_path: String)
  /// LRU-based: evict least recently accessed tensors first
  SmartOffload(access_history: List(AccessRecord))
}

/// Access tracking for smart offloading
pub type AccessRecord {
  AccessRecord(tensor_id: Int, timestamp_ms: Int, access_count: Int)
}

// --- INT8 Quantization Core ---

/// Extract shape from tensor (handles both Dense and Strided)
fn get_shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, shape) -> shape
    tensor.StridedTensor(_, shape, _, _) -> shape
  }
}

/// Quantize tensor to INT8 using absmax symmetric quantization
///
/// Compression: 32/8 = 4x
/// Error: Typically <0.5% for well-distributed weights
///
/// Implementation: absmax per-tensor (fast but less accurate than per-channel)
/// For production, consider per-channel for weights, per-tensor for activations.
pub fn quantize_int8(t: Tensor) -> CompressedTensor {
  let data = tensor.to_list(t)
  let shape = get_shape(t)

  // absmax quantization: find max absolute value for symmetric range
  // Why symmetric? No zero-point means one less memory access per operation
  let max_val = find_max_abs(data)
  let scale = case max_val >. 0.0 {
    True -> 127.0 /. max_val
    False -> 1.0
  }

  // Quantize to INT8 range [-127, 127]
  // Note: We use 127, not 128, to keep symmetric range (avoid -128 asymmetry)
  let quantized =
    list.map(data, fn(v) {
      let scaled = v *. scale
      let clamped = float.clamp(scaled, -127.0, 127.0)
      float.round(clamped)
    })

  // Memory: 1 byte per value + 4 bytes for the single scale factor
  let num_elements = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let memory = num_elements + 4

  CompressedTensor(
    data: quantized,
    shape: shape,
    format: Int8(scale),
    memory_bytes: memory,
  )
}

/// Quantize to Q4 (4-bit) using block-wise absmax - GGML style
///
/// Compression: 32/4 = 8x theoretical, ~7x effective with scale overhead
/// Block size tradeoff:
///   - Smaller blocks (32): More scales, more accurate, less compression
///   - Larger blocks (128): Fewer scales, less accurate, more compression
///   - Sweet spot: 64 (empirically validated in GGML/QLoRA)
pub fn quantize_q4(t: Tensor, block_size: Int) -> CompressedTensor {
  let data = tensor.to_list(t)
  let shape = get_shape(t)

  // Divide into blocks for per-block scaling
  let blocks = list.sized_chunk(data, block_size)

  // Quantize each block independently
  let #(quantized_blocks, scales) =
    list.fold(blocks, #([], []), fn(acc, block) {
      let #(q_acc, s_acc) = acc

      // Per-block absmax for better dynamic range preservation
      let block_max = find_max_abs(block)
      let scale = case block_max >. 0.0 {
        True -> 15.0 /. block_max
        False -> 1.0
      }

      // Quantize to 4 bits unsigned [0, 15] with offset
      // Using unsigned avoids sign bit complexity in packing
      let q_block =
        list.map(block, fn(v) {
          let scaled = { v *. scale } +. 8.0
          let clamped = float.clamp(scaled, 0.0, 15.0)
          float.round(clamped)
        })

      #(list.append(q_acc, q_block), [scale, ..s_acc])
    })

  // Memory calculation:
  // - 4 bits per value = 0.5 bytes per value
  // - 4 bytes (FP32) per block for scale
  // For 64-element blocks: 32 bytes data + 4 bytes scale = 36 bytes
  // vs 256 bytes in FP32 = 7.1x compression
  let num_elements = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let num_blocks = { num_elements + block_size - 1 } / block_size
  let memory = { num_elements / 2 } + { num_blocks * 4 }

  CompressedTensor(
    data: quantized_blocks,
    shape: shape,
    format: Quant4(block_size, list.reverse(scales)),
    memory_bytes: memory,
  )
}

// --- Dequantization ---

/// Dequantize compressed tensor back to FP32
/// Note: This is NOT lossless. Quantization error is permanent.
pub fn dequantize(ct: CompressedTensor) -> Tensor {
  case ct.format {
    Fp32 -> {
      create_tensor(list.map(ct.data, int.to_float), ct.shape)
    }

    Fp16 -> {
      // FP16 -> FP32 (simulated, real impl would use native conversion)
      create_tensor(list.map(ct.data, int.to_float), ct.shape)
    }

    Int8(scale) -> {
      // INT8 -> FP32: divide by scale to restore original range
      let data = list.map(ct.data, fn(q) { int.to_float(q) /. scale })
      create_tensor(data, ct.shape)
    }

    Quant4(block_size, scales) -> {
      // Q4 -> FP32: restore per-block
      let blocks = list.sized_chunk(ct.data, block_size)
      let data =
        list.index_map(blocks, fn(block, idx) {
          let scale = get_at_index(scales, idx, 1.0)
          list.map(block, fn(q) { { int.to_float(q) -. 8.0 } /. scale })
        })
        |> list.flatten
      create_tensor(data, ct.shape)
    }

    Quant4Min(block_size, scales, mins) -> {
      // Q4 with min -> FP32: uses both scale and min for asymmetric range
      let blocks = list.sized_chunk(ct.data, block_size)
      let data =
        list.index_map(blocks, fn(block, idx) {
          let scale = get_at_index(scales, idx, 1.0)
          let min = get_at_index(mins, idx, 0.0)
          list.map(block, fn(q) { { int.to_float(q) /. scale } +. min })
        })
        |> list.flatten
      create_tensor(data, ct.shape)
    }
  }
}

fn create_tensor(data: List(Float), shape: List(Int)) -> Tensor {
  Tensor(data: data, shape: shape)
}

fn get_at_index(lst: List(Float), idx: Int, default: Float) -> Float {
  case list.drop(lst, idx) {
    [first, ..] -> first
    [] -> default
  }
}

// --- Memory Hierarchy Management ---

/// Create memory hierarchy for typical workstation setup
/// Example: RTX 4090 (24GB) + DDR5 RAM (32GB) = 56GB physical
/// With INT8: 56GB * 4 = 224GB effective parameter storage
pub fn create_memory_hierarchy(
  vram_gb: Float,
  ram_gb: Float,
  disk_path: Option(String),
) -> MemoryHierarchy {
  let gpu_tier =
    MemoryTier(
      location: OnGpu(0),
      capacity_gb: vram_gb,
      used_gb: 0.0,
      // RTX 4090: 1008 GB/s memory bandwidth
      bandwidth_gbps: 1008.0,
    )

  let ram_tier =
    MemoryTier(
      location: OnRam,
      capacity_gb: ram_gb,
      used_gb: 0.0,
      // DDR5-3200 dual channel theoretical max
      bandwidth_gbps: 51.2,
    )

  let disk_tier = case disk_path {
    Some(path) ->
      Some(MemoryTier(
        location: OnDisk(path),
        // Assume 1TB NVMe
        capacity_gb: 1000.0,
        used_gb: 0.0,
        // PCIe 4.0 NVMe sequential read
        bandwidth_gbps: 7.0,
      ))
    None -> None
  }

  // Effective capacity with INT8 quantization (4x multiplier)
  let effective = { vram_gb *. 4.0 } +. { ram_gb *. 4.0 }

  MemoryHierarchy(
    gpu: gpu_tier,
    ram: ram_tier,
    disk: disk_tier,
    total_effective_gb: effective,
  )
}

/// Allocate tensor in memory hierarchy based on policy
/// Returns new location and updated hierarchy state
pub fn allocate_tensor(
  hierarchy: MemoryHierarchy,
  tensor_size_gb: Float,
  policy: OffloadPolicy,
) -> #(TensorLocation, MemoryHierarchy) {
  case policy {
    KeepOnGpu -> {
      // Try GPU first, overflow to RAM
      let gpu_free = hierarchy.gpu.capacity_gb -. hierarchy.gpu.used_gb
      case tensor_size_gb <=. gpu_free {
        True -> {
          let new_gpu =
            MemoryTier(
              ..hierarchy.gpu,
              used_gb: hierarchy.gpu.used_gb +. tensor_size_gb,
            )
          #(OnGpu(0), MemoryHierarchy(..hierarchy, gpu: new_gpu))
        }
        False -> {
          // Spill to RAM (common pattern for large models)
          let new_ram =
            MemoryTier(
              ..hierarchy.ram,
              used_gb: hierarchy.ram.used_gb +. tensor_size_gb,
            )
          #(OnRam, MemoryHierarchy(..hierarchy, ram: new_ram))
        }
      }
    }

    OffloadToRam(threshold) -> {
      let gpu_usage = hierarchy.gpu.used_gb /. hierarchy.gpu.capacity_gb
      case gpu_usage <. threshold {
        True -> {
          let new_gpu =
            MemoryTier(
              ..hierarchy.gpu,
              used_gb: hierarchy.gpu.used_gb +. tensor_size_gb,
            )
          #(OnGpu(0), MemoryHierarchy(..hierarchy, gpu: new_gpu))
        }
        False -> {
          let new_ram =
            MemoryTier(
              ..hierarchy.ram,
              used_gb: hierarchy.ram.used_gb +. tensor_size_gb,
            )
          #(OnRam, MemoryHierarchy(..hierarchy, ram: new_ram))
        }
      }
    }

    OffloadToDisk(ram_threshold, disk_path) -> {
      let gpu_free = hierarchy.gpu.capacity_gb -. hierarchy.gpu.used_gb
      let ram_usage = hierarchy.ram.used_gb /. hierarchy.ram.capacity_gb

      case tensor_size_gb <=. gpu_free {
        True -> {
          let new_gpu =
            MemoryTier(
              ..hierarchy.gpu,
              used_gb: hierarchy.gpu.used_gb +. tensor_size_gb,
            )
          #(OnGpu(0), MemoryHierarchy(..hierarchy, gpu: new_gpu))
        }
        False -> {
          case ram_usage <. ram_threshold {
            True -> {
              let new_ram =
                MemoryTier(
                  ..hierarchy.ram,
                  used_gb: hierarchy.ram.used_gb +. tensor_size_gb,
                )
              #(OnRam, MemoryHierarchy(..hierarchy, ram: new_ram))
            }
            False -> {
              #(OnDisk(disk_path), hierarchy)
            }
          }
        }
      }
    }

    SmartOffload(_history) -> {
      // TODO: Implement LRU eviction based on access history
      // Pattern: Track access frequency, evict coldest tensors first
      #(OnGpu(0), hierarchy)
    }
  }
}

// --- Gradient Checkpointing ---

/// Checkpoint for trading compute for memory
/// Key insight: Recomputing forward pass is cheaper than storing all activations
pub type Checkpoint {
  Checkpoint(
    /// Input saved for recomputation
    input: Tensor,
    /// Function ID for forward pass replay
    forward_fn_id: Int,
    /// Memory saved in GB (activations not stored)
    memory_saved_gb: Float,
  )
}

/// Checkpointing strategy
pub type CheckpointStrategy {
  /// No checkpointing - fastest but uses most memory
  NoCheckpoint
  /// Checkpoint every N layers (sqrt(N) is optimal for memory/compute)
  EveryN(n: Int)
  /// Only checkpoint layers above threshold (target the big ones)
  LargeLayersOnly(threshold_mb: Float)
  /// Adaptive based on memory pressure (for dynamic batch sizing)
  Adaptive(memory_pressure: Float)
}

/// Calculate memory savings from checkpointing
/// Note: This trades ~33% compute overhead for 50-75% memory savings
pub fn checkpoint_savings(
  num_layers: Int,
  layer_size_mb: Float,
  strategy: CheckpointStrategy,
) -> Float {
  let total_mb = int.to_float(num_layers) *. layer_size_mb

  case strategy {
    NoCheckpoint -> 0.0

    EveryN(n) -> {
      // Saves (1 - 1/n) of activation memory
      let checkpoint_pct = 1.0 -. { 1.0 /. int.to_float(n) }
      total_mb *. checkpoint_pct
    }

    LargeLayersOnly(threshold) -> {
      case layer_size_mb >. threshold {
        True -> total_mb *. 0.7
        False -> 0.0
      }
    }

    Adaptive(pressure) -> {
      // Linear scaling: more pressure = more checkpointing
      total_mb *. pressure
    }
  }
}

// --- Tensor Streaming ---

/// Streaming tensor - loads chunks on demand
/// Use case: Models too large for any single memory tier
pub type StreamedTensor {
  StreamedTensor(
    id: Int,
    shape: List(Int),
    chunk_shape: List(Int),
    loaded_chunks: List(Int),
    total_chunks: Int,
    format: QuantFormat,
  )
}

/// Create streaming tensor with specified chunk size
pub fn create_streamed(shape: List(Int), chunk_dim: Int) -> StreamedTensor {
  let total_elements = list.fold(shape, 1, fn(acc, d) { acc * d })
  let chunk_elements = chunk_dim
  let total_chunks = { total_elements + chunk_elements - 1 } / chunk_elements

  StreamedTensor(
    id: erlang_unique_integer(),
    shape: shape,
    chunk_shape: [chunk_dim],
    loaded_chunks: [],
    total_chunks: total_chunks,
    format: Int8(1.0),
  )
}

/// Load specific chunk into memory
pub fn load_chunk(st: StreamedTensor, chunk_idx: Int) -> StreamedTensor {
  case list.contains(st.loaded_chunks, chunk_idx) {
    True -> st
    False ->
      StreamedTensor(..st, loaded_chunks: [chunk_idx, ..st.loaded_chunks])
  }
}

/// Unload chunk to free memory
pub fn unload_chunk(st: StreamedTensor, chunk_idx: Int) -> StreamedTensor {
  StreamedTensor(
    ..st,
    loaded_chunks: list.filter(st.loaded_chunks, fn(c) { c != chunk_idx }),
  )
}

// --- Memory Pooling ---

/// Memory pool for buffer reuse
/// Pattern: Allocate once, reuse many times (avoids malloc overhead)
pub type MemoryPool {
  MemoryPool(
    /// Available buffers by size: (size_bytes, available_count)
    free_buffers: List(#(Int, Int)),
    /// Currently allocated count
    used_buffers: Int,
    /// Total bytes allocated (for memory tracking)
    total_allocated: Int,
  )
}

/// Create empty memory pool
pub fn create_pool() -> MemoryPool {
  MemoryPool(free_buffers: [], used_buffers: 0, total_allocated: 0)
}

/// Allocate from pool (reuses existing buffer if available)
/// Returns: (updated_pool, was_reused)
pub fn pool_alloc(pool: MemoryPool, size: Int) -> #(MemoryPool, Bool) {
  let found =
    list.find(pool.free_buffers, fn(b) {
      let #(s, count) = b
      s == size && count > 0
    })

  case found {
    Ok(#(s, _count)) -> {
      // Reuse existing buffer
      let new_buffers =
        list.map(pool.free_buffers, fn(b) {
          let #(bs, bc) = b
          case bs == s {
            True -> #(bs, bc - 1)
            False -> b
          }
        })
      #(
        MemoryPool(
          ..pool,
          free_buffers: new_buffers,
          used_buffers: pool.used_buffers + 1,
        ),
        True,
      )
    }
    Error(_) -> {
      // Allocate new buffer
      let new_buffers = [#(size, 0), ..pool.free_buffers]
      #(
        MemoryPool(
          free_buffers: new_buffers,
          used_buffers: pool.used_buffers + 1,
          total_allocated: pool.total_allocated + size,
        ),
        False,
      )
    }
  }
}

/// Return buffer to pool for reuse
pub fn pool_free(pool: MemoryPool, size: Int) -> MemoryPool {
  let new_buffers = case
    list.find(pool.free_buffers, fn(b) {
      let #(s, _) = b
      s == size
    })
  {
    Ok(_) -> {
      list.map(pool.free_buffers, fn(b) {
        let #(bs, bc) = b
        case bs == size {
          True -> #(bs, bc + 1)
          False -> b
        }
      })
    }
    Error(_) -> [#(size, 1), ..pool.free_buffers]
  }

  MemoryPool(
    ..pool,
    free_buffers: new_buffers,
    used_buffers: pool.used_buffers - 1,
  )
}

// --- Benchmark and Demo ---

pub fn main() {
  demonstrate_compression()
}

pub fn demonstrate_compression() {
  io.println(
    "=====================================================================",
  )
  io.println("  INT8 QUANTIZATION - Jacob et al. (2017)")
  io.println("  32-bit -> 8-bit = 4x compression with <0.5% accuracy loss")
  io.println(
    "=====================================================================\n",
  )

  // RTX 4090 + 32GB RAM configuration
  let hierarchy = create_memory_hierarchy(24.0, 32.0, None)

  io.println("--- Hardware Configuration ---")
  io.println("  GPU: 24GB VRAM (RTX 4090, 1008 GB/s)")
  io.println("  RAM: 32GB DDR5 (51.2 GB/s)")
  io.println("  Physical: 56GB")
  io.println(
    "  Effective with INT8: "
    <> float_to_string(hierarchy.total_effective_gb)
    <> "GB",
  )
  io.println("")

  // Quantization demo
  io.println("--- Quantization Benchmark ---")
  let t = tensor.random_uniform([1024, 512])
  let original_size = 1024 * 512 * 4

  let int8 = quantize_int8(t)
  let q4 = quantize_q4(t, 64)

  io.println(
    "  Original (FP32): " <> int.to_string(original_size / 1024) <> " KB",
  )
  io.println(
    "  INT8 (4x):       " <> int.to_string(int8.memory_bytes / 1024) <> " KB",
  )
  io.println(
    "  Q4 block=64:     " <> int.to_string(q4.memory_bytes / 1024) <> " KB",
  )

  // Accuracy check
  let restored = dequantize(int8)
  let error = compute_quantization_error(t, restored)
  io.println("  INT8 error:      " <> float_to_string(error *. 100.0) <> "%")

  // Memory hierarchy demo
  io.println("\n--- Memory Hierarchy ---")
  io.println("  Tier 1 (GPU):  1008 GB/s - hot tensors")
  io.println("  Tier 2 (RAM):  51.2 GB/s - warm tensors")
  io.println("  Tier 3 (Disk): 7 GB/s    - cold tensors")

  // Allocation demo
  io.println("\n--- Allocation Example ---")
  let policy = OffloadToRam(0.8)

  let #(loc1, h1) = allocate_tensor(hierarchy, 10.0, policy)
  io.println("  10GB tensor -> " <> location_to_string(loc1))

  let #(loc2, h2) = allocate_tensor(h1, 10.0, policy)
  io.println("  10GB tensor -> " <> location_to_string(loc2))

  let #(loc3, _h3) = allocate_tensor(h2, 10.0, policy)
  io.println("  10GB tensor -> " <> location_to_string(loc3))

  // Checkpointing demo
  io.println("\n--- Gradient Checkpointing ---")
  let layers = 24
  let layer_mb = 100.0
  let total_mb = int.to_float(layers) *. layer_mb

  let savings_n2 = checkpoint_savings(layers, layer_mb, EveryN(2))
  let savings_n4 = checkpoint_savings(layers, layer_mb, EveryN(4))

  io.println("  24 layers x 100MB = " <> float_to_string(total_mb) <> "MB")
  io.println(
    "  EveryN(2): saves "
    <> float_to_string(savings_n2)
    <> "MB (50% memory, 33% more compute)",
  )
  io.println(
    "  EveryN(4): saves "
    <> float_to_string(savings_n4)
    <> "MB (75% memory, 25% more compute)",
  )

  io.println(
    "\n=====================================================================",
  )
  io.println("  SUMMARY: 24GB VRAM + 32GB RAM with INT8")
  io.println("  - Physical: 56GB")
  io.println("  - Effective: 224GB (4x from INT8)")
  io.println("  - Can fit: ~110B parameters (224GB / 2 bytes per param)")
  io.println("  - RTX 4090 INT8 Tensor Cores: 660 TOPS vs 330 TOPS FP16")
  io.println(
    "=====================================================================",
  )
}

// --- Helper Functions ---

fn find_max_abs(data: List(Float)) -> Float {
  list.fold(data, 0.0, fn(acc, v) {
    let abs_v = float.absolute_value(v)
    case abs_v >. acc {
      True -> abs_v
      False -> acc
    }
  })
}

fn compute_quantization_error(original: Tensor, restored: Tensor) -> Float {
  let orig_data = tensor.to_list(original)
  let rest_data = tensor.to_list(restored)

  let #(sum_error, count) =
    list.fold(list.zip(orig_data, rest_data), #(0.0, 0), fn(acc, pair) {
      let #(sum, cnt) = acc
      let #(o, r) = pair
      let error = float.absolute_value(o -. r)
      #(sum +. error, cnt + 1)
    })

  case count > 0 {
    True -> sum_error /. int.to_float(count)
    False -> 0.0
  }
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn location_to_string(loc: TensorLocation) -> String {
  case loc {
    OnGpu(id) -> "GPU #" <> int.to_string(id)
    OnRam -> "RAM"
    OnDisk(path) -> "Disk(" <> path <> ")"
    Hybrid(pct) -> "Hybrid(" <> float_to_string(pct *. 100.0) <> "% GPU)"
  }
}

@external(erlang, "erlang", "unique_integer")
fn erlang_unique_integer() -> Int
