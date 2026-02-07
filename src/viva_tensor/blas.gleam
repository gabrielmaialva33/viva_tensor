// viva_tensor/blas.gleam - Intelligent BLAS Backend Selection
//
// Auto-detects and uses the BEST available BLAS implementation:
// 1. Intel MKL (817+ GFLOPS) - Best for Intel CPUs
// 2. OpenBLAS DYNAMIC_ARCH (500+ GFLOPS) - Best FOSS option
// 3. Zig SIMD GEMM (420 GFLOPS) - Fallback, always available
//
// Tuning is applied automatically based on CPU topology.

// No imports needed - uses FFI

/// BLAS backend types
pub type BlasBackend {
  IntelMKL
  OpenBLAS
  ZigSIMD
  Unknown
}

/// CPU topology info from NIF
pub type CpuTopology {
  CpuTopology(
    logical_cpus: Int,
    physical_cores: Int,
    l2_cache_kb: Int,
    l3_cache_kb: Int,
    has_avx2: Bool,
    has_avx512: Bool,
  )
}

/// Detect the best available BLAS backend
@external(erlang, "viva_tensor_blas", "detect_backend")
pub fn detect_backend() -> BlasBackend

/// Get CPU topology for tuning
@external(erlang, "viva_tensor_blas", "get_cpu_topology")
pub fn get_cpu_topology() -> CpuTopology

/// Configure MKL/OpenBLAS threads for optimal performance
@external(erlang, "viva_tensor_blas", "configure_threads")
pub fn configure_threads(num_threads: Int) -> Result(Nil, String)

/// Set thread affinity (scatter for hybrid CPUs)
@external(erlang, "viva_tensor_blas", "set_affinity")
pub fn set_affinity(mode: String) -> Result(Nil, String)

/// Auto-configure for maximum performance
pub fn auto_configure() -> Result(BlasBackend, String) {
  let backend = detect_backend()
  let topo = get_cpu_topology()

  // Use all physical cores (P + E on hybrid)
  let optimal_threads = topo.physical_cores

  case configure_threads(optimal_threads) {
    Ok(_) -> {
      // Set scatter affinity for hybrid CPUs
      case set_affinity("scatter") {
        Ok(_) -> Ok(backend)
        Error(e) -> Error(e)
      }
    }
    Error(e) -> Error(e)
  }
}

/// Get backend name as string
pub fn backend_name(backend: BlasBackend) -> String {
  case backend {
    IntelMKL -> "Intel MKL (800+ GFLOPS)"
    OpenBLAS -> "OpenBLAS DYNAMIC_ARCH (500+ GFLOPS)"
    ZigSIMD -> "Zig SIMD GEMM (400+ GFLOPS)"
    Unknown -> "Unknown"
  }
}

/// Performance tier for the backend
pub fn expected_gflops(backend: BlasBackend, matrix_size: Int) -> Int {
  case backend {
    IntelMKL ->
      case matrix_size {
        s if s >= 4000 -> 800
        s if s >= 2000 -> 600
        s if s >= 1000 -> 450
        _ -> 300
      }
    OpenBLAS ->
      case matrix_size {
        s if s >= 4000 -> 500
        s if s >= 2000 -> 400
        s if s >= 1000 -> 300
        _ -> 200
      }
    ZigSIMD ->
      case matrix_size {
        s if s >= 4000 -> 450
        s if s >= 2000 -> 350
        s if s >= 1000 -> 250
        _ -> 150
      }
    Unknown -> 100
  }
}
