//// NIF Benchmark - Apple Accelerate vs Pure Gleam
////
//// Demonstra a diferença de performance entre:
//// - NIF com Apple Accelerate (cblas_dgemm, vDSP)
//// - Pure Gleam/Erlang implementation
////
//// Execute: gleam run -m examples/nif_benchmark

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/core/ffi

// ============================================================================
// MAIN
// ============================================================================

pub fn main() {
  io.println("")
  io.println("╔═══════════════════════════════════════════════════════════════╗")
  io.println("║       viva_tensor - NIF BENCHMARK (Apple Accelerate)          ║")
  io.println("║                  NIF vs Pure Gleam Performance                ║")
  io.println("╚═══════════════════════════════════════════════════════════════╝")
  io.println("")

  // Check NIF status
  let nif_loaded = ffi.is_nif_loaded()
  let backend = ffi.nif_backend_info()

  io.println("━━━ BACKEND INFO ━━━")
  io.println("  NIF Loaded: " <> bool_to_string(nif_loaded))
  io.println("  Backend: " <> backend)
  io.println("")

  // Run benchmarks with different sizes
  io.println("━━━ MATMUL BENCHMARK ━━━")
  io.println("")

  // Small matrix (warm-up)
  benchmark_matmul(32, 32, 32)

  // Medium matrices
  benchmark_matmul(64, 64, 64)
  benchmark_matmul(128, 128, 128)

  // Large matrices (where NIF really shines)
  benchmark_matmul(256, 256, 256)
  benchmark_matmul(512, 512, 512)

  io.println("")
  io.println("━━━ DOT PRODUCT BENCHMARK ━━━")
  io.println("")

  benchmark_dot(1000)
  benchmark_dot(10_000)
  benchmark_dot(100_000)

  io.println("")
  io.println("━━━ SUM BENCHMARK ━━━")
  io.println("")

  benchmark_sum(10_000)
  benchmark_sum(100_000)
  benchmark_sum(1_000_000)

  io.println("")
  io.println("═══════════════════════════════════════════════════════════════")
  io.println("                    BENCHMARK COMPLETE!")
  io.println("═══════════════════════════════════════════════════════════════")
}

// ============================================================================
// MATMUL BENCHMARK
// ============================================================================

fn benchmark_matmul(m: Int, n: Int, k: Int) {
  let size_str = int.to_string(m) <> "x" <> int.to_string(k) <> " @ " <> int.to_string(k) <> "x" <> int.to_string(n)
  io.println("  Matrix: " <> size_str)

  // Create test data
  let a_data = random_floats(m * k)
  let b_data = random_floats(k * n)

  // Benchmark NIF
  let #(nif_result, nif_time) = time_nif_matmul(a_data, b_data, m, n, k)

  // Benchmark Pure Erlang
  let #(pure_result, pure_time) = time_pure_matmul(a_data, b_data, m, n, k)

  // Calculate speedup
  let speedup = case nif_time >. 0.0 {
    True -> pure_time /. nif_time
    False -> 0.0
  }

  // Verify results match (first few elements)
  let results_match = verify_results(nif_result, pure_result)

  io.println("    NIF Time:    " <> format_time(nif_time))
  io.println("    Pure Time:   " <> format_time(pure_time))
  io.println("    Speedup:     " <> float_to_str(speedup) <> "x")
  io.println("    Results OK:  " <> bool_to_string(results_match))
  io.println("")
}

fn time_nif_matmul(a: List(Float), b: List(Float), m: Int, n: Int, k: Int) -> #(List(Float), Float) {
  let start = now_microseconds()
  let result = ffi.nif_matmul(a, b, m, n, k)
  let end = now_microseconds()
  let time_ms = int.to_float(end - start) /. 1000.0

  case result {
    Ok(data) -> #(data, time_ms)
    Error(_) -> #([], time_ms)
  }
}

fn time_pure_matmul(a: List(Float), b: List(Float), m: Int, n: Int, k: Int) -> #(List(Float), Float) {
  let start = now_microseconds()

  // Pure Erlang implementation using arrays
  let a_arr = ffi.list_to_array(a)
  let b_arr = ffi.list_to_array(b)
  let result_arr = ffi.array_matmul(a_arr, b_arr, m, n, k)
  let result = ffi.array_to_list(result_arr)

  let end = now_microseconds()
  let time_ms = int.to_float(end - start) /. 1000.0

  #(result, time_ms)
}

// ============================================================================
// DOT PRODUCT BENCHMARK
// ============================================================================

fn benchmark_dot(size: Int) {
  io.println("  Vector size: " <> int.to_string(size))

  let a_data = random_floats(size)
  let b_data = random_floats(size)

  // NIF
  let start_nif = now_microseconds()
  let _nif_result = ffi.nif_dot(a_data, b_data)
  let end_nif = now_microseconds()
  let nif_time = int.to_float(end_nif - start_nif) /. 1000.0

  // Pure Erlang
  let start_pure = now_microseconds()
  let a_arr = ffi.list_to_array(a_data)
  let b_arr = ffi.list_to_array(b_data)
  let _pure_result = ffi.array_dot(a_arr, b_arr)
  let end_pure = now_microseconds()
  let pure_time = int.to_float(end_pure - start_pure) /. 1000.0

  let speedup = case nif_time >. 0.0 {
    True -> pure_time /. nif_time
    False -> 0.0
  }

  io.println("    NIF Time:    " <> format_time(nif_time))
  io.println("    Pure Time:   " <> format_time(pure_time))
  io.println("    Speedup:     " <> float_to_str(speedup) <> "x")
  io.println("")
}

// ============================================================================
// SUM BENCHMARK
// ============================================================================

fn benchmark_sum(size: Int) {
  io.println("  Vector size: " <> int.to_string(size))

  let data = random_floats(size)

  // NIF
  let start_nif = now_microseconds()
  let _nif_result = ffi.nif_sum(data)
  let end_nif = now_microseconds()
  let nif_time = int.to_float(end_nif - start_nif) /. 1000.0

  // Pure Erlang
  let start_pure = now_microseconds()
  let arr = ffi.list_to_array(data)
  let _pure_result = ffi.array_sum(arr)
  let end_pure = now_microseconds()
  let pure_time = int.to_float(end_pure - start_pure) /. 1000.0

  let speedup = case nif_time >. 0.0 {
    True -> pure_time /. nif_time
    False -> 0.0
  }

  io.println("    NIF Time:    " <> format_time(nif_time))
  io.println("    Pure Time:   " <> format_time(pure_time))
  io.println("    Speedup:     " <> float_to_str(speedup) <> "x")
  io.println("")
}

// ============================================================================
// HELPERS
// ============================================================================

/// Generate random floats
fn random_floats(n: Int) -> List(Float) {
  list.range(0, n - 1)
  |> list.map(fn(_) { ffi.random_uniform() })
}

/// Verify results match within tolerance
fn verify_results(a: List(Float), b: List(Float)) -> Bool {
  case a, b {
    [], [] -> True
    [x, ..xs], [y, ..ys] -> {
      let diff = float_abs(x -. y)
      case diff <. 0.0001 {
        True -> verify_results(xs, ys)
        False -> False
      }
    }
    _, _ -> False
  }
}

fn float_abs(x: Float) -> Float {
  case x <. 0.0 {
    True -> 0.0 -. x
    False -> x
  }
}

fn format_time(ms: Float) -> String {
  case ms <. 1.0 {
    True -> float_to_str(ms *. 1000.0) <> " μs"
    False -> {
      case ms <. 1000.0 {
        True -> float_to_str(ms) <> " ms"
        False -> float_to_str(ms /. 1000.0) <> " s"
      }
    }
  }
}

fn float_to_str(f: Float) -> String {
  let rounded = float.round(f *. 100.0) |> int.to_float
  let result = rounded /. 100.0
  float.to_string(result)
}

fn bool_to_string(b: Bool) -> String {
  case b {
    True -> "Yes"
    False -> "No"
  }
}

/// Get current time in microseconds (via FFI)
fn now_microseconds() -> Int {
  ffi.now_microseconds()
}
