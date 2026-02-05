//// NIF Benchmark - Three-Way Comparison
////
//// Compares performance of:
//// 1. Pure Erlang (baseline)
//// 2. Apple Accelerate NIF (C, cblas_dgemm)
//// 3. Zig SIMD NIF (cross-platform)
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
  io.println(
    "╔═══════════════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║           viva_tensor - THREE-WAY NIF BENCHMARK                           ║",
  )
  io.println(
    "║       Pure Erlang vs Apple Accelerate vs Zig SIMD                         ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Check backend status
  let c_nif_loaded = ffi.is_nif_loaded()
  let zig_nif_loaded = ffi.zig_is_loaded()
  let c_backend = ffi.nif_backend_info()
  let zig_backend = ffi.zig_backend_info()

  io.println("━━━ BACKEND STATUS ━━━")
  io.println("  Apple Accelerate NIF: " <> bool_to_string(c_nif_loaded))
  io.println("    Backend: " <> c_backend)
  io.println("  Zig SIMD NIF: " <> bool_to_string(zig_nif_loaded))
  io.println("    Backend: " <> zig_backend)
  io.println("")

  // Matmul benchmarks
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("                        MATRIX MULTIPLICATION")
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("")

  // Warmup
  benchmark_matmul(32, 32, 32)

  // Real tests
  benchmark_matmul(64, 64, 64)
  benchmark_matmul(128, 128, 128)
  benchmark_matmul(256, 256, 256)

  // Dot product benchmarks
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("                          DOT PRODUCT")
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("")

  benchmark_dot(10_000)
  benchmark_dot(100_000)

  // Sum benchmarks
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("                              SUM")
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("")

  benchmark_sum(100_000)
  benchmark_sum(1_000_000)

  io.println("")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
  io.println("                         BENCHMARK COMPLETE!")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
}

// ============================================================================
// MATMUL BENCHMARK
// ============================================================================

fn benchmark_matmul(m: Int, n: Int, k: Int) {
  let size_str =
    int.to_string(m)
    <> "x"
    <> int.to_string(k)
    <> " @ "
    <> int.to_string(k)
    <> "x"
    <> int.to_string(n)
  io.println("  Matrix: " <> size_str)
  io.println("  ┌────────────────────┬──────────────┬──────────────┐")
  io.println("  │ Backend            │ Time         │ Speedup      │")
  io.println("  ├────────────────────┼──────────────┼──────────────┤")

  // Create test data
  let a_data = random_floats(m * k)
  let b_data = random_floats(k * n)

  // 1. Pure Erlang (baseline)
  let #(_pure_result, pure_time) = time_pure_matmul(a_data, b_data, m, n, k)
  io.println(
    "  │ Pure Erlang        │ "
    <> pad_time(pure_time)
    <> " │ 1.00x        │",
  )

  // 2. Apple Accelerate (C NIF)
  let #(_c_result, c_time) = time_c_matmul(a_data, b_data, m, n, k)
  let c_speedup = case c_time >. 0.0 {
    True -> pure_time /. c_time
    False -> 0.0
  }
  io.println(
    "  │ Apple Accelerate   │ "
    <> pad_time(c_time)
    <> " │ "
    <> pad_speedup(c_speedup)
    <> " │",
  )

  // 3. Zig SIMD
  let #(_zig_result, zig_time) = time_zig_matmul(a_data, b_data, m, n, k)
  let zig_speedup = case zig_time >. 0.0 {
    True -> pure_time /. zig_time
    False -> 0.0
  }
  io.println(
    "  │ Zig SIMD           │ "
    <> pad_time(zig_time)
    <> " │ "
    <> pad_speedup(zig_speedup)
    <> " │",
  )

  io.println("  └────────────────────┴──────────────┴──────────────┘")
  io.println("")
}

fn time_pure_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> #(List(Float), Float) {
  let start = now_microseconds()
  let a_arr = ffi.list_to_array(a)
  let b_arr = ffi.list_to_array(b)
  let result_arr = ffi.array_matmul(a_arr, b_arr, m, n, k)
  let result = ffi.array_to_list(result_arr)
  let end = now_microseconds()
  #(result, int.to_float(end - start) /. 1000.0)
}

fn time_c_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> #(List(Float), Float) {
  let start = now_microseconds()
  let result = ffi.nif_matmul(a, b, m, n, k)
  let end = now_microseconds()
  case result {
    Ok(data) -> #(data, int.to_float(end - start) /. 1000.0)
    Error(_) -> #([], int.to_float(end - start) /. 1000.0)
  }
}

fn time_zig_matmul(
  a: List(Float),
  b: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> #(List(Float), Float) {
  let start = now_microseconds()
  let result = ffi.zig_matmul(a, b, m, n, k)
  let end = now_microseconds()
  case result {
    Ok(data) -> #(data, int.to_float(end - start) /. 1000.0)
    Error(_) -> #([], int.to_float(end - start) /. 1000.0)
  }
}

// ============================================================================
// DOT PRODUCT BENCHMARK
// ============================================================================

fn benchmark_dot(size: Int) {
  io.println("  Vector size: " <> int.to_string(size))
  io.println("  ┌────────────────────┬──────────────┬──────────────┐")
  io.println("  │ Backend            │ Time         │ Speedup      │")
  io.println("  ├────────────────────┼──────────────┼──────────────┤")

  let a_data = random_floats(size)
  let b_data = random_floats(size)

  // Pure Erlang
  let start_pure = now_microseconds()
  let a_arr = ffi.list_to_array(a_data)
  let b_arr = ffi.list_to_array(b_data)
  let _pure_result = ffi.array_dot(a_arr, b_arr)
  let end_pure = now_microseconds()
  let pure_time = int.to_float(end_pure - start_pure) /. 1000.0
  io.println(
    "  │ Pure Erlang        │ "
    <> pad_time(pure_time)
    <> " │ 1.00x        │",
  )

  // Apple Accelerate
  let start_c = now_microseconds()
  let _c_result = ffi.nif_dot(a_data, b_data)
  let end_c = now_microseconds()
  let c_time = int.to_float(end_c - start_c) /. 1000.0
  let c_speedup = case c_time >. 0.0 {
    True -> pure_time /. c_time
    False -> 0.0
  }
  io.println(
    "  │ Apple Accelerate   │ "
    <> pad_time(c_time)
    <> " │ "
    <> pad_speedup(c_speedup)
    <> " │",
  )

  // Zig SIMD
  let start_zig = now_microseconds()
  let _zig_result = ffi.zig_dot(a_data, b_data)
  let end_zig = now_microseconds()
  let zig_time = int.to_float(end_zig - start_zig) /. 1000.0
  let zig_speedup = case zig_time >. 0.0 {
    True -> pure_time /. zig_time
    False -> 0.0
  }
  io.println(
    "  │ Zig SIMD           │ "
    <> pad_time(zig_time)
    <> " │ "
    <> pad_speedup(zig_speedup)
    <> " │",
  )

  io.println("  └────────────────────┴──────────────┴──────────────┘")
  io.println("")
}

// ============================================================================
// SUM BENCHMARK
// ============================================================================

fn benchmark_sum(size: Int) {
  io.println("  Vector size: " <> int.to_string(size))
  io.println("  ┌────────────────────┬──────────────┬──────────────┐")
  io.println("  │ Backend            │ Time         │ Speedup      │")
  io.println("  ├────────────────────┼──────────────┼──────────────┤")

  let data = random_floats(size)

  // Pure Erlang
  let start_pure = now_microseconds()
  let arr = ffi.list_to_array(data)
  let _pure_result = ffi.array_sum(arr)
  let end_pure = now_microseconds()
  let pure_time = int.to_float(end_pure - start_pure) /. 1000.0
  io.println(
    "  │ Pure Erlang        │ "
    <> pad_time(pure_time)
    <> " │ 1.00x        │",
  )

  // Apple Accelerate
  let start_c = now_microseconds()
  let _c_result = ffi.nif_sum(data)
  let end_c = now_microseconds()
  let c_time = int.to_float(end_c - start_c) /. 1000.0
  let c_speedup = case c_time >. 0.0 {
    True -> pure_time /. c_time
    False -> 0.0
  }
  io.println(
    "  │ Apple Accelerate   │ "
    <> pad_time(c_time)
    <> " │ "
    <> pad_speedup(c_speedup)
    <> " │",
  )

  // Zig SIMD
  let start_zig = now_microseconds()
  let _zig_result = ffi.zig_sum(data)
  let end_zig = now_microseconds()
  let zig_time = int.to_float(end_zig - start_zig) /. 1000.0
  let zig_speedup = case zig_time >. 0.0 {
    True -> pure_time /. zig_time
    False -> 0.0
  }
  io.println(
    "  │ Zig SIMD           │ "
    <> pad_time(zig_time)
    <> " │ "
    <> pad_speedup(zig_speedup)
    <> " │",
  )

  io.println("  └────────────────────┴──────────────┴──────────────┘")
  io.println("")
}

// ============================================================================
// HELPERS
// ============================================================================

fn random_floats(n: Int) -> List(Float) {
  list.range(0, n - 1)
  |> list.map(fn(_) { ffi.random_uniform() })
}

fn pad_time(ms: Float) -> String {
  let str = format_time(ms)
  let len = string_length(str)
  let padding = 12 - len
  str <> repeat_char(" ", padding)
}

fn pad_speedup(speedup: Float) -> String {
  let str = float_to_str(speedup) <> "x"
  let len = string_length(str)
  let padding = 12 - len
  str <> repeat_char(" ", padding)
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
    True -> "Yes ✓"
    False -> "No ✗"
  }
}

fn string_length(s: String) -> Int {
  string_length_ffi(s)
}

@external(erlang, "string", "length")
fn string_length_ffi(s: String) -> Int

fn repeat_char(char: String, n: Int) -> String {
  case n <= 0 {
    True -> ""
    False -> char <> repeat_char(char, n - 1)
  }
}

fn now_microseconds() -> Int {
  ffi.now_microseconds()
}
