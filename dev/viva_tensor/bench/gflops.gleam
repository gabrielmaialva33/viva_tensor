//// GFLOPS Benchmark - Performance measurement in GFLOPS
////
//// Measures actual computational throughput for tensor operations.
//// Also validates numerical correctness across all backends.
////
//// Execute: gleam run -m examples/gflops_benchmark

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import viva_tensor/core/ffi
import viva_tensor/core/ops
import viva_tensor/core/tensor

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║              viva_tensor - GFLOPS BENCHMARK                               ║",
  )
  io.println(
    "║          Performance measurement & numerical correctness                  ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Show backends
  io.println("━━━ BACKEND STATUS ━━━")
  io.println(ops.all_backends_info())
  io.println("")

  // Numerical correctness tests
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("                     NUMERICAL CORRECTNESS")
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("")

  test_matmul_correctness()
  test_dot_correctness()
  test_sum_correctness()

  // GFLOPS benchmarks
  io.println("")
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("                        GFLOPS MEASUREMENT")
  io.println(
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
  )
  io.println("")
  io.println("  Matrix multiplication: 2*M*N*K FLOPs")
  io.println("")

  benchmark_gflops(128, 128, 128)
  benchmark_gflops(256, 256, 256)
  benchmark_gflops(512, 512, 512)

  io.println("")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
  io.println("                       BENCHMARK COMPLETE!")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
}

// =============================================================================
// NUMERICAL CORRECTNESS TESTS
// =============================================================================

fn test_matmul_correctness() {
  io.println("  Matrix Multiplication Correctness:")

  // Simple 2x2 test case with known answer
  // [1 2] @ [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
  // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
  let a_data = [1.0, 2.0, 3.0, 4.0]
  let b_data = [5.0, 6.0, 7.0, 8.0]
  let expected = [19.0, 22.0, 43.0, 50.0]

  // Test Pure Erlang
  let a = result.unwrap(tensor.new(a_data, [2, 2]), tensor.zeros([2, 2]))
  let b = result.unwrap(tensor.new(b_data, [2, 2]), tensor.zeros([2, 2]))

  let pure_result = ops.matmul_fast(a, b)
  let pure_ok = case pure_result {
    Ok(t) -> check_close(tensor.to_list(t), expected, 1.0e-10)
    Error(_) -> False
  }
  io.println("    Pure Erlang:      " <> result_to_string(pure_ok))

  // Test Apple Accelerate
  case ffi.is_nif_loaded() {
    True -> {
      case ffi.nif_matmul(a_data, b_data, 2, 2, 2) {
        Ok(result) -> {
          let ok = check_close(result, expected, 1.0e-10)
          io.println("    Apple Accelerate: " <> result_to_string(ok))
        }
        Error(_) -> io.println("    Apple Accelerate: ✗ Error")
      }
    }
    False -> io.println("    Apple Accelerate: - Not available")
  }

  // Test Zig SIMD
  case ffi.zig_is_loaded() {
    True -> {
      case ffi.zig_matmul(a_data, b_data, 2, 2, 2) {
        Ok(result) -> {
          let ok = check_close(result, expected, 1.0e-10)
          io.println("    Zig SIMD:         " <> result_to_string(ok))
        }
        Error(_) -> io.println("    Zig SIMD:         ✗ Error")
      }
    }
    False -> io.println("    Zig SIMD:         - Not available")
  }

  io.println("")
}

fn test_dot_correctness() {
  io.println("  Dot Product Correctness:")

  // [1, 2, 3, 4] · [5, 6, 7, 8] = 5 + 12 + 21 + 32 = 70
  let a_data = [1.0, 2.0, 3.0, 4.0]
  let b_data = [5.0, 6.0, 7.0, 8.0]
  let expected = 70.0

  // Test Pure Erlang
  let a_arr = ffi.list_to_array(a_data)
  let b_arr = ffi.list_to_array(b_data)
  let pure_result = ffi.array_dot(a_arr, b_arr)
  let pure_ok = float_close(pure_result, expected, 1.0e-10)
  io.println("    Pure Erlang:      " <> result_to_string(pure_ok))

  // Test Apple Accelerate
  case ffi.is_nif_loaded() {
    True -> {
      case ffi.nif_dot(a_data, b_data) {
        Ok(result) -> {
          let ok = float_close(result, expected, 1.0e-10)
          io.println("    Apple Accelerate: " <> result_to_string(ok))
        }
        Error(_) -> io.println("    Apple Accelerate: ✗ Error")
      }
    }
    False -> io.println("    Apple Accelerate: - Not available")
  }

  // Test Zig SIMD
  case ffi.zig_is_loaded() {
    True -> {
      case ffi.zig_dot(a_data, b_data) {
        Ok(result) -> {
          let ok = float_close(result, expected, 1.0e-10)
          io.println("    Zig SIMD:         " <> result_to_string(ok))
        }
        Error(_) -> io.println("    Zig SIMD:         ✗ Error")
      }
    }
    False -> io.println("    Zig SIMD:         - Not available")
  }

  io.println("")
}

fn test_sum_correctness() {
  io.println("  Sum Reduction Correctness:")

  // sum([1, 2, 3, 4, 5]) = 15
  let data = [1.0, 2.0, 3.0, 4.0, 5.0]
  let expected = 15.0

  // Test Pure Erlang
  let arr = ffi.list_to_array(data)
  let pure_result = ffi.array_sum(arr)
  let pure_ok = float_close(pure_result, expected, 1.0e-10)
  io.println("    Pure Erlang:      " <> result_to_string(pure_ok))

  // Test Apple Accelerate
  case ffi.is_nif_loaded() {
    True -> {
      case ffi.nif_sum(data) {
        Ok(result) -> {
          let ok = float_close(result, expected, 1.0e-10)
          io.println("    Apple Accelerate: " <> result_to_string(ok))
        }
        Error(_) -> io.println("    Apple Accelerate: ✗ Error")
      }
    }
    False -> io.println("    Apple Accelerate: - Not available")
  }

  // Test Zig SIMD
  case ffi.zig_is_loaded() {
    True -> {
      case ffi.zig_sum(data) {
        Ok(result) -> {
          let ok = float_close(result, expected, 1.0e-10)
          io.println("    Zig SIMD:         " <> result_to_string(ok))
        }
        Error(_) -> io.println("    Zig SIMD:         ✗ Error")
      }
    }
    False -> io.println("    Zig SIMD:         - Not available")
  }
}

// =============================================================================
// GFLOPS BENCHMARK
// =============================================================================

fn benchmark_gflops(m: Int, n: Int, k: Int) {
  let size_str =
    int.to_string(m)
    <> "x"
    <> int.to_string(k)
    <> " @ "
    <> int.to_string(k)
    <> "x"
    <> int.to_string(n)

  // FLOPs for matmul: 2*M*N*K (multiply and add for each element)
  let flops = 2 * m * n * k

  io.println("  " <> size_str <> " (" <> int.to_string(flops) <> " FLOPs):")
  io.println("  ┌────────────────────┬──────────────┬──────────────┐")
  io.println("  │ Backend            │ Time (ms)    │ GFLOPS       │")
  io.println("  ├────────────────────┼──────────────┼──────────────┤")

  let a_data = random_floats(m * k)
  let b_data = random_floats(k * n)

  // Pure Erlang
  let a_arr = ffi.list_to_array(a_data)
  let b_arr = ffi.list_to_array(b_data)
  let #(pure_time, _) =
    time_it(fn() { ffi.array_matmul(a_arr, b_arr, m, n, k) })
  let pure_gflops = compute_gflops(flops, pure_time)
  print_gflops_row("Pure Erlang", pure_time, pure_gflops)

  // Apple Accelerate
  case ffi.is_nif_loaded() {
    True -> {
      let #(accel_time, _) =
        time_it(fn() { ffi.nif_matmul(a_data, b_data, m, n, k) })
      let accel_gflops = compute_gflops(flops, accel_time)
      print_gflops_row("Apple Accelerate", accel_time, accel_gflops)
    }
    False -> print_gflops_row("Apple Accelerate", 0.0, 0.0)
  }

  // Zig SIMD
  case ffi.zig_is_loaded() {
    True -> {
      let #(zig_time, _) =
        time_it(fn() { ffi.zig_matmul(a_data, b_data, m, n, k) })
      let zig_gflops = compute_gflops(flops, zig_time)
      print_gflops_row("Zig SIMD", zig_time, zig_gflops)
    }
    False -> print_gflops_row("Zig SIMD", 0.0, 0.0)
  }

  io.println("  └────────────────────┴──────────────┴──────────────┘")
  io.println("")
}

fn compute_gflops(flops: Int, time_ms: Float) -> Float {
  case time_ms >. 0.0 {
    True -> int.to_float(flops) /. { time_ms *. 1_000_000.0 }
    False -> 0.0
  }
}

fn print_gflops_row(name: String, time_ms: Float, gflops: Float) {
  let time_str = case time_ms >. 0.0 {
    True -> float_to_str(time_ms)
    False -> "N/A"
  }
  let gflops_str = case gflops >. 0.0 {
    True -> float_to_str(gflops)
    False -> "N/A"
  }
  io.println(
    "  │ "
    <> pad_right(name, 18)
    <> " │ "
    <> pad_right(time_str, 12)
    <> " │ "
    <> pad_right(gflops_str, 12)
    <> " │",
  )
}

// =============================================================================
// HELPERS
// =============================================================================

fn time_it(f: fn() -> a) -> #(Float, a) {
  let start = ffi.now_microseconds()
  let result = f()
  let end = ffi.now_microseconds()
  let time_ms = int.to_float(end - start) /. 1000.0
  #(time_ms, result)
}

fn random_floats(n: Int) -> List(Float) {
  list.range(0, n - 1)
  |> list.map(fn(_) { ffi.random_uniform() })
}

fn check_close(actual: List(Float), expected: List(Float), tol: Float) -> Bool {
  case list.length(actual) == list.length(expected) {
    True ->
      list.map2(actual, expected, fn(a, e) { float_close(a, e, tol) })
      |> list.all(fn(x) { x })
    False -> False
  }
}

fn float_close(a: Float, b: Float, tol: Float) -> Bool {
  let diff = case a -. b <. 0.0 {
    True -> b -. a
    False -> a -. b
  }
  diff <. tol
}

fn result_to_string(ok: Bool) -> String {
  case ok {
    True -> "✓ PASS"
    False -> "✗ FAIL"
  }
}

fn float_to_str(f: Float) -> String {
  let rounded = float.round(f *. 100.0) |> int.to_float
  let result = rounded /. 100.0
  float.to_string(result)
}

fn pad_right(s: String, width: Int) -> String {
  let len = string_length(s)
  let padding = width - len
  case padding > 0 {
    True -> s <> repeat_string(" ", padding)
    False -> s
  }
}

fn repeat_string(s: String, n: Int) -> String {
  case n <= 0 {
    True -> ""
    False -> s <> repeat_string(s, n - 1)
  }
}

@external(erlang, "string", "length")
fn string_length(s: String) -> Int
