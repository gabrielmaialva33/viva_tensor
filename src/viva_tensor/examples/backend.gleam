//// Backend Demo - Demonstrating pluggable backends
////
//// Shows how to:
//// 1. Auto-select the best available backend
//// 2. Explicitly choose a specific backend
//// 3. Compare performance across backends
////
//// Execute: gleam run -m examples/backend_demo

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/result
import viva_tensor/backend/protocol as backend
import viva_tensor/core/ffi

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("")
  io.println(
    "╔═══════════════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║              viva_tensor - BACKEND PROTOCOL DEMO                          ║",
  )
  io.println(
    "║          Pluggable computation backends for BEAM                          ║",
  )
  io.println(
    "╚═══════════════════════════════════════════════════════════════════════════╝",
  )
  io.println("")

  // Show available backends
  io.println("━━━ AVAILABLE BACKENDS ━━━")
  io.println("")

  let backends = [
    backend.Pure,
    backend.Accelerate,
    backend.Zig,
    backend.Distributed([]),
  ]

  list.each(backends, fn(b) {
    let status = case backend.is_available(b) {
      True -> "✓ Available"
      False -> "✗ Not available"
    }
    io.println("  " <> backend.name(b) <> ": " <> status)
    case backend.is_available(b) {
      True -> io.println("    └─ " <> backend.info(b))
      False -> Nil
    }
  })

  io.println("")

  // Auto-select best backend
  let best = backend.auto_select()
  io.println("━━━ AUTO-SELECTED BACKEND ━━━")
  io.println("  Best available: " <> backend.name(best))
  io.println("  Info: " <> backend.info(best))
  io.println("")

  // Demo: Matrix multiplication with different backends
  io.println("━━━ MATRIX MULTIPLICATION DEMO ━━━")
  io.println("")

  let m = 64
  let n = 64
  let k = 64
  let size_str =
    int.to_string(m)
    <> "x"
    <> int.to_string(k)
    <> " @ "
    <> int.to_string(k)
    <> "x"
    <> int.to_string(n)

  io.println("  Matrix size: " <> size_str)
  io.println("")

  // Create test data
  let a = random_floats(m * k)
  let b = random_floats(k * n)

  // Test each available backend
  io.println("  ┌────────────────────┬──────────────┬──────────────┐")
  io.println("  │ Backend            │ Time         │ Result       │")
  io.println("  ├────────────────────┼──────────────┼──────────────┤")

  // Pure Erlang
  let #(pure_time, pure_ok) = benchmark_backend(backend.Pure, a, b, m, n, k)
  print_backend_row("Pure Erlang", pure_time, pure_ok)

  // Accelerate (if available)
  case backend.is_available(backend.Accelerate) {
    True -> {
      let #(acc_time, acc_ok) =
        benchmark_backend(backend.Accelerate, a, b, m, n, k)
      print_backend_row("Apple Accelerate", acc_time, acc_ok)
    }
    False -> print_backend_row("Apple Accelerate", 0.0, False)
  }

  // Zig SIMD (if available)
  case backend.is_available(backend.Zig) {
    True -> {
      let #(zig_time, zig_ok) = benchmark_backend(backend.Zig, a, b, m, n, k)
      print_backend_row("Zig SIMD", zig_time, zig_ok)
    }
    False -> print_backend_row("Zig SIMD", 0.0, False)
  }

  io.println("  └────────────────────┴──────────────┴──────────────┘")
  io.println("")

  // Demo: Dot product
  io.println("━━━ DOT PRODUCT DEMO ━━━")
  io.println("")

  let vec_size = 10_000
  let vec_a = random_floats(vec_size)
  let vec_b = random_floats(vec_size)

  io.println("  Vector size: " <> int.to_string(vec_size))
  io.println("")

  // Compute with auto-selected backend
  let dot_start = ffi.now_microseconds()
  let dot_result = backend.dot(best, vec_a, vec_b)
  let dot_end = ffi.now_microseconds()
  let dot_time = int.to_float(dot_end - dot_start) /. 1000.0

  case dot_result {
    Ok(value) -> {
      io.println(
        "  Backend: "
        <> backend.name(best)
        <> " → "
        <> float_to_str(value)
        <> " ("
        <> format_time(dot_time)
        <> ")",
      )
    }
    Error(_) -> io.println("  Error computing dot product")
  }

  io.println("")

  // Demo: Reduction operations
  io.println("━━━ REDUCTION OPERATIONS ━━━")
  io.println("")

  let data = random_floats(100_000)
  io.println("  Data size: 100,000 elements")
  io.println("")

  // Sum
  let sum_start = ffi.now_microseconds()
  let sum_result = backend.sum(best, data)
  let sum_end = ffi.now_microseconds()
  let sum_time = int.to_float(sum_end - sum_start) /. 1000.0

  case sum_result {
    Ok(value) -> {
      io.println(
        "  Sum: " <> float_to_str(value) <> " (" <> format_time(sum_time) <> ")",
      )
    }
    Error(_) -> io.println("  Error computing sum")
  }

  // Scale
  let scale_start = ffi.now_microseconds()
  let scale_result = backend.scale(best, list.take(data, 1000), 2.0)
  let scale_end = ffi.now_microseconds()
  let scale_time = int.to_float(scale_end - scale_start) /. 1000.0

  case scale_result {
    Ok(_) -> {
      io.println("  Scale (×2.0, 1000 elements): " <> format_time(scale_time))
    }
    Error(_) -> io.println("  Error computing scale")
  }

  io.println("")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
  io.println("                         DEMO COMPLETE!")
  io.println(
    "═══════════════════════════════════════════════════════════════════════════",
  )
}

// =============================================================================
// HELPERS
// =============================================================================

fn benchmark_backend(
  b: backend.Backend,
  a: List(Float),
  b_data: List(Float),
  m: Int,
  n: Int,
  k: Int,
) -> #(Float, Bool) {
  let start = ffi.now_microseconds()
  let result = backend.matmul(b, a, b_data, m, n, k)
  let end = ffi.now_microseconds()
  let time = int.to_float(end - start) /. 1000.0

  #(time, result.is_ok(result))
}

fn print_backend_row(name: String, time: Float, ok: Bool) {
  let time_str = case time >. 0.0 {
    True -> format_time(time)
    False -> "N/A"
  }
  let result_str = case ok {
    True -> "✓ OK"
    False -> "✗ Error/N/A"
  }
  io.println(
    "  │ "
    <> pad_right(name, 18)
    <> " │ "
    <> pad_right(time_str, 12)
    <> " │ "
    <> pad_right(result_str, 12)
    <> " │",
  )
}

fn random_floats(n: Int) -> List(Float) {
  list.range(0, n - 1)
  |> list.map(fn(_) { ffi.random_uniform() })
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
