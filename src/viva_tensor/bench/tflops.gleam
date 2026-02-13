//// TFLOPS Benchmark - Measure ALL backends
////
//// Comprehensive TFLOPS measurement across:
//// Pure Erlang, Zig SIMD, CUDA FP32, CUDA FP16, Sparse 2:4
////
//// Execute: gleam run -m bench/tflops

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/core/ffi
import viva_tensor/tflops.{type Backend, PureErlang}

// =============================================================================
// MAIN
// =============================================================================

pub fn main() {
  io.println("")
  io.println("╔═══════════════════════════════════════════════════════════╗")
  io.println("║              viva_tensor - TFLOPS BENCHMARK              ║")
  io.println("╚═══════════════════════════════════════════════════════════╝")
  io.println("")

  // Show backends
  print_backends()
  io.println("")

  // Detect available backends
  let backends = tflops.detect_backends()
  let iterations = 10

  // Matrix sizes to benchmark
  let sizes = [256, 512, 1024, 2048, 4096]

  // Run benchmarks
  io.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
  io.println(
    "  MATMUL BENCHMARK (warmup 2 + "
    <> int.to_string(iterations)
    <> " iterations)",
  )
  io.println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
  io.println("")

  list.each(sizes, fn(size) { benchmark_size(size, backends, iterations) })

  // Summary: peak at largest feasible size per backend
  io.println("═══════════════════════════════════════════════════════════")
  io.println("  Peak TFLOPS (largest matrix per backend):")
  io.println("═══════════════════════════════════════════════════════════")
  io.println("")

  let fast_backends = filter_backends_for_size(backends, 4096)
  let peak_results =
    list.map(fast_backends, fn(backend) {
      tflops.measure_matmul_averaged(backend, 4096, 4096, 4096, iterations)
    })
  list.each(peak_results, fn(r) { io.println("  " <> tflops.format_result(r)) })

  // Pure Erlang peak at 512 (feasible size)
  case list.contains(backends, PureErlang) {
    True -> {
      let erlang_peak =
        tflops.measure_matmul_averaged(PureErlang, 512, 512, 512, iterations)
      io.println("  " <> tflops.format_result(erlang_peak))
    }
    False -> Nil
  }

  io.println("")
  print_theoretical_peaks(backends)
  io.println("")
  io.println("  BENCHMARK COMPLETE!")
  io.println("")
}

// =============================================================================
// BENCHMARK
// =============================================================================

fn benchmark_size(size: Int, backends: List(Backend), iterations: Int) {
  let flops = 2 * size * size * size
  let flops_str = format_flops(flops)

  io.println(
    "  "
    <> int.to_string(size)
    <> "x"
    <> int.to_string(size)
    <> " @ "
    <> int.to_string(size)
    <> "x"
    <> int.to_string(size)
    <> " ("
    <> flops_str
    <> " FLOPs):",
  )

  // Skip very large sizes for slow backends
  let effective_backends = filter_backends_for_size(backends, size)

  let results =
    list.map(effective_backends, fn(backend) {
      tflops.measure_matmul_averaged(backend, size, size, size, iterations)
    })

  io.println(tflops.format_table(results))
  io.println("")
}

fn filter_backends_for_size(backends: List(Backend), size: Int) -> List(Backend) {
  list.filter(backends, fn(b) {
    case b {
      // Pure Erlang is too slow for large matrices
      PureErlang -> size <= 512
      _ -> True
    }
  })
}

// =============================================================================
// DISPLAY
// =============================================================================

fn print_backends() {
  io.println("  BACKENDS:")

  io.println(
    "    " <> status_icon(True) <> " Pure Erlang    (always available)",
  )

  let zig = ffi.zig_is_loaded()
  let zig_info = case zig {
    True -> ffi.zig_backend_info()
    False -> "not compiled"
  }
  io.println(
    "    " <> status_icon(zig) <> " Zig SIMD       (" <> zig_info <> ")",
  )

  let cuda = cuda_available()
  io.println(
    "    "
    <> status_icon(cuda)
    <> " CUDA FP32      "
    <> case cuda {
      True -> "(GPU detected)"
      False -> "(no GPU)"
    },
  )

  let fp16 = ffi.ct16_available()
  io.println(
    "    "
    <> status_icon(fp16)
    <> " CUDA FP16      "
    <> case fp16 {
      True -> "(Tensor Cores)"
      False -> "(unavailable)"
    },
  )

  let sparse = ffi.sparse_available()
  io.println(
    "    "
    <> status_icon(sparse)
    <> " Sparse 2:4     "
    <> case sparse {
      True -> "(cuSPARSELt)"
      False -> "(unavailable)"
    },
  )
}

fn print_theoretical_peaks(backends: List(Backend)) {
  io.println("  Theoretical peaks:")
  list.each(backends, fn(b) {
    let peak = tflops.theoretical_peak(b)
    io.println(
      "    "
      <> pad_right(tflops.backend_name(b), 16)
      <> format_peak(peak)
      <> " TFLOPS",
    )
  })
}

fn status_icon(available: Bool) -> String {
  case available {
    True -> "+"
    False -> "-"
  }
}

fn format_flops(flops: Int) -> String {
  case flops >= 1_000_000_000 {
    True -> {
      let gf = int.to_float(flops) /. 1_000_000_000.0
      float_to_str2(gf) <> "G"
    }
    False ->
      case flops >= 1_000_000 {
        True -> {
          let mf = int.to_float(flops) /. 1_000_000.0
          float_to_str2(mf) <> "M"
        }
        False -> int.to_string(flops)
      }
  }
}

fn format_peak(peak: Float) -> String {
  case peak <. 0.01 {
    True -> "<0.01"
    False -> float_to_str2(peak)
  }
}

fn cuda_available() -> Bool {
  case ffi.ct_from_list([1.0, 0.0, 0.0, 1.0], [2, 2]) {
    Ok(_) -> True
    Error(_) -> False
  }
}

// =============================================================================
// HELPERS
// =============================================================================

fn float_to_str2(f: Float) -> String {
  let rounded = float.round(f *. 100.0)
  let whole = rounded / 100
  let frac = abs_int(rounded - whole * 100)
  int.to_string(whole) <> "." <> pad_left_zero(int.to_string(frac), 2)
}

fn abs_int(n: Int) -> Int {
  case n < 0 {
    True -> 0 - n
    False -> n
  }
}

fn pad_left_zero(s: String, width: Int) -> String {
  let len = string_length(s)
  case len < width {
    True -> repeat_string("0", width - len) <> s
    False -> s
  }
}

fn pad_right(s: String, width: Int) -> String {
  let len = string_length(s)
  case len < width {
    True -> s <> repeat_string(" ", width - len)
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
