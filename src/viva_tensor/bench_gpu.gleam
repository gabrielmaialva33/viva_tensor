//// Benchmark Comparativo: viva_tensor (Pure Gleam) vs viva_glands (Rust/Candle GPU)
////
//// Run: gleam run -m viva_tensor/bench_gpu

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor/tensor

@external(erlang, "timer", "tc")
fn timer_tc(f: fn() -> a) -> #(Int, a)

pub fn main() {
  io.println(
    "╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║  BENCHMARK: viva_tensor (Gleam) vs viva_glands (Rust/Candle)    ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  // Check if glands is available
  let glands_available = check_glands()

  case glands_available {
    True -> {
      io.println("✅ viva_glands (Rust/Candle) disponível!\n")
      run_comparative_benchmarks()
    }
    False -> {
      io.println("⚠️  viva_glands não disponível - rodando só Pure Gleam\n")
      run_gleam_only_benchmarks()
    }
  }
}

fn check_glands() -> Bool {
  // Try to call glands_check, if it fails, glands not available
  case safe_glands_check() {
    Ok(_) -> True
    Error(_) -> False
  }
}

fn safe_glands_check() -> Result(String, Nil) {
  // This will fail if NIF not loaded
  Error(Nil)
  // In production: Ok(glands_check())
}

fn run_comparative_benchmarks() {
  io.println("━━━ SIMILARITY BENCHMARK ━━━")
  bench_similarity_comparative()

  io.println("\n━━━ DOT PRODUCT BENCHMARK ━━━")
  bench_dot_comparative()

  io.println("\n━━━ MATMUL BENCHMARK ━━━")
  bench_matmul_comparative()
}

fn run_gleam_only_benchmarks() {
  let sizes = [100, 500, 1000, 2000, 4096, 8192]

  io.println("━━━ PURE GLEAM PERFORMANCE ━━━\n")

  // Dot product scaling
  io.println("DOT PRODUCT (cosine similarity base):")
  list.each(sizes, fn(n) {
    let a = tensor.random_uniform([n])
    let b = tensor.random_uniform([n])

    let #(time_us, _) =
      timer_tc(fn() {
        let _ = tensor.dot(a, b)
        Nil
      })

    let ops_per_sec = case time_us > 0 {
      True -> 1_000_000.0 /. int.to_float(time_us)
      False -> 999_999.0
    }

    io.println(
      "  "
      <> int.to_string(n)
      <> "d: "
      <> int.to_string(time_us)
      <> "μs ("
      <> float_to_string(ops_per_sec)
      <> " ops/s)",
    )
  })

  // Matmul scaling
  io.println("\nMATMUL (matrix sizes):")
  let mat_sizes = [32, 64, 128, 256]
  list.each(mat_sizes, fn(n) {
    let a = tensor.ones([n, n])
    let b = tensor.ones([n, n])

    // Warmup
    let _ = tensor.matmul(a, b)

    // Benchmark (average of 5)
    let times =
      list.range(1, 5)
      |> list.map(fn(_) {
        let #(t, _) =
          timer_tc(fn() {
            let _ = tensor.matmul(a, b)
            Nil
          })
        t
      })

    let avg_time = list.fold(times, 0, fn(acc, t) { acc + t }) / 5

    let flops = 2 * n * n * n
    // 2*n³ for matmul
    let gflops = case avg_time > 0 {
      True -> int.to_float(flops) /. int.to_float(avg_time) /. 1000.0
      // μs to GFLOPS
      False -> 0.0
    }

    io.println(
      "  "
      <> int.to_string(n)
      <> "x"
      <> int.to_string(n)
      <> ": "
      <> int.to_string(avg_time)
      <> "μs ("
      <> float_to_string(gflops)
      <> " GFLOPS)",
    )
  })

  // Reductions scaling
  io.println("\nREDUCTIONS (sum, mean, max):")
  let large_sizes = [1000, 10_000, 100_000, 1_000_000]
  list.each(large_sizes, fn(n) {
    let t = tensor.random_uniform([n])

    let #(sum_time, _) =
      timer_tc(fn() {
        let _ = tensor.sum(t)
        Nil
      })
    let #(mean_time, _) =
      timer_tc(fn() {
        let _ = tensor.mean(t)
        Nil
      })
    let #(max_time, _) =
      timer_tc(fn() {
        let _ = tensor.max(t)
        Nil
      })

    let throughput = case sum_time > 0 {
      True -> int.to_float(n) /. int.to_float(sum_time)
      // elements per μs = M elem/s
      False -> 0.0
    }

    io.println(
      "  "
      <> format_number(n)
      <> ": sum="
      <> int.to_string(sum_time)
      <> "μs, mean="
      <> int.to_string(mean_time)
      <> "μs, max="
      <> int.to_string(max_time)
      <> "μs ("
      <> float_to_string(throughput)
      <> "M elem/s)",
    )
  })

  // Summary
  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║                    PURE GLEAM SUMMARY                            ║",
  )
  io.println(
    "╠══════════════════════════════════════════════════════════════════╣",
  )
  io.println(
    "║  ✅ Zero dependencies - runs anywhere BEAM runs                  ║",
  )
  io.println(
    "║  ✅ Type-safe with Result types                                  ║",
  )
  io.println(
    "║  ✅ Named tensors for semantic clarity                           ║",
  )
  io.println(
    "║  ⚡ For GPU acceleration: use viva_glands backend                ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝",
  )
}

fn bench_similarity_comparative() {
  // This would compare if glands is available
  io.println("(Comparativo requer viva_glands carregado)")
}

fn bench_dot_comparative() {
  io.println("(Comparativo requer viva_glands carregado)")
}

fn bench_matmul_comparative() {
  io.println("(Comparativo requer viva_glands carregado)")
}

fn float_to_string(f: Float) -> String {
  let rounded = int.to_float(float.round(f *. 100.0)) /. 100.0
  float.to_string(rounded)
}

fn format_number(n: Int) -> String {
  case n >= 1_000_000 {
    True -> int.to_string(n / 1_000_000) <> "M"
    False ->
      case n >= 1000 {
        True -> int.to_string(n / 1000) <> "K"
        False -> int.to_string(n)
      }
  }
}
