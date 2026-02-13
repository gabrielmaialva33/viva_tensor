import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import viva_tensor/core/ffi

// =============================================================================
// VIVA TENSOR BENCHMARK SUITE
// Run on both Windows (native) and WSL to compare performance
// =============================================================================

pub fn main() {
  io.println("=" |> string.repeat(60))
  io.println("  VIVA TENSOR BENCHMARK SUITE - Mad Scientist Edition")
  io.println("=" |> string.repeat(60))
  io.println("")

  // Check backend
  let backend = ffi.zig_backend_info()
  io.println("Backend: " <> backend)
  io.println("")

  // ============================================================
  // Part 1: Matrix Multiplication (Goto GEMM)
  // ============================================================
  io.println(">>> MATMUL BENCHMARKS (Goto GEMM, f64)")
  io.println("-" |> string.repeat(60))

  // Sizes to benchmark
  let sizes = [100, 200, 500, 1000, 1500, 2000]

  list.each(sizes, fn(n) {
    let iterations = case n {
      100 -> 100
      200 -> 50
      500 -> 20
      1000 -> 5
      1500 -> 3
      _ -> 2
    }
    bench_matmul(n, iterations)
  })

  io.println("")

  // ============================================================
  // Part 2: LNS (Log-Number System) - IADD mul
  // ============================================================
  io.println(">>> LNS BENCHMARKS (IADD mul, f32)")
  io.println("-" |> string.repeat(60))

  bench_lns(10_000, 1000)
  bench_lns(100_000, 100)
  bench_lns(1_000_000, 10)

  io.println("")

  // ============================================================
  // Part 3: Horde (SoA Physics)
  // ============================================================
  io.println(">>> HORDE BENCHMARKS (SoA Physics, 60fps target)")
  io.println("-" |> string.repeat(60))

  bench_horde(1000, 1000)
  bench_horde(10_000, 100)
  bench_horde(100_000, 10)

  io.println("")

  // ============================================================
  // Part 4: HDC (Hyperdimensional Computing)
  // ============================================================
  io.println(">>> HDC BENCHMARKS (10K-bit vectors)")
  io.println("-" |> string.repeat(60))

  bench_hdc(10_048, 10_000)

  io.println("")
  io.println("=" |> string.repeat(60))
  io.println("  BENCHMARK COMPLETE")
  io.println("=" |> string.repeat(60))
}

fn bench_matmul(n: Int, iterations: Int) {
  // Create tensors
  let shape = [n, n]
  let assert Ok(a) = ffi.nt_ones(shape)
  let assert Ok(b) = ffi.nt_ones(shape)

  // Warmup
  let assert Ok(_) = ffi.nt_matmul(a, b, n, n, n)
  let assert Ok(_) = ffi.nt_matmul(a, b, n, n, n)
  let assert Ok(_) = ffi.nt_matmul(a, b, n, n, n)

  // Benchmark
  let start = ffi.now_microseconds()

  list.range(1, iterations)
  |> list.each(fn(_) {
    let assert Ok(_) = ffi.nt_matmul(a, b, n, n, n)
    Nil
  })

  let end = ffi.now_microseconds()
  let avg_us = int.to_float(end - start) /. int.to_float(iterations)

  // GFLOPS: 2 * M * N * K / time_seconds / 1e9
  let flops = 2.0 *. int.to_float(n * n * n)
  let gflops = flops /. avg_us /. 1000.0

  let size_str = int.to_string(n) <> "x" <> int.to_string(n)
  let time_str = float_to_string_2(avg_us /. 1000.0) <> " ms"
  let gflops_str = float_to_string_1(gflops) <> " GFLOPS"

  io.println("  " <> pad_right(size_str, 12) <> time_str <> "  " <> gflops_str)
}

fn bench_lns(size: Int, iterations: Int) {
  // Create f64 tensor, convert to LNS
  let assert Ok(t1) = ffi.nt_fill([size], 2.0)
  let assert Ok(t2) = ffi.nt_fill([size], 3.0)
  let assert Ok(lns_a) = ffi.lns_from_f64(t1)
  let assert Ok(lns_b) = ffi.lns_from_f64(t2)

  // Warmup
  let assert Ok(_) = ffi.lns_mul(lns_a, lns_b)

  // Benchmark LNS mul
  let start = ffi.now_microseconds()

  list.range(1, iterations)
  |> list.each(fn(_) {
    let assert Ok(_) = ffi.lns_mul(lns_a, lns_b)
    Nil
  })

  let end = ffi.now_microseconds()
  let avg_us = int.to_float(end - start) /. int.to_float(iterations)

  // GFLOPS-equivalent (actually GIOP for IADD)
  let ops = int.to_float(size)
  let gops = ops /. avg_us /. 1000.0

  let size_str = format_size(size)
  let time_str = float_to_string_1(avg_us) <> " us"
  let gops_str = float_to_string_1(gops) <> " GIOP (IADD)"

  io.println(
    "  LNS mul " <> pad_right(size_str, 8) <> time_str <> "  " <> gops_str,
  )
}

fn bench_horde(entity_count: Int, iterations: Int) {
  // Create horde with 2D entities
  let assert Ok(horde) = ffi.horde_create(entity_count, 2)

  // Set initial positions and velocities
  let positions =
    list.range(0, entity_count - 1)
    |> list.flat_map(fn(i) { [int.to_float(i), int.to_float(i)] })

  let velocities =
    list.range(0, entity_count - 1)
    |> list.flat_map(fn(_) { [1.0, 0.5] })

  let assert Ok(_) = ffi.horde_set_positions(horde, positions)
  let assert Ok(_) = ffi.horde_set_velocities(horde, velocities)

  // Warmup
  let assert Ok(_) = ffi.horde_integrate(horde, 0.01667)
  let assert Ok(_) = ffi.horde_dampen(horde, 0.99)

  // Benchmark: integrate + dampen (one physics step)
  let start = ffi.now_microseconds()

  list.range(1, iterations)
  |> list.each(fn(_) {
    let assert Ok(_) = ffi.horde_integrate(horde, 0.01667)
    let assert Ok(_) = ffi.horde_dampen(horde, 0.99)
    Nil
  })

  let end = ffi.now_microseconds()
  let avg_us = int.to_float(end - start) /. int.to_float(iterations)

  // FPS equivalent (if we spend entire frame on physics)
  let fps_equiv = 1_000_000.0 /. avg_us

  let size_str = format_size(entity_count)
  let time_str = float_to_string_1(avg_us) <> " us"
  let fps_str = float_to_string_0(fps_equiv) <> " fps"

  io.println(
    "  Horde " <> pad_right(size_str, 10) <> time_str <> "  " <> fps_str,
  )
}

fn bench_hdc(dim: Int, iterations: Int) {
  // Create random hypervectors
  let assert Ok(vec_a) = ffi.hdc_random(dim, 42)
  let assert Ok(vec_b) = ffi.hdc_random(dim, 43)

  // Warmup
  let assert Ok(_) = ffi.hdc_bind(vec_a, vec_b)
  let assert Ok(_) = ffi.hdc_similarity(vec_a, vec_b)

  // Benchmark XOR bind
  let start1 = ffi.now_microseconds()
  list.range(1, iterations)
  |> list.each(fn(_) {
    let assert Ok(_) = ffi.hdc_bind(vec_a, vec_b)
    Nil
  })
  let end1 = ffi.now_microseconds()
  let bind_us = int.to_float(end1 - start1) /. int.to_float(iterations)

  // Benchmark similarity
  let start2 = ffi.now_microseconds()
  list.range(1, iterations)
  |> list.each(fn(_) {
    let assert Ok(_) = ffi.hdc_similarity(vec_a, vec_b)
    Nil
  })
  let end2 = ffi.now_microseconds()
  let sim_us = int.to_float(end2 - start2) /. int.to_float(iterations)

  let bind_ops = 1_000_000.0 /. bind_us
  let sim_ops = 1_000_000.0 /. sim_us

  io.println(
    "  HDC bind:       "
    <> float_to_string_1(bind_us)
    <> " us  "
    <> float_to_string_0(bind_ops)
    <> " ops/sec",
  )
  io.println(
    "  HDC similarity: "
    <> float_to_string_1(sim_us)
    <> " us  "
    <> float_to_string_0(sim_ops)
    <> " ops/sec",
  )
}

// Helper functions

fn format_size(n: Int) -> String {
  case n {
    _ if n >= 1_000_000 -> int.to_string(n / 1_000_000) <> "M"
    _ if n >= 1000 -> int.to_string(n / 1000) <> "K"
    _ -> int.to_string(n)
  }
}

fn pad_right(s: String, width: Int) -> String {
  let len = string.length(s)
  case len < width {
    True -> s <> string.repeat(" ", width - len)
    False -> s
  }
}

fn float_to_string_0(f: Float) -> String {
  int.to_string(float.round(f))
}

fn float_to_string_1(f: Float) -> String {
  let i = float.round(f *. 10.0)
  let whole = i / 10
  let frac = int.absolute_value(i) % 10
  int.to_string(whole) <> "." <> int.to_string(frac)
}

fn float_to_string_2(f: Float) -> String {
  let i = float.round(f *. 100.0)
  let whole = i / 100
  let frac = int.absolute_value(i) % 100
  let frac_str = case frac < 10 {
    True -> "0" <> int.to_string(frac)
    False -> int.to_string(frac)
  }
  int.to_string(whole) <> "." <> frac_str
}
