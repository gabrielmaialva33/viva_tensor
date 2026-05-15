//// CPU performance baseline for viva_tensor vs NumPy.
////
//// For each op, this script times the pure-Gleam viva_tensor implementation
//// at the same shapes used by ``bench/perf/compare_numpy.py`` and writes one
//// CSV row per (op, size) under ``bench/perf/viva_tensor_results.csv``.
////
//// CSV columns: ``op,size,iters,total_ms,per_op_us,gflops_estimate``.
////
//// We time manually with ``erlang:monotonic_time`` (via
//// ``viva_tensor/core/ffi.now_microseconds``) instead of relying on
//// ``gleamy_bench``: ``gleamy_bench`` is dependency-only here and the manual
//// loop keeps the CSV format identical to the NumPy half.
////
//// Iteration counts are deliberately small (5 warmup + 20 timed, except for
//// the giant matmul/transpose paths which use 1+3). On a slow box the full
//// run should still finish in well under 5 minutes.
////
//// Execute: gleam run -m viva_tensor/bench/perf

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import simplifile
import viva_tensor as t
import viva_tensor/core/ffi

const out_path = "bench/perf/viva_tensor_results.csv"

// Iteration policy:
// - Most ops:        5 warmup, 20 timed.
// - Heavy matmul/transpose at 1024+: 1 warmup, 3 timed (still > 5 min budget).
const warmup_default = 5

const timed_default = 20

const warmup_heavy = 1

const timed_heavy = 3

pub fn main() {
  io.println("viva_tensor CPU benchmarks (pure Gleam, no NIF)")
  io.println(string.repeat("-", 64))

  let rows =
    []
    |> bench_matmul
    |> bench_transpose
    |> bench_add
    |> bench_mul
    |> bench_sum
    |> bench_softmax
    |> bench_layer_norm

  // Rows were prepended in reverse order — flip for readability.
  let ordered = list.reverse(rows)
  let csv = render_csv(ordered)
  let _ = simplifile.write(out_path, csv)

  io.println(string.repeat("-", 64))
  io.println(
    "wrote " <> int.to_string(list.length(ordered)) <> " rows -> " <> out_path,
  )
}

// =============================================================================
// Op benches — each prepends its row(s) to the accumulator.
// =============================================================================

fn bench_matmul(rows: List(Row)) -> List(Row) {
  // 2048x2048 in pure Gleam matmul is genuinely slow (O(n^3) over BEAM
  // floats). Keep the heavy size at small iters so the bench finishes.
  let rows =
    rows
    |> measure_op(
      "matmul",
      "1024x1024",
      warmup_heavy,
      timed_heavy,
      Some(flops_matmul(1024)),
      fn() {
        let a = t.random_uniform([1024, 1024])
        let b = t.random_uniform([1024, 1024])
        fn() {
          let _ = t.matmul(a, b)
          Nil
        }
      },
    )
  rows
  |> measure_op(
    "matmul",
    "2048x2048",
    warmup_heavy,
    timed_heavy,
    Some(flops_matmul(2048)),
    fn() {
      let a = t.random_uniform([2048, 2048])
      let b = t.random_uniform([2048, 2048])
      fn() {
        let _ = t.matmul(a, b)
        Nil
      }
    },
  )
}

fn bench_transpose(rows: List(Row)) -> List(Row) {
  rows
  |> measure_op("transpose", "2048x2048", warmup_heavy, timed_heavy, None, fn() {
    let a = t.random_uniform([2048, 2048])
    fn() {
      let _ = t.transpose(a)
      Nil
    }
  })
}

fn bench_add(rows: List(Row)) -> List(Row) {
  let rows =
    rows
    |> measure_op(
      "add",
      "1024x1024",
      warmup_default,
      timed_default,
      Some(1024 * 1024),
      fn() {
        let a = t.random_uniform([1024, 1024])
        let b = t.random_uniform([1024, 1024])
        fn() {
          let _ = t.add(a, b)
          Nil
        }
      },
    )
  rows
  |> measure_op(
    "add",
    "2048x2048",
    warmup_heavy,
    timed_heavy,
    Some(2048 * 2048),
    fn() {
      let a = t.random_uniform([2048, 2048])
      let b = t.random_uniform([2048, 2048])
      fn() {
        let _ = t.add(a, b)
        Nil
      }
    },
  )
}

fn bench_mul(rows: List(Row)) -> List(Row) {
  let rows =
    rows
    |> measure_op(
      "mul",
      "1024x1024",
      warmup_default,
      timed_default,
      Some(1024 * 1024),
      fn() {
        let a = t.random_uniform([1024, 1024])
        let b = t.random_uniform([1024, 1024])
        fn() {
          let _ = t.mul(a, b)
          Nil
        }
      },
    )
  rows
  |> measure_op(
    "mul",
    "2048x2048",
    warmup_heavy,
    timed_heavy,
    Some(2048 * 2048),
    fn() {
      let a = t.random_uniform([2048, 2048])
      let b = t.random_uniform([2048, 2048])
      fn() {
        let _ = t.mul(a, b)
        Nil
      }
    },
  )
}

fn bench_sum(rows: List(Row)) -> List(Row) {
  rows
  |> measure_op(
    "sum",
    "2048x2048",
    warmup_default,
    timed_default,
    Some(2048 * 2048),
    fn() {
      let a = t.random_uniform([2048, 2048])
      fn() {
        let _ = t.sum(a)
        Nil
      }
    },
  )
}

fn bench_softmax(rows: List(Row)) -> List(Row) {
  rows
  |> measure_op(
    "softmax",
    "512x1024",
    warmup_default,
    timed_default,
    Some(5 * 512 * 1024),
    fn() {
      let a = t.random_uniform([512, 1024])
      fn() {
        let _ = t.softmax(a, 1)
        Nil
      }
    },
  )
}

fn bench_layer_norm(rows: List(Row)) -> List(Row) {
  rows
  |> measure_op(
    "layer_norm",
    "512x1024",
    warmup_default,
    timed_default,
    Some(5 * 512 * 1024),
    fn() {
      let a = t.random_uniform([512, 1024])
      let layer = t.layer_norm_init(1024)
      fn() {
        let _ = t.layer_norm_forward(layer, a)
        Nil
      }
    },
  )
}

// =============================================================================
// Timing primitives
// =============================================================================

pub type Row {
  Row(
    op: String,
    size: String,
    iters: Int,
    total_us: Int,
    median_us: Int,
    gflops: Option(Float),
  )
}

pub type Option(a) {
  Some(a)
  None
}

/// Measure ``op_factory()`` ``timed`` times after ``warmup`` warmup runs.
/// ``op_factory`` returns the actual closure to time — this lets us allocate
/// inputs once per op (outside the timed window) and avoid re-randomizing.
fn measure_op(
  rows: List(Row),
  op: String,
  size: String,
  warmup: Int,
  timed: Int,
  flops: Option(Int),
  op_factory: fn() -> fn() -> Nil,
) -> List(Row) {
  let runner = op_factory()
  let _ = run_n_times(runner, warmup)
  let samples = collect_samples(runner, timed, [])
  let median_us = median_int(samples)
  let total_us = list.fold(samples, 0, fn(acc, s) { acc + s })
  let gflops = case flops {
    Some(f) -> Some(gflops_from(f, median_us))
    None -> None
  }
  let row =
    Row(
      op: op,
      size: size,
      iters: timed,
      total_us: total_us,
      median_us: median_us,
      gflops: gflops,
    )
  print_row(row)
  [row, ..rows]
}

fn run_n_times(runner: fn() -> Nil, n: Int) -> Nil {
  case n <= 0 {
    True -> Nil
    False -> {
      let _ = runner()
      run_n_times(runner, n - 1)
    }
  }
}

fn collect_samples(
  runner: fn() -> Nil,
  remaining: Int,
  acc: List(Int),
) -> List(Int) {
  case remaining <= 0 {
    True -> list.reverse(acc)
    False -> {
      let start = ffi.now_microseconds()
      let _ = runner()
      let elapsed = ffi.now_microseconds() - start
      collect_samples(runner, remaining - 1, [elapsed, ..acc])
    }
  }
}

fn median_int(samples: List(Int)) -> Int {
  let sorted = list.sort(samples, int.compare)
  let n = list.length(sorted)
  case n {
    0 -> 0
    _ -> {
      // Use middle element for odd, lower-middle for even (good enough here).
      let mid = n / 2
      case
        sorted
        |> list.drop(mid)
        |> list.first
      {
        Ok(v) -> v
        Error(_) -> 0
      }
    }
  }
}

fn gflops_from(flops: Int, median_us: Int) -> Float {
  case median_us <= 0 {
    True -> 0.0
    False -> int.to_float(flops) /. { int.to_float(median_us) *. 1000.0 }
  }
}

fn flops_matmul(n: Int) -> Int {
  2 * n * n * n
}

// =============================================================================
// Output
// =============================================================================

fn print_row(row: Row) {
  let gflops_str = case row.gflops {
    Some(g) -> float_to_str(g)
    None -> "-"
  }
  io.println(
    "  "
    <> pad_right(row.op, 12)
    <> " "
    <> pad_right(row.size, 13)
    <> " median="
    <> pad_left(int.to_string(row.median_us), 10)
    <> " us  gflops="
    <> gflops_str,
  )
}

fn render_csv(rows: List(Row)) -> String {
  let header = "op,size,iters,total_ms,per_op_us,gflops_estimate\n"
  list.fold(rows, header, fn(acc, row) {
    let total_ms = int.to_float(row.total_us) /. 1000.0
    let per_op_us = int.to_float(row.median_us)
    let gflops_field = case row.gflops {
      Some(g) -> float_to_str(g)
      None -> ""
    }
    acc
    <> row.op
    <> ","
    <> row.size
    <> ","
    <> int.to_string(row.iters)
    <> ","
    <> float_to_str(total_ms)
    <> ","
    <> float_to_str(per_op_us)
    <> ","
    <> gflops_field
    <> "\n"
  })
}

fn float_to_str(f: Float) -> String {
  // Round to 4 decimals via float.to_precision (available in stdlib >= 0.44).
  let rounded = float.to_precision(f, 4)
  float.to_string(rounded)
}

fn pad_right(s: String, width: Int) -> String {
  let len = string.length(s)
  case width - len {
    pad if pad > 0 -> s <> string.repeat(" ", pad)
    _ -> s
  }
}

fn pad_left(s: String, width: Int) -> String {
  let len = string.length(s)
  case width - len {
    pad if pad > 0 -> string.repeat(" ", pad) <> s
    _ -> s
  }
}
