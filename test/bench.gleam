//// Heavy Benchmarks - viva_tensor vs the world
////
//// Run: gleam run -m viva_tensor/bench

import gleam/int
import gleam/io
import gleam/list
import gleamy/bench
import viva_tensor/tensor

pub fn main() {
  io.println("╔════════════════════════════════════════════════════════════╗")
  io.println("║  viva_tensor HEAVY BENCHMARKS - Pure Gleam vs The World    ║")
  io.println("╚════════════════════════════════════════════════════════════╝\n")

  // Warm up BEAM
  let _ = tensor.ones([100, 100])

  // 1. MATMUL - The big one
  io.println("━━━ MATRIX MULTIPLICATION ━━━")
  bench_matmul()

  // 2. Element-wise ops at scale
  io.println("\n━━━ ELEMENT-WISE OPERATIONS (100K elements) ━━━")
  bench_elementwise()

  // 3. Reductions
  io.println("\n━━━ REDUCTIONS (100K elements) ━━━")
  bench_reductions()

  // 4. Broadcasting
  io.println("\n━━━ BROADCASTING ━━━")
  bench_broadcasting()

  // 5. Memory efficiency - Strided
  io.println("\n━━━ ZERO-COPY (Strided) vs COPY ━━━")
  bench_strided()

  io.println("\n╔════════════════════════════════════════════════════════════╗")
  io.println("║  BENCHMARKS COMPLETE - Gleam tensor lib ready for battle!  ║")
  io.println("╚════════════════════════════════════════════════════════════╝")
}

fn bench_matmul() {
  let sizes = [32, 64, 128, 256]

  list.each(sizes, fn(n) {
    let a = tensor.ones([n, n])
    let b = tensor.ones([n, n])

    bench.run(
      [bench.Input(int.to_string(n) <> "x" <> int.to_string(n), #(a, b))],
      [
        bench.Function("matmul", fn(pair) {
          let #(x, y) = pair
          fn() {
            let _ = tensor.matmul(x, y)
            Nil
          }
        }),
      ],
      [bench.Duration(2000), bench.Warmup(200)],
    )
    |> bench.table([bench.IPS, bench.Min, bench.Mean])
    |> io.println()
  })
}

fn bench_elementwise() {
  let a = tensor.random_uniform([1000, 100])
  let b = tensor.random_uniform([1000, 100])

  bench.run(
    [bench.Input("100K floats", #(a, b))],
    [
      bench.Function("add", fn(pair) {
        let #(x, y) = pair
        fn() {
          let _ = tensor.add(x, y)
          Nil
        }
      }),
      bench.Function("mul", fn(pair) {
        let #(x, y) = pair
        fn() {
          let _ = tensor.mul(x, y)
          Nil
        }
      }),
      bench.Function("scale 2x", fn(pair) {
        let #(x, _) = pair
        fn() {
          let _ = tensor.scale(x, 2.0)
          Nil
        }
      }),
    ],
    [bench.Duration(2000), bench.Warmup(200)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.Mean, bench.Max])
  |> io.println()
}

fn bench_reductions() {
  let t = tensor.random_uniform([100_000])

  bench.run(
    [bench.Input("100K", t)],
    [
      bench.Function("sum", fn(x) {
        fn() {
          let _ = tensor.sum(x)
          Nil
        }
      }),
      bench.Function("mean", fn(x) {
        fn() {
          let _ = tensor.mean(x)
          Nil
        }
      }),
      bench.Function("max", fn(x) {
        fn() {
          let _ = tensor.max(x)
          Nil
        }
      }),
      bench.Function("variance", fn(x) {
        fn() {
          let _ = tensor.variance(x)
          Nil
        }
      }),
    ],
    [bench.Duration(2000), bench.Warmup(200)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.Mean, bench.Max])
  |> io.println()
}

fn bench_broadcasting() {
  let a = tensor.ones([1000, 100])
  let b = tensor.ones([100])
  // broadcast [100] to [1000, 100]

  bench.run(
    [bench.Input("[1000,100] + [100]", #(a, b))],
    [
      bench.Function("add_broadcast", fn(pair) {
        let #(x, y) = pair
        fn() {
          let _ = tensor.add_broadcast(x, y)
          Nil
        }
      }),
      bench.Function("mul_broadcast", fn(pair) {
        let #(x, y) = pair
        fn() {
          let _ = tensor.mul_broadcast(x, y)
          Nil
        }
      }),
    ],
    [bench.Duration(2000), bench.Warmup(200)],
  )
  |> bench.table([bench.IPS, bench.Min, bench.Mean, bench.Max])
  |> io.println()
}

fn bench_strided() {
  let list_t = tensor.ones([500, 500])
  let strided_t = tensor.to_strided(list_t)

  io.println("250K elements - transpose comparison:")

  // Transpose benchmark
  bench.run(
    [bench.Input("List backend", list_t)],
    [
      bench.Function("transpose (copy)", fn(t) {
        fn() {
          let _ = tensor.transpose(t)
          Nil
        }
      }),
    ],
    [bench.Duration(1000), bench.Warmup(100)],
  )
  |> bench.table([bench.IPS, bench.Mean])
  |> io.println()

  bench.run(
    [bench.Input("Strided backend", strided_t)],
    [
      bench.Function("transpose (zero-copy)", fn(t) {
        fn() {
          let _ = tensor.transpose_strided(t)
          Nil
        }
      }),
    ],
    [bench.Duration(1000), bench.Warmup(100)],
  )
  |> bench.table([bench.IPS, bench.Mean])
  |> io.println()

  // Show the speed difference
  io.println("\n⚡ Zero-copy transpose is O(1) vs O(n²) for copy!")
}
