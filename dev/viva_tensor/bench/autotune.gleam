//// CUTLASS INT4 2:4 sparse autotuner sweep.
////
//// CUTLASS exposes 30+ kernel configurations for the INT4 2:4 sparse GEMM
//// path on Ada (SM89). They differ in:
////   - Threadblock shape (128x128x256, 256x128x256, etc.)
////   - Warp shape (32x32x256, 64x64x256, ...)
////   - Stages (2, 3, 4)
////   - Swizzle (NoSwizzle, Swizzle<4>, Swizzle<8>)
////   - Epilogue (Linear, LinearCombinationClamp)
////
//// The default `config=10` used by `peak.gleam` is a reasonable starting
//// point, but the optimal choice depends on shape. This sweep finds it.
////
//// Run: gleam run -m viva_tensor/bench/autotune

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/order
import gleam/result

pub fn main() {
  io.println("\n╔══════════════════════════════════════════════════════════════════╗")
  io.println("║   CUTLASS INT4 2:4 sparse autotuner — RTX 4090                   ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝\n")

  let _ = is_loaded()
  io.println("NIF info: " <> backend_info() <> "\n")

  // Known-good configs distilled from earlier sweep (skip 6-9, 16-19, 37-39
  // which return -100 = invalid_config on Ada SM89). Top performers across
  // 4096² were 22-36 (Universal variants with different swizzles/epilogues).
  let configs = [
    10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36,
  ]
  let shapes = [2048, 4096, 8192]
  let iters = 20

  list.each(shapes, fn(n) {
    io.println(
      "─── "
      <> int.to_string(n)
      <> "×"
      <> int.to_string(n)
      <> " (iters="
      <> int.to_string(iters)
      <> ") ───",
    )

    let flops = 2.0 *. int_to_float(n * n * n * iters)
    let results =
      configs
      |> list.filter_map(fn(cfg) {
        case cutlass_int4_sparse_bench(n, n, n, iters, cfg, 1) {
          Ok(us) if us > 0 -> Ok(#(cfg, us, flops /. int_to_float(us) /. 1.0e6))
          _ -> Error(Nil)
        }
      })
      |> list.sort(fn(a, b) {
        let #(_, _, ta) = a
        let #(_, _, tb) = b
        case tb >. ta {
          True -> order.Gt
          False -> order.Lt
        }
      })

    let top = list.take(results, 5)
    io.println("  top 5 configs (TFLOPS, sorted desc):")
    list.each(top, fn(triple) {
      let #(cfg, us, tflops) = triple
      io.println(
        "    cfg="
        <> pad_left(int.to_string(cfg), 3)
        <> "  "
        <> pad_left(int.to_string(us), 7)
        <> " µs   "
        <> pad_left(float.to_string(round1(tflops)), 7)
        <> " TFLOPS",
      )
    })

    case top {
      [#(best_cfg, _, best_tflops), ..] -> {
        let baseline_tflops =
          list.find(results, fn(triple) {
            let #(cfg, _, _) = triple
            cfg == 10
          })
          |> result.map(fn(triple) {
            let #(_, _, t) = triple
            t
          })
          |> result.unwrap(0.0)
        let speedup = case baseline_tflops >. 0.0 {
          True -> best_tflops /. baseline_tflops
          False -> 1.0
        }
        io.println(
          "  → best: cfg="
          <> int.to_string(best_cfg)
          <> "  ("
          <> float.to_string(round2(speedup))
          <> "× vs cfg=10 default)",
        )
      }
      _ -> Nil
    }
    io.println("")
  })

  io.println(
    "Use the winning config in `peak.gleam` by changing the last arg to",
  )
  io.println("`cutlass_int4_sparse_bench(n, n, n, iters, BEST_CFG, 1)`.")
}

fn int_to_float(i: Int) -> Float {
  int.to_float(i)
}

fn round1(x: Float) -> Float {
  float.to_precision(x, 1)
}

fn round2(x: Float) -> Float {
  float.to_precision(x, 2)
}

fn pad_left(s: String, n: Int) -> String {
  let len = string_length(s)
  case len >= n {
    True -> s
    False -> repeat(" ", n - len) <> s
  }
}

fn repeat(s: String, n: Int) -> String {
  case n <= 0 {
    True -> ""
    False -> s <> repeat(s, n - 1)
  }
}

@external(erlang, "string", "length")
fn string_length(s: String) -> Int

@external(erlang, "viva_tensor_zig", "is_loaded")
fn is_loaded() -> Bool

@external(erlang, "viva_tensor_zig", "backend_info")
fn backend_info() -> String

@external(erlang, "viva_tensor_zig", "cutlass_int4_sparse_bench")
fn cutlass_int4_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  config: Int,
  split_k: Int,
) -> Result(Int, String)
