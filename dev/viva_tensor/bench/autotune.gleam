//// CUTLASS / cuSPARSELt sparse autotuner sweep.
////
//// Sweeps kernel configs across three sparse Tensor Core paths:
////   1) CUTLASS INT4 2:4 sparse  (configs 0-36, split-K 1 + 2)
////   2) CUTLASS INT8 2:4 sparse  (configs 0-28, split-K 1 + 2)
////   3) cuSPARSELt INT8 2:4      (modes 0=auto, 2=splitK1k, 3=splitK2k)
////
//// Reports the winning config per shape. Re-running `peak.gleam` with
//// these in the top-of-file lookup table gives the highest sustained
//// TFLOPS the hardware can deliver on Ada SM89.
////
//// Run: gleam run -m viva_tensor/bench/autotune

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/order
import gleam/result

pub fn main() {
  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println(
    "║   viva_tensor sparse autotuner — RTX 4090                        ║",
  )
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  let _ = is_loaded()
  io.println("NIF info: " <> backend_info() <> "\n")

  let shapes = [2048, 4096, 8192]
  let iters = 20

  list.each(shapes, fn(n) {
    io.println(
      "════ "
      <> int.to_string(n)
      <> "×"
      <> int.to_string(n)
      <> " (iters="
      <> int.to_string(iters)
      <> ") ════",
    )
    let flops = 2.0 *. int_to_float(n * n * n * iters)

    sweep_int4(n, iters, flops)
    sweep_int8_cutlass(n, iters, flops)
    sweep_cusparselt(n, iters, flops)

    io.println("")
  })
}

// =============================================================================
// CUTLASS INT4 2:4 sparse — configs × split-K
// =============================================================================

fn sweep_int4(n: Int, iters: Int, flops: Float) -> Nil {
  let configs = [
    10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36,
  ]
  let split_ks = [1, 2]
  io.println("┌─ CUTLASS INT4 2:4 sparse (sweep configs × split-K)")
  let results =
    list.flat_map(configs, fn(cfg) {
      list.filter_map(split_ks, fn(sk) {
        case cutlass_int4_sparse_bench(n, n, n, iters, cfg, sk) {
          Ok(us) if us > 0 ->
            Ok(#(cfg, sk, us, flops /. int_to_float(us) /. 1.0e6))
          _ -> Error(Nil)
        }
      })
    })
    |> sort_by_tflops()
  print_top("INT4", results, 3)
}

// =============================================================================
// CUTLASS INT8 2:4 sparse — configs × split-K (now timed via cudaEvent)
// =============================================================================

fn sweep_int8_cutlass(n: Int, iters: Int, flops: Float) -> Nil {
  let configs = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28]
  let split_ks = [1, 2]
  io.println("├─ CUTLASS INT8 2:4 sparse (sweep configs × split-K)")
  let results =
    list.flat_map(configs, fn(cfg) {
      list.filter_map(split_ks, fn(sk) {
        case cutlass_int8_sparse_bench_ex(n, n, n, iters, cfg, sk) {
          Ok(us) if us > 0 ->
            Ok(#(cfg, sk, us, flops /. int_to_float(us) /. 1.0e6))
          _ -> Error(Nil)
        }
      })
    })
    |> sort_by_tflops()
  print_top("INT8 CUTLASS", results, 3)
}

// =============================================================================
// cuSPARSELt INT8 2:4 — modes
// =============================================================================

fn sweep_cusparselt(n: Int, iters: Int, flops: Float) -> Nil {
  // mode 0 = MatmulSearch auto (works); modes 2/3 hang on some shapes,
  // so we don't sweep them by default. Re-add them if you've patched
  // cuda_cusparselt_int8.cu to be reentrant.
  let modes = [0]
  io.println("└─ cuSPARSELt INT8 2:4 sparse (mode 0 only — auto-search)")
  let results =
    list.filter_map(modes, fn(mode) {
      case cusparselt_int8_sparse_bench(n, n, n, iters, mode) {
        Ok(us) if us > 0 ->
          Ok(#(mode, 1, us, flops /. int_to_float(us) /. 1.0e6))
        _ -> Error(Nil)
      }
    })
    |> sort_by_tflops()
  print_top("cuSPARSELt", results, 3)
}

// =============================================================================
// Helpers
// =============================================================================

fn sort_by_tflops(
  rs: List(#(Int, Int, Int, Float)),
) -> List(#(Int, Int, Int, Float)) {
  list.sort(rs, fn(a, b) {
    let #(_, _, _, ta) = a
    let #(_, _, _, tb) = b
    case tb >. ta {
      True -> order.Gt
      False -> order.Lt
    }
  })
}

fn print_top(
  label: String,
  results: List(#(Int, Int, Int, Float)),
  k: Int,
) -> Nil {
  let baseline =
    list.first(results)
    |> result.map(fn(triple) {
      let #(_, _, _, t) = triple
      t
    })
    |> result.unwrap(0.0)
  let _ = baseline
  list.each(list.take(results, k), fn(quad) {
    let #(cfg, sk, us, tflops) = quad
    io.println(
      "   "
      <> pad_left(label, 13)
      <> "  cfg="
      <> pad_left(int.to_string(cfg), 3)
      <> " split_k="
      <> int.to_string(sk)
      <> "  "
      <> pad_left(int.to_string(us), 7)
      <> " µs   "
      <> pad_left(float.to_string(round1(tflops)), 7)
      <> " TFLOPS",
    )
  })
}

fn int_to_float(i: Int) -> Float {
  int.to_float(i)
}

fn round1(x: Float) -> Float {
  float.to_precision(x, 1)
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

@external(erlang, "viva_tensor_zig", "cutlass_int8_sparse_bench_ex")
fn cutlass_int8_sparse_bench_ex(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  config: Int,
  split_k: Int,
) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cusparselt_int8_sparse_bench")
fn cusparselt_int8_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  mode: Int,
) -> Result(Int, String)
