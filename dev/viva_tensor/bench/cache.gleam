//// Autotune persistent cache writer.
////
//// Walks all sparse + fused paths via the autotune NIFs and writes the
//// winning configs to `priv/autotune_cache.term`. The cache is a flat
//// Erlang term list of `{Path, N, Cfg, SplitK, Us, TFLOPS}` tuples,
//// readable from any BEAM project via `file:consult/1`.
////
//// Later runs (e.g. peak.gleam) can read the cache via file:consult and
//// pick the saved best config instead of re-sweeping. Saves ~30s per
//// shape when only the winner is needed.
////
//// Run: gleam run -m viva_tensor/bench/cache

import gleam/float
import gleam/int
import gleam/io
import gleam/list

pub fn main() {
  io.println(
    "\n╔══════════════════════════════════════════════════════════════════╗",
  )
  io.println("║       Autotune persistent cache writer                           ║")
  io.println(
    "╚══════════════════════════════════════════════════════════════════╝\n",
  )

  let _ = is_loaded()
  let cache_path = "priv/autotune_cache.term"
  io.println("cache target: " <> cache_path <> "\n")

  let shapes = [2048, 4096, 8192]
  let iters = 20

  let entries =
    list.flat_map(shapes, fn(n) {
      io.println("── " <> int.to_string(n) <> "×" <> int.to_string(n) <> " ──")
      [
        sweep_int4(n, iters),
        sweep_int8(n, iters),
        sweep_fp16(n, iters),
      ]
      |> list.filter_map(fn(opt) { opt })
    })

  let dump =
    entries
    |> list.map(fn(entry) {
      let #(path, n, cfg, sk, us, tflops) = entry
      "{"
      <> "\"" <> path <> "\""
      <> ", " <> int.to_string(n)
      <> ", " <> int.to_string(cfg)
      <> ", " <> int.to_string(sk)
      <> ", " <> int.to_string(us)
      <> ", " <> float.to_string(round1(tflops))
      <> "}.\n"
    })
    |> list.fold("", fn(acc, line) { acc <> line })

  case write_file(cache_path, dump) {
    Ok(_) -> io.println("\n✓ wrote " <> int.to_string(list.length(entries)) <> " entries")
    Error(reason) -> io.println("\n✗ write failed: " <> reason)
  }
}

fn sweep_int4(n: Int, iters: Int) -> Result(#(String, Int, Int, Int, Int, Float), Nil) {
  let configs = [
    10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
    33, 34, 35, 36,
  ]
  let result =
    list.fold(configs, #(0, 0, 0.0), fn(acc, cfg) {
      let #(_, best_us, _) = acc
      case cutlass_int4_sparse_bench(n, n, n, iters, cfg, 1) {
        Ok(us) if us > 0 && { best_us == 0 || us < best_us } -> {
          let flops = 2.0 *. int.to_float(n * n * n * iters)
          #(cfg, us, flops /. int.to_float(us) /. 1.0e6)
        }
        _ -> acc
      }
    })
  let #(cfg, us, tflops) = result
  case us > 0 {
    True -> {
      io.println(
        "  int4    cfg="
        <> pad_left(int.to_string(cfg), 3)
        <> "  "
        <> pad_left(int.to_string(us), 7)
        <> " µs   "
        <> float.to_string(round1(tflops))
        <> " TFLOPS",
      )
      Ok(#("int4_2_4_sparse", n, cfg, 1, us, tflops))
    }
    False -> Error(Nil)
  }
}

fn sweep_int8(n: Int, iters: Int) -> Result(#(String, Int, Int, Int, Int, Float), Nil) {
  let configs = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 25, 26, 27, 28]
  let split_ks = [1, 2]
  let result =
    list.fold(configs, #(0, 0, 0, 0.0), fn(acc, cfg) {
      list.fold(split_ks, acc, fn(inner_acc, sk) {
        let #(_, _, best_us, _) = inner_acc
        case cutlass_int8_sparse_bench_ex(n, n, n, iters, cfg, sk) {
          Ok(us) if us > 0 && { best_us == 0 || us < best_us } -> {
            let flops = 2.0 *. int.to_float(n * n * n * iters)
            #(cfg, sk, us, flops /. int.to_float(us) /. 1.0e6)
          }
          _ -> inner_acc
        }
      })
    })
  let #(cfg, sk, us, tflops) = result
  case us > 0 {
    True -> {
      io.println(
        "  int8    cfg="
        <> pad_left(int.to_string(cfg), 3)
        <> " sk="
        <> int.to_string(sk)
        <> "  "
        <> pad_left(int.to_string(us), 7)
        <> " µs   "
        <> float.to_string(round1(tflops))
        <> " TFLOPS",
      )
      Ok(#("int8_2_4_sparse", n, cfg, sk, us, tflops))
    }
    False -> Error(Nil)
  }
}

fn sweep_fp16(n: Int, iters: Int) -> Result(#(String, Int, Int, Int, Int, Float), Nil) {
  case cublaslt_fp16_algo_sweep(n, n, n, iters, 16) {
    Ok(us) if us > 0 -> {
      let flops = 2.0 *. int.to_float(n * n * n * iters)
      let tflops = flops /. int.to_float(us) /. 1.0e6
      io.println(
        "  fp16    algo-sweep   "
        <> pad_left(int.to_string(us), 7)
        <> " µs   "
        <> float.to_string(round1(tflops))
        <> " TFLOPS",
      )
      Ok(#("fp16_dense_algo_sweep", n, 0, 16, us, tflops))
    }
    _ -> Error(Nil)
  }
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

@external(erlang, "simplifile", "write")
fn write_file(path: String, content: String) -> Result(Nil, String)

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

@external(erlang, "viva_tensor_zig", "cublaslt_fp16_algo_sweep")
fn cublaslt_fp16_algo_sweep(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  max_algos: Int,
) -> Result(Int, String)
