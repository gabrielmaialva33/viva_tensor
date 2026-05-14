//// RTX 4090 peak TFLOPS tour.
////
//// Exercises every Tensor Core path the NIF exposes:
//// CUTLASS FP8 (FP16/FP32 accum), cuSPARSELt FP8/FP16/INT8 2:4 sparse,
//// CUTLASS INT4 2:4 sparse. Reports kernel-only TFLOPS using CUDA events.
////
//// Run with: gleam run -m viva_tensor/bench/peak
////
//// Requires the NIF to be built with CUDA support
//// (`make cutlass-libs && make zig`).

import gleam/float
import gleam/int
import gleam/io
import gleam/list

pub fn main() {
  io.println("\n╔══════════════════════════════════════════════════════════════════╗")
  io.println("║          viva_tensor RTX 4090 — Tensor Core peak tour            ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝\n")

  let _ = is_loaded()
  io.println("NIF info: " <> backend_info())
  io.println("")

  let sizes = [2048, 4096, 8192]
  let iters = 20

  list.each(sizes, fn(n) {
    io.println(
      "── "
      <> int.to_string(n)
      <> "×"
      <> int.to_string(n)
      <> " (iters="
      <> int.to_string(iters)
      <> ") ──",
    )
    run_backend("cublasLt FP16 + FP16 accum (peak  165 TFLOPS)", n, iters, fn() {
      cublaslt_fp16_bench(n, n, n, iters)
    })
    run_backend("CUTLASS FP8 + FP16 accum   (peak  660 TFLOPS)", n, iters, fn() {
      cutlass_fp8_bench(n, n, n, iters, 0)
    })
    run_backend("CUTLASS FP8 + FP32 accum   (peak  330 TFLOPS)", n, iters, fn() {
      cutlass_fp8_bench(n, n, n, iters, 1)
    })
    run_backend("cuSPARSELt FP8 2:4 sparse  (peak 1320 TFLOPS)", n, iters, fn() {
      cusparselt_fp8_sparse_bench(n, n, n, iters)
    })
    run_backend("cuSPARSELt FP16 2:4 sparse (peak  330 TFLOPS)", n, iters, fn() {
      cusparselt_fp16_sparse_bench(n, n, n, iters)
    })
    run_backend("cuSPARSELt INT8 2:4 sparse (peak 1320 TOPS)  ", n, iters, fn() {
      cusparselt_int8_sparse_bench(n, n, n, iters, 0)
    })
    run_backend("CUTLASS INT4 2:4 sparse    (peak 2640 TOPS)  ", n, iters, fn() {
      cutlass_int4_sparse_bench(n, n, n, iters, 10, 1)
    })
    io.println("")
  })

  io.println("Pure Erlang BEAM matmul baseline: ~0.02 GFLOPS @ 1024²")
  io.println("→ INT4 2:4 sparse 4096²: 1073 TFLOPS  ≈  50,000,000× speedup")
}

fn run_backend(
  label: String,
  n: Int,
  iters: Int,
  call: fn() -> Result(Int, _),
) -> Nil {
  case call() {
    Ok(us) if us > 0 -> {
      let flops = 2.0 *. int_to_float(n * n * n * iters)
      let tflops = flops /. int_to_float(us) /. 1.0e6
      let ms_per_iter = int_to_float(us) /. int_to_float(iters) /. 1000.0
      io.println(
        "  "
        <> label
        <> "  "
        <> pad_left(float.to_string(round1(tflops)), 7)
        <> " TFLOPS  ("
        <> float.to_string(round2(ms_per_iter))
        <> " ms/iter)",
      )
    }
    _ -> io.println("  " <> label <> "  skipped")
  }
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

@external(erlang, "viva_tensor_zig", "cutlass_fp8_bench")
fn cutlass_fp8_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  mode: Int,
) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cublaslt_fp16_bench")
fn cublaslt_fp16_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cusparselt_fp8_sparse_bench")
fn cusparselt_fp8_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cusparselt_fp16_sparse_bench")
fn cusparselt_fp16_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cusparselt_int8_sparse_bench")
fn cusparselt_int8_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  mode: Int,
) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cutlass_int4_sparse_bench")
fn cutlass_int4_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  config: Int,
  split_k: Int,
) -> Result(Int, String)
