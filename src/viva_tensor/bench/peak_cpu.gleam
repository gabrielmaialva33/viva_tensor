//// Quick peak-GFLOPS matmul benchmark for the CPU NIF path (MKL).
//// Run: gleam run -m viva_tensor/bench/peak_cpu

import gleam/float
import gleam/int
import gleam/io
import viva_tensor

@external(erlang, "os", "perf_counter")
fn perf_counter(unit: Int) -> Int

fn now_ns() -> Int {
  perf_counter(1_000_000_000)
}

fn round2(x: Float) -> Float {
  int.to_float(float.truncate(x *. 100.0)) /. 100.0
}

fn repeat_matmul(
  f: fn(viva_tensor.Tensor, viva_tensor.Tensor) -> Result(viva_tensor.Tensor, a),
  a: viva_tensor.Tensor,
  b: viva_tensor.Tensor,
  n: Int,
) -> Nil {
  case n {
    0 -> Nil
    _ -> {
      let _ = f(a, b)
      repeat_matmul(f, a, b, n - 1)
    }
  }
}

fn bench_one(label: String, n: Int, iters: Int, f) -> Nil {
  // native_fill builds NativeTensor refs so matmul routes to the MKL NIF.
  let assert Ok(a) = viva_tensor.native_fill([n, n], 0.5)
  let assert Ok(b) = viva_tensor.native_fill([n, n], 0.25)

  // warmup
  let _ = f(a, b)

  let t0 = now_ns()
  repeat_matmul(f, a, b, iters)
  let t1 = now_ns()

  let total_ns = int.to_float(t1 - t0)
  let avg_s = total_ns /. int.to_float(iters) /. 1_000_000_000.0
  // matmul FLOPs = 2 * n^3
  let nf = int.to_float(n)
  let flops = 2.0 *. nf *. nf *. nf
  let gflops = flops /. avg_s /. 1_000_000_000.0
  let ms = avg_s *. 1000.0

  io.println(
    "  "
    <> label
    <> "  "
    <> int.to_string(n)
    <> "x"
    <> int.to_string(n)
    <> "  ->  "
    <> float.to_string(round2(ms))
    <> " ms/iter   "
    <> float.to_string(round2(gflops))
    <> " GFLOP/s",
  )
}

pub fn main() {
  io.println("")
  io.println("=== viva_tensor CPU matmul: DGEMM (FP64) vs SGEMM (FP32) ===")
  bench_one("dgemm", 512, 20, viva_tensor.matmul)
  bench_one("sgemm", 512, 20, viva_tensor.matmul_f32)
  bench_one("dgemm", 1024, 10, viva_tensor.matmul)
  bench_one("sgemm", 1024, 10, viva_tensor.matmul_f32)
  bench_one("dgemm", 2048, 5, viva_tensor.matmul)
  bench_one("sgemm", 2048, 5, viva_tensor.matmul_f32)
  bench_one("dgemm", 4096, 3, viva_tensor.matmul)
  bench_one("sgemm", 4096, 3, viva_tensor.matmul_f32)
  io.println("")
}
