//// Compare FP64 DGEMM vs FP32-cast SGEMM vs FP32-native (first-class).
//// Run: gleam run -m viva_tensor/bench/peak_f32

import gleam/float
import gleam/int
import gleam/io
import viva_tensor
import viva_tensor/f32

@external(erlang, "os", "perf_counter")
fn perf_counter(unit: Int) -> Int

fn now_ns() -> Int {
  perf_counter(1_000_000_000)
}

fn round2(x: Float) -> Float {
  int.to_float(float.truncate(x *. 100.0)) /. 100.0
}

fn gflops_line(label: String, n: Int, iters: Int, t0: Int, t1: Int) -> Nil {
  let avg_s = int.to_float(t1 - t0) /. int.to_float(iters) /. 1_000_000_000.0
  let nf = int.to_float(n)
  let g = 2.0 *. nf *. nf *. nf /. avg_s /. 1_000_000_000.0
  io.println(
    "  "
    <> label
    <> "  "
    <> int.to_string(n)
    <> "x"
    <> int.to_string(n)
    <> "  ->  "
    <> float.to_string(round2(avg_s *. 1000.0))
    <> " ms/iter   "
    <> float.to_string(round2(g))
    <> " GFLOP/s",
  )
}

fn rep64(a, b, n: Int) -> Nil {
  case n {
    0 -> Nil
    _ -> {
      let _ = viva_tensor.matmul(a, b)
      rep64(a, b, n - 1)
    }
  }
}

fn rep64_cast(a, b, n: Int) -> Nil {
  case n {
    0 -> Nil
    _ -> {
      let _ = viva_tensor.matmul_f32(a, b)
      rep64_cast(a, b, n - 1)
    }
  }
}

fn repf32(a, b, n: Int) -> Nil {
  case n {
    0 -> Nil
    _ -> {
      let _ = f32.matmul(a, b)
      repf32(a, b, n - 1)
    }
  }
}

fn bench(n: Int, iters: Int) -> Nil {
  let assert Ok(a64) = viva_tensor.native_fill([n, n], 0.5)
  let assert Ok(b64) = viva_tensor.native_fill([n, n], 0.25)
  let assert Ok(a32) = f32.fill([n, n], 0.5)
  let assert Ok(b32) = f32.fill([n, n], 0.25)

  let _ = viva_tensor.matmul(a64, b64)
  let t0 = now_ns()
  rep64(a64, b64, iters)
  let t1 = now_ns()
  gflops_line("dgemm-fp64 ", n, iters, t0, t1)

  let _ = viva_tensor.matmul_f32(a64, b64)
  let t2 = now_ns()
  rep64_cast(a64, b64, iters)
  let t3 = now_ns()
  gflops_line("sgemm-cast ", n, iters, t2, t3)

  let _ = f32.matmul(a32, b32)
  let t4 = now_ns()
  repf32(a32, b32, iters)
  let t5 = now_ns()
  gflops_line("f32-native ", n, iters, t4, t5)
  io.println("")
}

pub fn main() {
  io.println("")
  io.println("=== matmul: FP64 vs FP32-cast vs FP32-native ===")
  bench(512, 30)
  bench(1024, 12)
  bench(2048, 6)
  bench(4096, 3)
}
