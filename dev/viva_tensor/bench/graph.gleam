//// CUDA Graphs vs loop launch — kernel launch overhead measurement.
////
//// We run a tiny `y = a*x + y` (axpy) kernel many times two ways:
////
////   1) `loop`  — N cudaLaunchKernel calls via stream API. Each pays the
////                full driver overhead (~5-15µs per launch).
////   2) `graph` — same kernel captured into a cudaGraph then replayed N
////                times via cudaGraphLaunch. Overhead is paid once during
////                instantiation; each replay is ~10µs flat.
////
//// For short-running kernels, graph speedup is dramatic. Real production
//// NIFs that dispatch hot loops of small GEMMs (attention heads, MoE
//// experts, fused activations) get the same benefit.
////
//// CUTLASS GEMM doesn't compose cleanly with stream capture because it
//// queries device props during the call; the launch-overhead win we
//// demonstrate here generalises to any GEMM kernel that doesn't.
////
//// Run: gleam run -m viva_tensor/bench/graph

import gleam/float
import gleam/int
import gleam/io
import gleam/list

pub fn main() {
  io.println("\n╔══════════════════════════════════════════════════════════════════╗")
  io.println("║       CUDA Graphs vs loop — kernel launch overhead               ║")
  io.println("╚══════════════════════════════════════════════════════════════════╝\n")

  let _ = is_loaded()
  io.println("NIF info: " <> backend_info())
  io.println("Kernel: axpy y = 2x + y (one fused mul-add per element)\n")

  // Vector sizes — small ones expose launch overhead, large ones amortise it.
  let sizes = [1024, 16_384, 262_144, 4_194_304, 67_108_864]
  let iters = 500

  io.println(
    "iters per row = " <> int.to_string(iters),
  )
  io.println("")
  io.println(
    "vector size   │  loop µs    graph µs   │ speedup  µs/launch loop  µs/launch graph",
  )
  io.println(
    "──────────────┼──────────────────────────┼─────────────────────────────────────────",
  )

  list.each(sizes, fn(n) {
    let loop = cuda_axpy_loop_bench(n, iters)
    let graph = cuda_axpy_graph_bench(n, iters)
    case loop, graph {
      Ok(us_loop), Ok(us_graph) if us_loop > 0 && us_graph > 0 -> {
        let speedup = int_to_float(us_loop) /. int_to_float(us_graph)
        let per_loop = int_to_float(us_loop) /. int_to_float(iters)
        let per_graph = int_to_float(us_graph) /. int_to_float(iters)
        io.println(
          pad_left(int.to_string(n), 12)
          <> "  │  "
          <> pad_left(int.to_string(us_loop), 7)
          <> "    "
          <> pad_left(int.to_string(us_graph), 7)
          <> "    │  "
          <> pad_left(float.to_string(round2(speedup)), 5)
          <> "×    "
          <> pad_left(float.to_string(round2(per_loop)), 6)
          <> "             "
          <> pad_left(float.to_string(round2(per_graph)), 5),
        )
      }
      _, _ -> io.println(int.to_string(n) <> "  skipped")
    }
  })
  io.println("")
  io.println(
    "Interpretation: where the per-launch µs (last cols) shrinks dramatically with",
  )
  io.println(
    "graph, the kernel work was short enough that launch overhead was dominant.",
  )
}

fn int_to_float(i: Int) -> Float {
  int.to_float(i)
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

@external(erlang, "viva_tensor_zig", "cuda_axpy_loop_bench")
fn cuda_axpy_loop_bench(n: Int, iters: Int) -> Result(Int, String)

@external(erlang, "viva_tensor_zig", "cuda_axpy_graph_bench")
fn cuda_axpy_graph_bench(n: Int, iters: Int) -> Result(Int, String)
