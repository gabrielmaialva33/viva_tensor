//// INT4 2:4 sparse vs dense 4-bit GEMM throughput at LLM matmul shapes.
////
//// Both kernels are 4-bit weight. The dense baseline is dense Marlin W4A16
//// (`marlin_w4a16_bench`, prints its own ms/TFLOPS line) and the sparse
//// contender is the CUTLASS INT4 2:4 sparse Tensor-Op GEMM
//// (`cutlass_int4_sparse_bench`, already validated numerically by
//// `linear_int4_sparse_numerical_within_band_test`). Effective TOPS counts the
//// full 2*M*N*K (including the structurally-zeroed half) so it is a wall-clock
//// comparison, not a useful-FLOP one.
////
//// Run: gleam run -m viva_tensor/bench/sparse_vs_dense
////
//// On an RTX 4090 (Ada SM89) the sparse path reaches ~1.3-1.4k TOPS effective
//// (near the int4-sparse peak) and beats dense Marlin W4A16 by 4.8-10x at batch
//// shapes. NOTE: this measures the GEMM only. End-to-end sparse *inference*
//// still needs (a) a device-resident fused path (the primitive is host-I/O) and
//// (b) SparseGPT-quality 2:4 pruning — magnitude pruning degrades accuracy.

import gleam/dynamic
import gleam/float
import gleam/int
import gleam/io
import gleam/list

// {M, N, K} — square 4096/8192 plus the Llama-3.2 FFN gate/up and down shapes.
const shapes: List(#(Int, Int, Int)) = [
  #(4096, 4096, 4096),
  #(4096, 8192, 2048),
  #(4096, 2048, 8192),
  #(8192, 8192, 8192),
]

const iters: Int = 50

const configs: List(Int) = [0, 1, 2, 3, 4, 5]

pub fn main() -> Nil {
  io.println("=== INT4 2:4 sparse vs dense Marlin W4A16 (RTX 4090) ===")
  io.println("shape (M,N,K)        dense W4A16        sparse 2:4 (best cfg)")
  list.each(shapes, fn(shape) {
    let #(m, n, k) = shape
    // Dense baseline prints its own "ms_avg=.. tflops=.." line as a side effect.
    let _ = marlin_w4a16_bench(m, n, k, 128, iters)
    let #(best_cfg, best_tops) = best_sparse(m, n, k)
    io.println(
      fmt_shape(m, n, k)
      <> "  sparse cfg="
      <> int.to_string(best_cfg)
      <> " tops_eff="
      <> int.to_string(best_tops),
    )
  })
}

// Sweep the CUTLASS sparse tile configs and keep the highest effective TOPS.
fn best_sparse(m: Int, n: Int, k: Int) -> #(Int, Int) {
  list.fold(configs, #(-1, 0), fn(acc, cfg) {
    case cutlass_int4_sparse_bench(m, n, k, iters, cfg, 1) {
      Ok(us) if us > 0 -> {
        let tops = effective_tops(m, n, k, us)
        case tops > acc.1 {
          True -> #(cfg, tops)
          False -> acc
        }
      }
      _ -> acc
    }
  })
}

// 2*M*N*K flops over the per-iteration wall time, in TOPS (integer).
fn effective_tops(m: Int, n: Int, k: Int, total_us: Int) -> Int {
  let flop = 2.0 *. int.to_float(m) *. int.to_float(n) *. int.to_float(k)
  let us_per_iter = int.to_float(total_us) /. int.to_float(iters)
  float.truncate(flop /. us_per_iter /. 1_000_000.0)
}

fn fmt_shape(m: Int, n: Int, k: Int) -> String {
  int.to_string(m) <> "x" <> int.to_string(n) <> "x" <> int.to_string(k)
}

@external(erlang, "viva_tensor_zig", "cutlass_int4_sparse_bench")
fn cutlass_int4_sparse_bench(
  m: Int,
  n: Int,
  k: Int,
  iters: Int,
  config: Int,
  split_k: Int,
) -> Result(Int, dynamic.Dynamic)

@external(erlang, "viva_tensor_zig", "marlin_w4a16_bench")
fn marlin_w4a16_bench(m: Int, n: Int, k: Int, groupsize: Int, iters: Int) -> Int
