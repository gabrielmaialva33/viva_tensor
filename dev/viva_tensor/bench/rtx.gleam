//// RTX 4090 vs MKL benchmark for the accelerated tensor API.
////
//// Run outside the sandbox for real CUDA access:
////   gleam run -m viva_tensor/bench/rtx

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import gleam/string
import viva_tensor as t
import viva_tensor/core/ffi

pub fn main() {
  io.println("╔════════════════════════════════════════════════════════════╗")
  io.println("║  viva_tensor RTX 4090 vs MKL benchmark                    ║")
  io.println("╚════════════════════════════════════════════════════════════╝")
  io.println("")
  io.println("GPU timings call accelerated_sync() after every operation.")
  io.println("That measures completed work, not just CUDA enqueue latency.")
  io.println("")

  list.each([128, 256, 512, 1024], bench_square_size)
}

fn bench_square_size(n: Int) {
  let iterations = iterations_for(n)
  let shape = [n, n]
  let bias_shape = [n]
  let a = t.ones(shape)
  let b = t.ones(shape)
  let bias = t.ones(bias_shape)

  io.println("━━━ " <> int.to_string(n) <> "x" <> int.to_string(n) <> " ━━━")
  io.println("iterations: " <> int.to_string(iterations))

  bench_mkl(n, iterations, a, b, bias)
  bench_rtx(n, iterations, a, b, bias)

  io.println("")
}

fn bench_mkl(
  n: Int,
  iterations: Int,
  a: t.Tensor,
  b: t.Tensor,
  bias: t.Tensor,
) {
  case
    t.native_from_list(t.to_list(a), [n, n]),
    t.native_from_list(t.to_list(b), [n, n]),
    t.native_from_list(t.to_list(bias), [n]),
    t.native_zeros([n, n])
  {
    Ok(a_native), Ok(b_native), Ok(bias_native), Ok(out_native) -> {
      run_matmul_case("mkl native matmul alloc", n, iterations, fn() {
        let _ = t.matmul(a_native, b_native)
        Nil
      })

      run_matmul_case("mkl native matmul_into", n, iterations, fn() {
        let _ = t.matmul_into(out_native, a_native, b_native)
        Nil
      })

      run_matmul_case("mkl native linear_relu_into", n, iterations, fn() {
        let _ = t.linear_relu_into(out_native, a_native, b_native, bias_native)
        Nil
      })
    }

    _, _, _, _ ->
      io.println("  " <> pad_right("mkl native", 32) <> "unavailable")
  }
}

fn bench_rtx(
  n: Int,
  iterations: Int,
  a: t.Tensor,
  b: t.Tensor,
  bias: t.Tensor,
) {
  case
    t.to_rtx4090_fp16(a),
    t.to_rtx4090_fp16(b),
    t.to_rtx4090_fp16(bias),
    t.to_rtx4090_fp16(t.zeros([n, n])),
    t.gpu_workspace()
  {
    Ok(a_gpu), Ok(b_gpu), Ok(bias_gpu), Ok(out_gpu), Ok(workspace) -> {
      run_matmul_case("rtx matmul_auto upload/call", n, iterations, fn() {
        let _ = t.matmul_auto(a, b)
        let _ = t.accelerated_sync()
        Nil
      })

      run_matmul_case("rtx fp16 persistent alloc", n, iterations, fn() {
        let _ = t.matmul_accelerated(a_gpu, b_gpu)
        let _ = t.accelerated_sync()
        Nil
      })

      run_matmul_case("rtx fp16 matmul_into", n, iterations, fn() {
        let _ = t.matmul_accelerated_into(out_gpu, a_gpu, b_gpu)
        let _ = t.accelerated_sync()
        Nil
      })

      run_matmul_case("rtx fp16 linear_relu_into", n, iterations, fn() {
        let _ = t.linear_relu_accelerated_into(out_gpu, a_gpu, b_gpu, bias_gpu)
        let _ = t.accelerated_sync()
        Nil
      })

      case
        t.workspace_from_tensor(workspace, a),
        t.linear_layer(workspace, b, bias)
      {
        Ok(input), Ok(layer) -> {
          case t.linear_output(workspace, layer, n) {
            Ok(layer_out) ->
              run_matmul_case("rtx workspace linear layer", n, iterations, fn() {
                let _ = t.linear_relu_forward_into(layer_out, input, layer)
                let _ = t.accelerated_sync()
                Nil
              })
            Error(_) -> Nil
          }
        }

        _, _ -> Nil
      }
    }

    _, _, _, _, _ ->
      io.println("  " <> pad_right("rtx fp16", 32) <> "unavailable")
  }
}

fn run_matmul_case(label: String, n: Int, iterations: Int, f: fn() -> Nil) {
  f()

  let start = ffi.now_microseconds()
  range_int(1, iterations)
  |> list.each(fn(_) { f() })
  let stop = ffi.now_microseconds()

  let avg_us = int.to_float(stop - start) /. int.to_float(iterations)
  let gflops = matmul_gflops(n, avg_us)

  io.println(
    "  "
    <> pad_right(label, 32)
    <> pad_left(float_to_string_3(avg_us /. 1000.0), 10)
    <> " ms  "
    <> pad_left(float_to_string_1(gflops), 10)
    <> " GFLOPS",
  )
}

fn matmul_gflops(n: Int, avg_us: Float) -> Float {
  let nf = int.to_float(n)
  let flops = 2.0 *. nf *. nf *. nf
  flops /. avg_us /. 1000.0
}

fn iterations_for(n: Int) -> Int {
  case n {
    128 -> 40
    256 -> 25
    512 -> 10
    _ -> 5
  }
}

fn float_to_string_1(value: Float) -> String {
  let rounded = int.to_float(float.round(value *. 10.0)) /. 10.0
  float.to_string(rounded)
}

fn float_to_string_3(value: Float) -> String {
  let rounded = int.to_float(float.round(value *. 1000.0)) /. 1000.0
  float.to_string(rounded)
}

fn pad_right(value: String, width: Int) -> String {
  let size = string.length(value)
  case size >= width {
    True -> value
    False -> value <> string.repeat(" ", width - size)
  }
}

fn pad_left(value: String, width: Int) -> String {
  let size = string.length(value)
  case size >= width {
    True -> value
    False -> string.repeat(" ", width - size) <> value
  }
}

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}
