//// Compare pure-Gleam TurboQuant vs the native NIF fast path.
//// Run: gleam run -m viva_tensor/bench/turboquant_cmp

import gleam/float
import gleam/int
import gleam/io
import gleam/list
import viva_tensor
import viva_tensor/quant/turboquant

@external(erlang, "os", "perf_counter")
fn perf_counter(unit: Int) -> Int

fn now_ns() -> Int {
  perf_counter(1_000_000_000)
}

fn round3(x: Float) -> Float {
  int.to_float(float.truncate(x *. 1000.0)) /. 1000.0
}

fn sqrt(x: Float) -> Float {
  let assert Ok(v) = float.square_root(x)
  v
}

fn rel_l2(r: List(Float), a: List(Float)) -> Float {
  let #(num, den) =
    list.zip(r, a)
    |> list.fold(#(0.0, 0.0), fn(acc, p) {
      let #(rv, av) = p
      let d = rv -. av
      let #(n, dd) = acc
      #(n +. d *. d, dd +. rv *. rv)
    })
  case den >. 0.0 {
    True -> sqrt(num) /. sqrt(den)
    False -> 0.0
  }
}

// 256-dim with ~2.5% outliers (LLM-activation-like), deterministic.
fn gen() -> List(Float) {
  do_gen(256, 777, [])
}

fn do_gen(n: Int, seed: Int, acc: List(Float)) -> List(Float) {
  case n {
    0 -> acc
    _ -> {
      let next = { seed * 1_103_515_245 + 12_345 } % 2_147_483_648
      let base = int.to_float(next % 2000 - 1000) /. 1000.0
      let v = case n % 40 {
        0 -> base *. 3.0
        _ -> base *. 0.1
      }
      do_gen(n - 1, next, [v, ..acc])
    }
  }
}

pub fn main() {
  let data = gen()
  let bits = 4
  let seed = 1234

  // --- pure Gleam round-trip ---
  let cfg = turboquant.Config(bits: bits, seed: seed, use_qjl_residual: False)
  let t0 = now_ns()
  let assert Ok(q) = turboquant.quantize(data, cfg)
  let recovered = turboquant.dequantize(q)
  let t1 = now_ns()
  let gleam_us = int.to_float(t1 - t0) /. 1000.0
  let gleam_err = rel_l2(data, recovered)

  // --- native NIF round-trip ---
  let assert Ok(tensor) = viva_tensor.native_from_list(data, [1, 256])
  let t2 = now_ns()
  let assert Ok(qt) = turboquant.quantize_dequantize_native(tensor, bits, seed)
  let native = viva_tensor.to_list(qt)
  let t3 = now_ns()
  let native_us = int.to_float(t3 - t2) /. 1000.0
  let native_err = rel_l2(data, native)

  io.println("=== TurboQuant 4-bit, 256-dim (outlier data) ===")
  io.println(
    "  pure-Gleam (uniform) : "
    <> float.to_string(round3(gleam_us))
    <> " us   err "
    <> float.to_string(round3(gleam_err *. 100.0))
    <> "%",
  )
  io.println(
    "  native NIF (Lloyd-Max): "
    <> float.to_string(round3(native_us))
    <> " us   err "
    <> float.to_string(round3(native_err *. 100.0))
    <> "%",
  )
  io.println(
    "  speedup: " <> float.to_string(round3(gleam_us /. native_us)) <> "x",
  )
}
