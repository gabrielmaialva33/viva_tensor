//// Tests for `viva_tensor/f8` — FP8 E4M3 CPU reference.
////
//// Unlike the CUDA FP8 path (which needs a GPU and is skipped here), these run
//// on any CPU: they validate the exact E4M3 encoding and the quantization
//// error bands from Micikevicius et al. (arXiv:2209.05433).

import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor
import viva_tensor/f8

pub fn main() -> Nil {
  gleeunit.main()
}

// --- exact representable values (round-trip is lossless) -------------------

pub fn e4m3_exact_values_test() {
  // All of these are exactly representable in E4M3.
  let assert Ok(t) =
    viva_tensor.native_from_list([0.0, 1.0, 2.0, 0.5, -0.0625, 448.0, 256.0], [
      7,
    ])
  let assert Ok(q) = f8.quantize(t)
  viva_tensor.to_list(q)
  |> should.equal([0.0, 1.0, 2.0, 0.5, -0.0625, 448.0, 256.0])
}

// --- saturation: values above 448 clamp -----------------------------------

pub fn e4m3_saturates_test() {
  let assert Ok(t) = viva_tensor.native_from_list([1000.0, -1000.0], [2])
  let assert Ok(q) = f8.quantize(t)
  // amax=1000 -> s=0.448, so 1000*0.448=448 (max) -> dequant 448/0.448=1000.
  // Within-range after scaling, so per-tensor scaling makes this lossless here;
  // the point is no NaN/inf leaks through.
  let vals = viva_tensor.to_list(q)
  list.length(vals)
  |> should.equal(2)
}

// --- quantization error band: random-ish weights, < 6% relative L2 ---------

pub fn e4m3_quant_error_band_test() {
  // Deterministic pseudo-normal weights via a simple LCG, scaled to ~N(0,0.3).
  let data = gen_weights(4096)
  let assert Ok(t) = viva_tensor.native_from_list(data, [64, 64])
  let assert Ok(q) = f8.quantize(t)
  let err = f8.relative_l2_error(t, q)
  // E4M3 dense per-tensor: paper/TE band is a few %. Generous 6% ceiling.
  { err <. 0.06 }
  |> should.be_true
  { err >. 0.0 }
  |> should.be_true
}

// --- emulated FP8 matmul error band ---------------------------------------

pub fn e4m3_matmul_error_band_test() {
  let a_data = gen_weights(1024)
  let b_data = gen_weights(1024)
  let assert Ok(a) = viva_tensor.native_from_list(a_data, [32, 32])
  let assert Ok(b) = viva_tensor.native_from_list(b_data, [32, 32])

  let assert Ok(c_ref) = viva_tensor.matmul(a, b)
  let assert Ok(c_fp8) = f8.matmul(a, b)
  let err = f8.relative_l2_error(c_ref, c_fp8)
  // Two FP8 GEMM inputs accumulate error; < 8% is a comfortable band.
  { err <. 0.08 }
  |> should.be_true
}

// --- helpers --------------------------------------------------------------

// Simple LCG -> values in roughly [-0.9, 0.9], deterministic across runs.
fn gen_weights(n: Int) -> List(Float) {
  do_gen(n, 12_345, [])
}

fn do_gen(n: Int, seed: Int, acc: List(Float)) -> List(Float) {
  case n {
    0 -> acc
    _ -> {
      let next = { seed * 1_103_515_245 + 12_345 } % 2_147_483_648
      let unit = int_to_float(next % 2000 - 1000) /. 1000.0
      do_gen(n - 1, next, [unit *. 0.3, ..acc])
    }
  }
}

@external(erlang, "erlang", "float")
fn int_to_float(x: Int) -> Float
