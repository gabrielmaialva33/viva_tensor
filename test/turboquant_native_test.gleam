//// Tests for the TurboQuant native fast path
//// (`turboquant.quantize_dequantize_native`) — the NIF the module header
//// promises. Runs on CPU; validates shape, the no-NaN zero case, and that the
//// random rotation crushes outlier error vs. naive uniform quantization.

import gleam/float
import gleam/list
import gleeunit/should
import viva_tensor
import viva_tensor/quant/turboquant

// Shape preserved through the native round-trip.
pub fn native_roundtrip_shape_test() {
  let assert Ok(t) = viva_tensor.native_from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
  let assert Ok(q) = turboquant.quantize_dequantize_native(t, 4, 1234)
  viva_tensor.shape(q)
  |> should.equal([2, 2])
  list.length(viva_tensor.to_list(q))
  |> should.equal(4)
}

// Zero tensor stays zero (no divide-by-norm NaN).
pub fn native_zeros_test() {
  let assert Ok(t) = viva_tensor.native_from_list([0.0, 0.0, 0.0, 0.0], [1, 4])
  let assert Ok(q) = turboquant.quantize_dequantize_native(t, 4, 1)
  viva_tensor.to_list(q)
  |> should.equal([0.0, 0.0, 0.0, 0.0])
}

// More bits -> lower error.
pub fn native_more_bits_less_error_test() {
  let data = outlier_vec()
  let assert Ok(t) = viva_tensor.native_from_list(data, [1, 16])
  let assert Ok(q4) = turboquant.quantize_dequantize_native(t, 4, 7)
  let assert Ok(q8) = turboquant.quantize_dequantize_native(t, 8, 7)
  {
    rel_l2(data, viva_tensor.to_list(q8))
    <. rel_l2(data, viva_tensor.to_list(q4))
  }
  |> should.be_true
}

// Headline property: in high dimension, the random rotation crushes outlier
// error vs. naive uniform quantization. (In low dim the rotation can't
// concentrate, so this only holds for large vectors — see the paper.)
pub fn native_beats_uniform_on_outliers_test() {
  let data = outlier_vec_256()
  let assert Ok(t) = viva_tensor.native_from_list(data, [1, 256])
  let assert Ok(q) = turboquant.quantize_dequantize_native(t, 4, 1234)
  let tq_err = rel_l2(data, viva_tensor.to_list(q))
  { tq_err <. uniform_rel_error(data, 4) }
  |> should.be_true
}

// Inner-product estimate is finite and reasonably close to the exact dot.
pub fn native_inner_product_sanity_test() {
  let q = gen256(256, 111, [])
  let k = gen256(256, 222, [])
  let assert Ok(qt) = viva_tensor.native_from_list(q, [256])
  let assert Ok(kt) = viva_tensor.native_from_list(k, [256])
  let exact = dot(q, k)
  let assert Ok(est) = turboquant.inner_product_native(qt, kt, 4, 7, True)
  // 4-bit estimate of a ~N(0,16) dot: within a few units.
  { float.absolute_value(est -. exact) <. 5.0 }
  |> should.be_true
}

// QJL residual lowers the inner-product error, summed over several rotations.
pub fn native_qjl_reduces_ip_error_test() {
  let q = gen256(256, 333, [])
  let k = gen256(256, 444, [])
  let assert Ok(qt) = viva_tensor.native_from_list(q, [256])
  let assert Ok(kt) = viva_tensor.native_from_list(k, [256])
  let exact = dot(q, k)
  let with_qjl = sum_ip_error(qt, kt, exact, True)
  let no_qjl = sum_ip_error(qt, kt, exact, False)
  { with_qjl <. no_qjl }
  |> should.be_true
}

fn sum_ip_error(
  qt: viva_tensor.Tensor,
  kt: viva_tensor.Tensor,
  exact: Float,
  use_qjl: Bool,
) -> Float {
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
  |> list.fold(0.0, fn(acc, seed) {
    case turboquant.inner_product_native(qt, kt, 3, seed, use_qjl) {
      Ok(est) -> acc +. float.absolute_value(est -. exact)
      Error(_) -> acc
    }
  })
}

fn dot(a: List(Float), b: List(Float)) -> Float {
  list.zip(a, b)
  |> list.fold(0.0, fn(acc, p) {
    let #(x, y) = p
    acc +. x *. y
  })
}

// --- helpers --------------------------------------------------------------

fn outlier_vec() -> List(Float) {
  [10.0, ..list.repeat(0.1, 15)]
}

// 256-dim, ~2.5% outliers (3x magnitude) — LLM-activation-like, deterministic.
fn outlier_vec_256() -> List(Float) {
  gen256(256, 777, [])
}

fn gen256(n: Int, seed: Int, acc: List(Float)) -> List(Float) {
  case n {
    0 -> acc
    _ -> {
      let next = { seed * 1_103_515_245 + 12_345 } % 2_147_483_648
      let base = int_to_float(next % 2000 - 1000) /. 1000.0
      let v = case n % 40 {
        0 -> base *. 3.0
        _ -> base *. 0.1
      }
      gen256(n - 1, next, [v, ..acc])
    }
  }
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
    True -> sqrtf(num) /. sqrtf(den)
    False -> 0.0
  }
}

fn uniform_rel_error(xs: List(Float), bits: Int) -> Float {
  let amax =
    list.fold(xs, 0.0, fn(m, x) { float.max(m, float.absolute_value(x)) })
  let levels = pow2(bits) - 1
  let step = 2.0 *. amax /. int_to_float(levels)
  let q = list.map(xs, fn(x) { step *. int_to_float(float.round(x /. step)) })
  rel_l2(xs, q)
}

fn sqrtf(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn pow2(b: Int) -> Int {
  case b {
    0 -> 1
    _ -> 2 * pow2(b - 1)
  }
}

@external(erlang, "erlang", "float")
fn int_to_float(x: Int) -> Float
