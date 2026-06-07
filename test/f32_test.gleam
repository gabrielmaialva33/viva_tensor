//// Tests for `viva_tensor/f32` — first-class FP32 tensors.
////
//// FP32 has ~7 significant digits, so exact integer-valued results (like the
//// 2x2 product below) are represented exactly and compared with `should.equal`.

import gleeunit
import gleeunit/should
import viva_tensor
import viva_tensor/f32

pub fn main() -> Nil {
  gleeunit.main()
}

// [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
pub fn matmul_2x2_test() {
  let assert Ok(a) = f32.from_floats([1.0, 2.0, 3.0, 4.0], [2, 2])
  let assert Ok(b) = f32.from_floats([5.0, 6.0, 7.0, 8.0], [2, 2])
  let assert Ok(c) = f32.matmul(a, b)

  f32.shape(c)
  |> should.equal([2, 2])

  let assert Ok(out) = f32.to_floats(c)
  out
  |> should.equal([19.0, 22.0, 43.0, 50.0])
}

pub fn fill_and_size_test() {
  let assert Ok(t) = f32.fill([3, 4], 1.5)
  f32.size(t)
  |> should.equal(Ok(12))
  f32.shape(t)
  |> should.equal([3, 4])
}

pub fn zeros_test() {
  let assert Ok(t) = f32.zeros([2, 2])
  let assert Ok(out) = f32.to_floats(t)
  out
  |> should.equal([0.0, 0.0, 0.0, 0.0])
}

pub fn matmul_shape_mismatch_test() {
  let assert Ok(a) = f32.fill([2, 3], 1.0)
  let assert Ok(b) = f32.fill([2, 2], 1.0)
  f32.matmul(a, b)
  |> should.be_error
}

// FP64 native tensor -> FP32 -> back to FP64 preserves exact half-representable
// values.
pub fn roundtrip_f64_test() {
  let assert Ok(t64) =
    viva_tensor.native_from_list([1.5, 2.5, 0.0, 4.0], [2, 2])
  let assert Ok(t32) = f32.from_tensor(t64)
  let assert Ok(back) = f32.to_tensor(t32)

  viva_tensor.to_list(back)
  |> should.equal([1.5, 2.5, 0.0, 4.0])
}
