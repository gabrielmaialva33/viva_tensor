import gleam/int
import gleam/list
import gleam/string
import gleeunit/should
import viva_tensor as t

// =============================================================================
// 1D
// =============================================================================

pub fn repr_1d_int_test() {
  let x = t.from_list([1.0, 22.0, 333.0, 4.0, 55.0])
  t.to_string(x)
  |> should.equal("tensor([  1.,  22., 333.,   4.,  55.])")
}

pub fn repr_1d_float_fixed_test() {
  let x = t.from_list([1.0, 2.5, 3.0, 4.25])
  t.to_string(x)
  |> should.equal("tensor([1.0000, 2.5000, 3.0000, 4.2500])")
}

pub fn repr_1d_float_sci_test() {
  let x = t.from_list([0.00001, 1.0, 2.0, 3.0, 4.0])
  // min < 1e-4 forces sci mode
  t.to_string(x)
  |> should.equal(
    "tensor([1.0000e-05, 1.0000e+00, 2.0000e+00, 3.0000e+00, 4.0000e+00])",
  )
}

// =============================================================================
// 2D
// =============================================================================

pub fn repr_2d_test() {
  let assert Ok(x) = t.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  t.to_string(x)
  |> should.equal("tensor([[1., 2.],\n        [3., 4.],\n        [5., 6.]])")
}

// =============================================================================
// 3D
// =============================================================================

pub fn repr_3d_test() {
  let data =
    list.range(0, 23)
    |> list.map(int.to_float)
  let assert Ok(x) = t.reshape(t.from_list(data), [2, 3, 4])
  t.to_string(x)
  |> should.equal(
    "tensor([[[ 0.,  1.,  2.,  3.],\n         [ 4.,  5.,  6.,  7.],\n         [ 8.,  9., 10., 11.]],\n\n        [[12., 13., 14., 15.],\n         [16., 17., 18., 19.],\n         [20., 21., 22., 23.]]])",
  )
}

// =============================================================================
// Elision
// =============================================================================

pub fn repr_1d_elision_test() {
  // 1500 elements > threshold(1000); shape suffix appears, elision applied.
  let data =
    list.range(0, 1499)
    |> list.map(int.to_float)
  let s = t.to_string(t.from_list(data))
  s
  |> string.contains("...")
  |> should.be_true()
  s
  |> string.contains("shape=(1500,)")
  |> should.be_true()
}

// =============================================================================
// NaN / Inf
// =============================================================================

pub fn repr_handles_empty_test() {
  let x = t.from_list([])
  let s = t.to_string(x)
  s
  |> string.contains("shape=(0,)")
  |> should.be_true()
}

// =============================================================================
// PrintOptions
// =============================================================================

pub fn repr_default_precision_test() {
  let x = t.from_list([3.141592653589793, 2.718281828459045])
  t.to_string(x)
  |> should.equal("tensor([3.1416, 2.7183])")
}

// =============================================================================
// inspect alias
// =============================================================================

pub fn inspect_matches_to_string_test() {
  let x = t.from_list([1.0, 2.0, 3.0])
  t.inspect(x)
  |> should.equal(t.to_string(x))
}
