import gleeunit/should
import viva_tensor/core/layout_math

pub fn compute_strides_uses_row_major_order_test() {
  layout_math.compute_strides([2, 3, 4]) |> should.equal([12, 4, 1])
  layout_math.compute_strides([5]) |> should.equal([1])
  layout_math.compute_strides([]) |> should.equal([])
}

pub fn flat_multi_roundtrip_test() {
  let shape = [2, 3, 4]
  let coordinates = layout_math.flat_to_multi(17, shape)

  coordinates |> should.equal([1, 1, 1])
  layout_math.multi_to_flat(coordinates, shape) |> should.equal(17)
}

pub fn broadcast_strides_marks_expanded_axes_with_zero_test() {
  layout_math.broadcast_strides([3], [1], [2, 3]) |> should.equal([0, 1])
  layout_math.broadcast_strides([2, 1], [1, 1], [2, 3])
  |> should.equal([1, 0])
}

pub fn list_helpers_handle_missing_indices_test() {
  layout_math.at([10, 20, 30], 1) |> should.equal(Ok(20))
  layout_math.at([10, 20, 30], -1) |> should.be_error()
  layout_math.dim_at([2, 3], 5) |> should.equal(0)
  layout_math.value_at([1.5], 3) |> should.equal(0.0)
}
