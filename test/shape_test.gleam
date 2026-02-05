import gleeunit
import gleeunit/should
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// SLICING OPERATIONS
// =============================================================================

pub fn slice_1d_test() {
  let t = tensor.from_list([10.0, 20.0, 30.0, 40.0, 50.0])
  case tensor.slice(t, [1], [3]) {
    Ok(s) -> {
      tensor.shape(s) |> should.equal([3])
      tensor.to_list(s) |> should.equal([20.0, 30.0, 40.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn slice_1d_start_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  case tensor.slice(t, [0], [2]) {
    Ok(s) -> tensor.to_list(s) |> should.equal([1.0, 2.0])
    Error(_) -> should.fail()
  }
}

pub fn slice_1d_end_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  case tensor.slice(t, [2], [2]) {
    Ok(s) -> tensor.to_list(s) |> should.equal([3.0, 4.0])
    Error(_) -> should.fail()
  }
}

pub fn slice_2d_test() {
  // [1, 2, 3]
  // [4, 5, 6]
  // [7, 8, 9]
  let t =
    tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape: [
      3,
      3,
    ])
  // Slice starting at (1,1) with size (2,2) -> [[5,6],[8,9]]
  case tensor.slice(t, [1, 1], [2, 2]) {
    Ok(s) -> {
      tensor.shape(s) |> should.equal([2, 2])
      tensor.to_list(s) |> should.equal([5.0, 6.0, 8.0, 9.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn slice_dimension_mismatch_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  // 1D tensor but 2D slice -> error
  tensor.slice(t, [0, 0], [1, 1]) |> should.be_error()
}

pub fn take_first_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let r = tensor.take_first(t, 3)
  tensor.shape(r) |> should.equal([3])
  tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0])
}

pub fn take_first_2d_test() {
  // [1, 2, 3]
  // [4, 5, 6]
  // [7, 8, 9]
  let t =
    tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape: [
      3,
      3,
    ])
  let r = tensor.take_first(t, 2)
  tensor.shape(r) |> should.equal([2, 3])
  tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

pub fn take_first_excess_test() {
  // Taking more than available should clamp
  let t = tensor.from_list([1.0, 2.0])
  let r = tensor.take_first(t, 10)
  tensor.to_list(r) |> should.equal([1.0, 2.0])
}

pub fn take_last_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let r = tensor.take_last(t, 2)
  tensor.shape(r) |> should.equal([2])
  tensor.to_list(r) |> should.equal([4.0, 5.0])
}

pub fn take_last_2d_test() {
  let t =
    tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape: [
      3,
      3,
    ])
  let r = tensor.take_last(t, 1)
  tensor.shape(r) |> should.equal([1, 3])
  tensor.to_list(r) |> should.equal([7.0, 8.0, 9.0])
}

// =============================================================================
// CONCATENATION
// =============================================================================

pub fn concat_test() {
  let a = tensor.from_list([1.0, 2.0])
  let b = tensor.from_list([3.0, 4.0, 5.0])
  let r = tensor.concat([a, b])
  tensor.shape(r) |> should.equal([5])
  tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0])
}

pub fn concat_single_test() {
  let a = tensor.from_list([1.0, 2.0])
  let r = tensor.concat([a])
  tensor.to_list(r) |> should.equal([1.0, 2.0])
}

pub fn concat_empty_vectors_test() {
  let a = tensor.from_list([])
  let b = tensor.from_list([1.0])
  let r = tensor.concat([a, b])
  tensor.to_list(r) |> should.equal([1.0])
}

pub fn concat_axis_0_test() {
  // [1, 2, 3] + [4, 5, 6] along axis 0 -> [2, 3]
  let a = tensor.Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
  let b = tensor.Tensor(data: [4.0, 5.0, 6.0], shape: [1, 3])
  case tensor.concat_axis([a, b], 0) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2, 3])
      tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn concat_axis_1_test() {
  // [[1, 2], [3, 4]] concat [[5], [6]] on axis=1 -> [[1, 2, 5], [3, 4, 6]]
  let a = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])
  let b = tensor.Tensor(data: [5.0, 6.0], shape: [2, 1])
  case tensor.concat_axis([a, b], 1) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2, 3])
      tensor.to_list(r) |> should.equal([1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn concat_axis_empty_test() {
  tensor.concat_axis([], 0) |> should.be_error()
}

pub fn concat_axis_single_test() {
  let a = tensor.Tensor(data: [1.0, 2.0], shape: [1, 2])
  case tensor.concat_axis([a], 0) {
    Ok(r) -> tensor.to_list(r) |> should.equal([1.0, 2.0])
    Error(_) -> should.fail()
  }
}

pub fn concat_axis_invalid_test() {
  let a = tensor.Tensor(data: [1.0, 2.0], shape: [1, 2])
  let b = tensor.Tensor(data: [3.0, 4.0], shape: [1, 2])
  // axis=5 out of bounds
  tensor.concat_axis([a, b], 5) |> should.be_error()
}

pub fn concat_axis_shape_mismatch_test() {
  let a = tensor.Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
  let b = tensor.Tensor(data: [1.0, 2.0], shape: [1, 2])
  // Non-concat dims must match (3 != 2)
  tensor.concat_axis([a, b], 0) |> should.be_error()
}

// =============================================================================
// STACKING
// =============================================================================

pub fn stack_axis_0_test() {
  // Stack [3] vectors along axis 0 -> [2, 3]
  let a = tensor.from_list([1.0, 2.0, 3.0])
  let b = tensor.from_list([4.0, 5.0, 6.0])
  case tensor.stack([a, b], 0) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2, 3])
      tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn stack_axis_1_test() {
  // Stack [3] vectors along axis 1 -> [3, 2]
  let a = tensor.from_list([1.0, 2.0, 3.0])
  let b = tensor.from_list([4.0, 5.0, 6.0])
  case tensor.stack([a, b], 1) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([3, 2])
      tensor.to_list(r) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn stack_empty_test() {
  tensor.stack([], 0) |> should.be_error()
}

pub fn stack_shape_mismatch_test() {
  let a = tensor.from_list([1.0, 2.0])
  let b = tensor.from_list([1.0, 2.0, 3.0])
  tensor.stack([a, b], 0) |> should.be_error()
}

// =============================================================================
// SQUEEZE_AXIS
// =============================================================================

pub fn squeeze_axis_test() {
  let t = tensor.Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
  case tensor.squeeze_axis(t, 0) {
    Ok(r) -> tensor.shape(r) |> should.equal([3])
    Error(_) -> should.fail()
  }
}

pub fn squeeze_axis_not_one_test() {
  let t = tensor.Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
  // axis 1 has size 3, not 1
  tensor.squeeze_axis(t, 1) |> should.be_error()
}

pub fn squeeze_axis_out_of_bounds_test() {
  let t = tensor.from_list([1.0, 2.0])
  tensor.squeeze_axis(t, 5) |> should.be_error()
}

// =============================================================================
// EXPAND_DIMS
// =============================================================================

pub fn expand_dims_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let r = tensor.expand_dims(t, 0)
  tensor.shape(r) |> should.equal([1, 3])
}

pub fn expand_dims_end_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let r = tensor.expand_dims(t, 1)
  tensor.shape(r) |> should.equal([3, 1])
}

// =============================================================================
// SUM_AXIS / MEAN_AXIS
// =============================================================================

pub fn sum_axis_0_test() {
  // [[1, 2, 3], [4, 5, 6]] sum along axis 0 -> [5, 7, 9]
  let t = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
  case tensor.sum_axis(t, 0) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([3])
      tensor.to_list(r) |> should.equal([5.0, 7.0, 9.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn sum_axis_1_test() {
  // [[1, 2, 3], [4, 5, 6]] sum along axis 1 -> [6, 15]
  let t = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
  case tensor.sum_axis(t, 1) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2])
      tensor.to_list(r) |> should.equal([6.0, 15.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn sum_axis_invalid_test() {
  let t = tensor.Tensor(data: [1.0, 2.0, 3.0], shape: [1, 3])
  tensor.sum_axis(t, 5) |> should.be_error()
}

pub fn mean_axis_test() {
  // [[2, 4], [6, 8]] mean along axis 0 -> [4, 6]
  let t = tensor.Tensor(data: [2.0, 4.0, 6.0, 8.0], shape: [2, 2])
  case tensor.mean_axis(t, 0) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2])
      tensor.to_list(r) |> should.equal([4.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}
