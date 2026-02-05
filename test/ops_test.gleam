import gleam/float
import gleam/int
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor/core/ops
import viva_tensor/core/tensor as core_tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// HELPERS
// =============================================================================

fn assert_close(actual: Float, expected: Float, tol: Float) -> Bool {
  float.absolute_value(actual -. expected) <. tol
}

fn assert_list_close(
  actual: List(Float),
  expected: List(Float),
  tol: Float,
) -> Bool {
  list.length(actual) == list.length(expected)
  && list.zip(actual, expected)
  |> list.all(fn(pair) {
    let #(a, e) = pair
    assert_close(a, e, tol)
  })
}

// =============================================================================
// ACTIVATION FUNCTIONS
// =============================================================================

pub fn relu_positive_test() {
  let t = core_tensor.from_list([1.0, 2.0, 3.0])
  let r = ops.relu(t)
  core_tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0])
}

pub fn relu_negative_test() {
  let t = core_tensor.from_list([-1.0, -2.0, -3.0])
  let r = ops.relu(t)
  core_tensor.to_list(r) |> should.equal([0.0, 0.0, 0.0])
}

pub fn relu_mixed_test() {
  let t = core_tensor.from_list([-2.0, 0.0, 3.0, -1.0, 5.0])
  let r = ops.relu(t)
  core_tensor.to_list(r) |> should.equal([0.0, 0.0, 3.0, 0.0, 5.0])
}

pub fn sigmoid_zero_test() {
  let t = core_tensor.from_list([0.0])
  let r = ops.sigmoid(t)
  let vals = core_tensor.to_list(r)
  // sigmoid(0) = 0.5
  assert_list_close(vals, [0.5], 0.0001) |> should.be_true()
}

pub fn sigmoid_large_positive_test() {
  let t = core_tensor.from_list([10.0])
  let r = ops.sigmoid(t)
  let vals = core_tensor.to_list(r)
  // sigmoid(10) ~= 0.99995
  let assert [v] = vals
  { v >. 0.999 } |> should.be_true()
}

pub fn sigmoid_large_negative_test() {
  let t = core_tensor.from_list([-10.0])
  let r = ops.sigmoid(t)
  let vals = core_tensor.to_list(r)
  // sigmoid(-10) ~= 0.00005
  let assert [v] = vals
  { v <. 0.001 } |> should.be_true()
}

pub fn sigmoid_range_test() {
  // All sigmoid outputs must be in (0, 1)
  let t = core_tensor.from_list([-5.0, -1.0, 0.0, 1.0, 5.0])
  let r = ops.sigmoid(t)
  let vals = core_tensor.to_list(r)
  vals
  |> list.all(fn(v) { v >. 0.0 && v <. 1.0 })
  |> should.be_true()
}

pub fn tanh_zero_test() {
  let t = core_tensor.from_list([0.0])
  let r = ops.tanh(t)
  let vals = core_tensor.to_list(r)
  assert_list_close(vals, [0.0], 0.0001) |> should.be_true()
}

pub fn tanh_range_test() {
  // tanh outputs in (-1, 1)
  let t = core_tensor.from_list([-5.0, -1.0, 0.0, 1.0, 5.0])
  let r = ops.tanh(t)
  let vals = core_tensor.to_list(r)
  vals
  |> list.all(fn(v) { v >. -1.0 && v <. 1.0 })
  |> should.be_true()
}

pub fn tanh_symmetry_test() {
  // tanh(-x) = -tanh(x)
  let t_pos = core_tensor.from_list([1.0, 2.0, 3.0])
  let t_neg = core_tensor.from_list([-1.0, -2.0, -3.0])
  let r_pos = core_tensor.to_list(ops.tanh(t_pos))
  let r_neg = core_tensor.to_list(ops.tanh(t_neg))
  let negated_pos = list.map(r_pos, fn(x) { 0.0 -. x })
  assert_list_close(r_neg, negated_pos, 0.0001) |> should.be_true()
}

pub fn softmax_sum_to_one_test() {
  let t = core_tensor.from_list([1.0, 2.0, 3.0])
  let r = ops.softmax(t)
  let vals = core_tensor.to_list(r)
  let total = list.fold(vals, 0.0, fn(acc, x) { acc +. x })
  assert_close(total, 1.0, 0.0001) |> should.be_true()
}

pub fn softmax_positive_test() {
  // All softmax outputs must be positive
  let t = core_tensor.from_list([-1.0, 0.0, 1.0])
  let r = ops.softmax(t)
  let vals = core_tensor.to_list(r)
  vals |> list.all(fn(v) { v >. 0.0 }) |> should.be_true()
}

pub fn softmax_ordering_test() {
  // Larger input -> larger probability
  let t = core_tensor.from_list([1.0, 2.0, 3.0])
  let r = ops.softmax(t)
  let vals = core_tensor.to_list(r)
  let assert [a, b, c] = vals
  { a <. b && b <. c } |> should.be_true()
}

pub fn softmax_uniform_test() {
  // Equal inputs -> equal outputs
  let t = core_tensor.from_list([1.0, 1.0, 1.0])
  let r = ops.softmax(t)
  let vals = core_tensor.to_list(r)
  assert_list_close(vals, [0.333333, 0.333333, 0.333333], 0.001)
  |> should.be_true()
}

// =============================================================================
// ELEMENT-WISE MATH OPERATIONS
// =============================================================================

pub fn negate_test() {
  let t = core_tensor.from_list([1.0, -2.0, 3.0])
  let r = ops.negate(t)
  core_tensor.to_list(r) |> should.equal([-1.0, 2.0, -3.0])
}

pub fn abs_test() {
  let t = core_tensor.from_list([-3.0, -1.0, 0.0, 2.0, 4.0])
  let r = ops.abs(t)
  core_tensor.to_list(r) |> should.equal([3.0, 1.0, 0.0, 2.0, 4.0])
}

pub fn square_test() {
  let t = core_tensor.from_list([1.0, 2.0, 3.0, -2.0])
  let r = ops.square(t)
  core_tensor.to_list(r) |> should.equal([1.0, 4.0, 9.0, 4.0])
}

pub fn sqrt_test() {
  let t = core_tensor.from_list([1.0, 4.0, 9.0, 16.0])
  let r = ops.sqrt(t)
  core_tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 4.0])
}

pub fn exp_test() {
  let t = core_tensor.from_list([0.0, 1.0])
  let r = ops.exp(t)
  let vals = core_tensor.to_list(r)
  // exp(0) = 1, exp(1) ~= 2.71828
  assert_list_close(vals, [1.0, 2.71828], 0.001) |> should.be_true()
}

pub fn log_test() {
  let t = core_tensor.from_list([1.0, 2.71828])
  let r = ops.log(t)
  let vals = core_tensor.to_list(r)
  // log(1) = 0, log(e) ~= 1
  assert_list_close(vals, [0.0, 1.0], 0.001) |> should.be_true()
}

pub fn exp_log_inverse_test() {
  // exp(log(x)) = x for positive x
  let t = core_tensor.from_list([0.5, 1.0, 2.0, 10.0])
  let r = ops.exp(ops.log(t))
  let vals = core_tensor.to_list(r)
  assert_list_close(vals, [0.5, 1.0, 2.0, 10.0], 0.0001) |> should.be_true()
}

pub fn pow_test() {
  let t = core_tensor.from_list([2.0, 3.0, 4.0])
  let r = ops.pow(t, 2.0)
  core_tensor.to_list(r) |> should.equal([4.0, 9.0, 16.0])
}

pub fn pow_zero_test() {
  let t = core_tensor.from_list([2.0, 3.0, 100.0])
  let r = ops.pow(t, 0.0)
  core_tensor.to_list(r) |> should.equal([1.0, 1.0, 1.0])
}

pub fn add_scalar_test() {
  let t = core_tensor.from_list([1.0, 2.0, 3.0])
  let r = ops.add_scalar(t, 10.0)
  core_tensor.to_list(r) |> should.equal([11.0, 12.0, 13.0])
}

pub fn map_indexed_test() {
  let t = core_tensor.from_list([10.0, 20.0, 30.0])
  let r = ops.map_indexed(t, fn(x, i) { x +. int.to_float(i) })
  core_tensor.to_list(r) |> should.equal([10.0, 21.0, 32.0])
}

pub fn clamp_test() {
  let t = core_tensor.from_list([-5.0, 0.0, 3.0, 10.0, 20.0])
  let r = ops.clamp(t, 0.0, 10.0)
  core_tensor.to_list(r) |> should.equal([0.0, 0.0, 3.0, 10.0, 10.0])
}

// =============================================================================
// EXTENDED REDUCTION OPERATIONS
// =============================================================================

pub fn variance_test() {
  // [1, 2, 3, 4] mean=2.5, variance = ((1-2.5)^2 + ... + (4-2.5)^2)/4 = 1.25
  let t = core_tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let v = ops.variance(t)
  assert_close(v, 1.25, 0.0001) |> should.be_true()
}

pub fn variance_constant_test() {
  // Constant values -> variance = 0
  let t = core_tensor.from_list([5.0, 5.0, 5.0])
  let v = ops.variance(t)
  assert_close(v, 0.0, 0.0001) |> should.be_true()
}

pub fn std_test() {
  // std = sqrt(variance) = sqrt(1.25) ~= 1.118
  let t = core_tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let s = ops.std(t)
  assert_close(s, 1.118, 0.01) |> should.be_true()
}

pub fn product_test() {
  let t = core_tensor.from_list([2.0, 3.0, 4.0])
  let p = ops.product(t)
  p |> should.equal(24.0)
}

pub fn product_with_zero_test() {
  let t = core_tensor.from_list([2.0, 0.0, 4.0])
  let p = ops.product(t)
  p |> should.equal(0.0)
}

pub fn norm_test() {
  // norm([3, 4]) = sqrt(9 + 16) = 5
  let t = core_tensor.from_list([3.0, 4.0])
  let n = ops.norm(t)
  assert_close(n, 5.0, 0.0001) |> should.be_true()
}

pub fn normalize_test() {
  let t = core_tensor.from_list([3.0, 4.0])
  let r = ops.normalize(t)
  let vals = core_tensor.to_list(r)
  // [3/5, 4/5] = [0.6, 0.8]
  assert_list_close(vals, [0.6, 0.8], 0.0001) |> should.be_true()
}

pub fn normalize_unit_norm_test() {
  // After normalize, norm should be ~1.0
  let t = core_tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let r = ops.normalize(t)
  let n = ops.norm(r)
  assert_close(n, 1.0, 0.001) |> should.be_true()
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

pub fn outer_product_test() {
  let a = core_tensor.from_list([1.0, 2.0])
  let b = core_tensor.from_list([3.0, 4.0, 5.0])
  case ops.outer(a, b) {
    Ok(r) -> {
      core_tensor.shape(r) |> should.equal([2, 3])
      // [1*3, 1*4, 1*5, 2*3, 2*4, 2*5]
      core_tensor.to_list(r)
      |> should.equal([3.0, 4.0, 5.0, 6.0, 8.0, 10.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn matmul_vec_test() {
  // [1, 2]   [5]   [1*5+2*6]   [17]
  // [3, 4] x [6] = [3*5+4*6] = [39]
  case core_tensor.matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0]) {
    Ok(mat) -> {
      let vec = core_tensor.from_list([5.0, 6.0])
      case ops.matmul_vec(mat, vec) {
        Ok(r) -> {
          core_tensor.shape(r) |> should.equal([2])
          core_tensor.to_list(r) |> should.equal([17.0, 39.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// BROADCASTING OPERATIONS
// =============================================================================

pub fn broadcast_shape_test() {
  case ops.broadcast_shape([2, 3], [3]) {
    Ok(s) -> s |> should.equal([2, 3])
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shape_scalar_test() {
  case ops.broadcast_shape([2, 3, 4], [1]) {
    Ok(s) -> s |> should.equal([2, 3, 4])
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shape_error_test() {
  case ops.broadcast_shape([2, 3], [4]) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn mul_broadcast_test() {
  let a = core_tensor.from_list([1.0, 2.0, 3.0])
  let b = core_tensor.from_list([2.0])
  case ops.mul_broadcast(a, b) {
    Ok(r) -> {
      core_tensor.to_list(r) |> should.equal([2.0, 4.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn broadcast_to_test() {
  let t = core_tensor.from_list([1.0, 2.0, 3.0])
  case ops.broadcast_to(t, [2, 3]) {
    Ok(r) -> {
      core_tensor.shape(r) |> should.equal([2, 3])
      core_tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// ERROR CASES
// =============================================================================

pub fn add_shape_mismatch_test() {
  let a = core_tensor.from_list([1.0, 2.0])
  let b = core_tensor.from_list([1.0, 2.0, 3.0])
  ops.add(a, b) |> should.be_error()
}

pub fn sub_shape_mismatch_test() {
  let a = core_tensor.from_list([1.0, 2.0])
  let b = core_tensor.from_list([1.0, 2.0, 3.0])
  ops.sub(a, b) |> should.be_error()
}

pub fn mul_shape_mismatch_test() {
  let a = core_tensor.from_list([1.0, 2.0])
  let b = core_tensor.from_list([1.0, 2.0, 3.0])
  ops.mul(a, b) |> should.be_error()
}

pub fn div_shape_mismatch_test() {
  let a = core_tensor.from_list([1.0, 2.0])
  let b = core_tensor.from_list([1.0, 2.0, 3.0])
  ops.div(a, b) |> should.be_error()
}

pub fn dot_shape_mismatch_test() {
  let a = core_tensor.from_list([1.0, 2.0])
  let b = core_tensor.from_list([1.0, 2.0, 3.0])
  ops.dot(a, b) |> should.be_error()
}

pub fn matmul_shape_mismatch_test() {
  // [2,2] x [3,3] -> incompatible inner dimensions
  let assert Ok(a) =
    core_tensor.matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
  let assert Ok(b) =
    core_tensor.matrix(rows: 3, cols: 3, data: [
      1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ])
  ops.matmul(a, b) |> should.be_error()
}

pub fn matmul_vec_mismatch_test() {
  // [2,3] @ [2] -> mismatch (need [3])
  let assert Ok(mat) =
    core_tensor.matrix(rows: 2, cols: 3, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let vec = core_tensor.from_list([1.0, 2.0])
  ops.matmul_vec(mat, vec) |> should.be_error()
}

pub fn outer_non_vector_test() {
  // outer requires 1D tensors
  let assert Ok(mat) =
    core_tensor.matrix(rows: 2, cols: 2, data: [1.0, 2.0, 3.0, 4.0])
  let vec = core_tensor.from_list([1.0])
  ops.outer(mat, vec) |> should.be_error()
}

pub fn transpose_non_2d_test() {
  let t = core_tensor.from_list([1.0, 2.0, 3.0])
  ops.transpose(t) |> should.be_error()
}
