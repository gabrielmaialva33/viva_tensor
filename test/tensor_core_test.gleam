import gleam/float
import gleam/int
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor/core/tensor as core_tensor
import viva_tensor/tensor

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// HELPERS
// =============================================================================

fn assert_close(actual: Float, expected: Float, tol: Float) -> Bool {
  float.absolute_value(actual -. expected) <. tol
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

pub fn eye_3x3_test() {
  let e = core_tensor.eye(3)
  core_tensor.shape(e) |> should.equal([3, 3])
  core_tensor.to_list(e)
  |> should.equal([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
}

pub fn eye_1x1_test() {
  let e = core_tensor.eye(1)
  core_tensor.shape(e) |> should.equal([1, 1])
  core_tensor.to_list(e) |> should.equal([1.0])
}

pub fn eye_identity_matmul_test() {
  // A @ I = A (using facade tensor with manual identity)
  let e = tensor.Tensor(data: [1.0, 0.0, 0.0, 1.0], shape: [2, 2])
  let assert Ok(a) = tensor.matrix(2, 2, [3.0, 7.0, 1.0, 4.0])
  let assert Ok(r) = tensor.matmul(a, e)
  tensor.to_list(r) |> should.equal([3.0, 7.0, 1.0, 4.0])
}

pub fn arange_test() {
  let t = core_tensor.arange(0.0, 5.0, 1.0)
  core_tensor.to_list(t) |> should.equal([0.0, 1.0, 2.0, 3.0, 4.0])
}

pub fn arange_step_test() {
  let t = core_tensor.arange(0.0, 10.0, 2.5)
  core_tensor.to_list(t) |> should.equal([0.0, 2.5, 5.0, 7.5])
}

pub fn arange_empty_test() {
  // start >= end -> empty
  let t = core_tensor.arange(5.0, 3.0, 1.0)
  core_tensor.to_list(t) |> should.equal([])
}

pub fn linspace_test() {
  let t = core_tensor.linspace(0.0, 1.0, 5)
  let vals = core_tensor.to_list(t)
  // [0.0, 0.25, 0.5, 0.75, 1.0]
  list.length(vals) |> should.equal(5)
  let assert [first, ..] = vals
  assert_close(first, 0.0, 0.001) |> should.be_true()
  let assert Ok(last) = list.last(vals)
  assert_close(last, 1.0, 0.001) |> should.be_true()
}

pub fn linspace_two_test() {
  let t = core_tensor.linspace(0.0, 10.0, 2)
  core_tensor.to_list(t) |> should.equal([0.0, 10.0])
}

pub fn linspace_single_test() {
  let t = core_tensor.linspace(5.0, 10.0, 1)
  core_tensor.to_list(t) |> should.equal([5.0])
}

pub fn random_normal_shape_test() {
  let t = tensor.random_normal([3, 4], 0.0, 1.0)
  tensor.shape(t) |> should.equal([3, 4])
  tensor.size(t) |> should.equal(12)
}

pub fn random_normal_mean_test() {
  // With enough samples, mean should be close to target
  let t = tensor.random_normal([1000], 5.0, 1.0)
  let vals = tensor.to_list(t)
  let total = list.fold(vals, 0.0, fn(acc, x) { acc +. x })
  let m = total /. 1000.0
  // Mean should be roughly 5.0 (within ~0.2 for 1000 samples)
  { m >. 4.5 && m <. 5.5 } |> should.be_true()
}

pub fn he_init_shape_test() {
  let w = tensor.he_init(64, 32)
  tensor.shape(w) |> should.equal([32, 64])
  tensor.size(w) |> should.equal(2048)
}

pub fn he_init_std_test() {
  // He: std ~= sqrt(2/fan_in)
  let w = tensor.he_init(100, 50)
  let vals = tensor.to_list(w)
  let n = list.length(vals)
  let total = list.fold(vals, 0.0, fn(acc, x) { acc +. x })
  let m = total /. int.to_float(n)
  let var =
    list.fold(vals, 0.0, fn(acc, x) {
      let d = x -. m
      acc +. d *. d
    })
    /. int.to_float(n)
  let std_val = float_sqrt(var)
  // Expected std ~= sqrt(2/100) = sqrt(0.02) ~= 0.1414
  { std_val >. 0.05 && std_val <. 0.25 } |> should.be_true()
}

pub fn matrix_invalid_size_test() {
  // 2x2 matrix with 3 elements -> error
  tensor.matrix(2, 2, [1.0, 2.0, 3.0]) |> should.be_error()
}

pub fn from_list2d_jagged_test() {
  // Jagged rows -> error
  tensor.from_list2d([[1.0, 2.0], [3.0]]) |> should.be_error()
}

pub fn from_list2d_empty_test() {
  case tensor.from_list2d([]) {
    Ok(t) -> tensor.shape(t) |> should.equal([0, 0])
    Error(_) -> should.fail()
  }
}

// =============================================================================
// ELEMENT ACCESS
// =============================================================================

pub fn get_test() {
  let t = tensor.from_list([10.0, 20.0, 30.0])
  case tensor.get(t, 0) {
    Ok(v) -> v |> should.equal(10.0)
    Error(_) -> should.fail()
  }
  case tensor.get(t, 2) {
    Ok(v) -> v |> should.equal(30.0)
    Error(_) -> should.fail()
  }
}

pub fn get_negative_index_test() {
  let t = tensor.from_list([1.0, 2.0])
  tensor.get(t, -1) |> should.be_error()
}

pub fn get2d_test() {
  // [[1, 2, 3],
  //  [4, 5, 6]]
  case tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      case tensor.get2d(m, 0, 0) {
        Ok(v) -> v |> should.equal(1.0)
        Error(_) -> should.fail()
      }
      case tensor.get2d(m, 0, 2) {
        Ok(v) -> v |> should.equal(3.0)
        Error(_) -> should.fail()
      }
      case tensor.get2d(m, 1, 1) {
        Ok(v) -> v |> should.equal(5.0)
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn get2d_1d_error_test() {
  let t = tensor.from_list([1.0, 2.0])
  tensor.get2d(t, 0, 0) |> should.be_error()
}

pub fn get_row_test() {
  case tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      case tensor.get_row(m, 0) {
        Ok(row) -> tensor.to_list(row) |> should.equal([1.0, 2.0, 3.0])
        Error(_) -> should.fail()
      }
      case tensor.get_row(m, 1) {
        Ok(row) -> tensor.to_list(row) |> should.equal([4.0, 5.0, 6.0])
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn get_row_out_of_bounds_test() {
  let assert Ok(m) = tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  tensor.get_row(m, 5) |> should.be_error()
}

pub fn get_col_test() {
  // [[1, 2],
  //  [3, 4],
  //  [5, 6]]
  case tensor.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      case tensor.get_col(m, 0) {
        Ok(col) -> tensor.to_list(col) |> should.equal([1.0, 3.0, 5.0])
        Error(_) -> should.fail()
      }
      case tensor.get_col(m, 1) {
        Ok(col) -> tensor.to_list(col) |> should.equal([2.0, 4.0, 6.0])
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn get_col_out_of_bounds_test() {
  let assert Ok(m) = tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  tensor.get_col(m, 5) |> should.be_error()
}

pub fn dim_test() {
  let t = tensor.Tensor(data: list.repeat(0.0, 24), shape: [2, 3, 4])
  case tensor.dim(t, 0) {
    Ok(d) -> d |> should.equal(2)
    Error(_) -> should.fail()
  }
  case tensor.dim(t, 1) {
    Ok(d) -> d |> should.equal(3)
    Error(_) -> should.fail()
  }
  case tensor.dim(t, 2) {
    Ok(d) -> d |> should.equal(4)
    Error(_) -> should.fail()
  }
}

pub fn dim_out_of_bounds_test() {
  let t = tensor.from_list([1.0, 2.0])
  tensor.dim(t, 3) |> should.be_error()
}

pub fn rows_test() {
  case tensor.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> tensor.rows(m) |> should.equal(3)
    Error(_) -> should.fail()
  }
}

pub fn cols_test() {
  case tensor.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> tensor.cols(m) |> should.equal(2)
    Error(_) -> should.fail()
  }
}

// =============================================================================
// ADDITIONAL OPERATIONS
// =============================================================================

pub fn add_scalar_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let r = tensor.add_scalar(t, 10.0)
  tensor.to_list(r) |> should.equal([11.0, 12.0, 13.0])
}

pub fn negate_test() {
  let t = tensor.from_list([1.0, -2.0, 3.0])
  let r = tensor.negate(t)
  tensor.to_list(r) |> should.equal([-1.0, 2.0, -3.0])
}

pub fn product_test() {
  let t = tensor.from_list([2.0, 3.0, 4.0])
  tensor.product(t) |> should.equal(24.0)
}

pub fn clone_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  let c = tensor.clone(t)
  tensor.to_list(c) |> should.equal([1.0, 2.0, 3.0])
  tensor.shape(c) |> should.equal([3])
}

pub fn to_list2d_test() {
  case tensor.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      case tensor.to_list2d(m) {
        Ok(rows) -> rows |> should.equal([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn to_list2d_1d_error_test() {
  let t = tensor.from_list([1.0, 2.0])
  tensor.to_list2d(t) |> should.be_error()
}

pub fn map_indexed_test() {
  let t = tensor.from_list([10.0, 20.0, 30.0])
  let r = tensor.map_indexed(t, fn(x, i) { x +. int.to_float(i) })
  tensor.to_list(r) |> should.equal([10.0, 21.0, 32.0])
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

pub fn outer_product_test() {
  let a = tensor.from_list([1.0, 2.0, 3.0])
  let b = tensor.from_list([4.0, 5.0])
  case tensor.outer(a, b) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([3, 2])
      tensor.to_list(r)
      |> should.equal([4.0, 5.0, 8.0, 10.0, 12.0, 15.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn matmul_vec_test() {
  // [[1, 0], [0, 1]] @ [3, 7] = [3, 7] (identity)
  let e = tensor.Tensor(data: [1.0, 0.0, 0.0, 1.0], shape: [2, 2])
  let v = tensor.from_list([3.0, 7.0])
  case tensor.matmul_vec(e, v) {
    Ok(r) -> tensor.to_list(r) |> should.equal([3.0, 7.0])
    Error(_) -> should.fail()
  }
}

pub fn norm_test() {
  // norm([3, 4]) = 5
  let t = tensor.from_list([3.0, 4.0])
  assert_close(tensor.norm(t), 5.0, 0.0001) |> should.be_true()
}

pub fn normalize_test() {
  let t = tensor.from_list([3.0, 4.0])
  let r = tensor.normalize(t)
  let n = tensor.norm(r)
  assert_close(n, 1.0, 0.001) |> should.be_true()
}

pub fn clamp_test() {
  let t = tensor.from_list([-10.0, 0.5, 5.0, 100.0])
  let r = tensor.clamp(t, 0.0, 1.0)
  tensor.to_list(r) |> should.equal([0.0, 0.5, 1.0, 1.0])
}

pub fn variance_test() {
  let t = tensor.from_list([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  let v = tensor.variance(t)
  assert_close(v, 4.0, 0.001) |> should.be_true()
}

pub fn std_test() {
  let t = tensor.from_list([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
  let s = tensor.std(t)
  assert_close(s, 2.0, 0.001) |> should.be_true()
}

// =============================================================================
// RESHAPE ERROR CASES
// =============================================================================

pub fn reshape_size_mismatch_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0])
  // [3] -> [2, 2] = 4 elements, mismatch
  tensor.reshape(t, [2, 2]) |> should.be_error()
}

pub fn reshape_valid_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  case tensor.reshape(t, [2, 3]) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2, 3])
      tensor.to_list(r) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn reshape_3d_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  case tensor.reshape(t, [1, 2, 3]) {
    Ok(r) -> tensor.shape(r) |> should.equal([1, 2, 3])
    Error(_) -> should.fail()
  }
}

// =============================================================================
// BROADCASTING
// =============================================================================

pub fn mul_broadcast_test() {
  let a = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])
  let b = tensor.from_list([10.0])
  case tensor.mul_broadcast(a, b) {
    Ok(r) -> {
      tensor.shape(r) |> should.equal([2, 2])
      tensor.to_list(r) |> should.equal([10.0, 20.0, 30.0, 40.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shape_test() {
  case tensor.broadcast_shape([2, 1], [1, 3]) {
    Ok(s) -> s |> should.equal([2, 3])
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shape_incompatible_test() {
  tensor.broadcast_shape([2, 3], [4, 5]) |> should.be_error()
}

// =============================================================================
// STRIDED OPERATIONS (extended)
// =============================================================================

pub fn strided_roundtrip_test() {
  let t = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let s = tensor.to_strided(t)
  let d = tensor.to_contiguous(s)
  tensor.to_list(d) |> should.equal([1.0, 2.0, 3.0, 4.0])
}

pub fn strided_is_contiguous_test() {
  let t = tensor.from_list([1.0, 2.0])
  // Dense tensor is always contiguous
  tensor.is_contiguous(t) |> should.be_true()
  // Freshly strided is contiguous
  let s = tensor.to_strided(t)
  tensor.is_contiguous(s) |> should.be_true()
}

pub fn get_fast_test() {
  let t = tensor.from_list([10.0, 20.0, 30.0])
  let s = tensor.to_strided(t)
  case tensor.get_fast(s, 1) {
    Ok(v) -> v |> should.equal(20.0)
    Error(_) -> should.fail()
  }
}
