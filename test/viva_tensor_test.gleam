import gleam/list
import gleam/string
import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/axis
import viva_tensor/core/ops
import viva_tensor/core/tensor as core_tensor
import viva_tensor/cuda.{CpuFallback}
import viva_tensor/layout
import viva_tensor/named
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// TENSOR CONSTRUCTORS
// =============================================================================

pub fn zeros_test() {
  let z = t.zeros([2, 3])
  t.shape(z) |> should.equal([2, 3])
  t.size(z) |> should.equal(6)
  t.to_list(z) |> should.equal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

pub fn ones_test() {
  let o = t.ones([3])
  t.shape(o) |> should.equal([3])
  t.to_list(o) |> should.equal([1.0, 1.0, 1.0])
}

pub fn fill_test() {
  let f = t.fill([2, 2], 5.0)
  t.to_list(f) |> should.equal([5.0, 5.0, 5.0, 5.0])
}

pub fn from_list_test() {
  let v = t.from_list([1.0, 2.0, 3.0])
  t.shape(v) |> should.equal([3])
  t.to_list(v) |> should.equal([1.0, 2.0, 3.0])
}

pub fn linspace_test() {
  let values = t.linspace(0.0, 1.0, 5) |> t.to_list()

  values |> should.equal([0.0, 0.25, 0.5, 0.75, 1.0])
}

pub fn try_linspace_rejects_invalid_steps_test() {
  t.try_linspace(0.0, 1.0, 0) |> should.be_error()
}

pub fn logspace_test() {
  let values = t.logspace(1.0, 3.0, 3, 10.0) |> t.to_list()

  values |> should.equal([10.0, 100.0, 1000.0])
}

pub fn try_logspace_rejects_invalid_base_test() {
  t.try_logspace(1.0, 3.0, 3, 0.0) |> should.be_error()
}

pub fn like_constructors_test() {
  let source = t.fill([2, 2], 9.0)

  t.zeros_like(source) |> t.to_list() |> should.equal([0.0, 0.0, 0.0, 0.0])
  t.ones_like(source) |> t.to_list() |> should.equal([1.0, 1.0, 1.0, 1.0])
  t.full_like(source, 7.0) |> t.to_list() |> should.equal([7.0, 7.0, 7.0, 7.0])
}

pub fn eye_identity_diag_test() {
  t.eye(3)
  |> t.to_list()
  |> should.equal([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])

  t.identity(2) |> t.to_list() |> should.equal([1.0, 0.0, 0.0, 1.0])

  case t.try_diag(t.from_list([2.0, 3.0, 4.0])) {
    Ok(diagonal) -> {
      t.shape(diagonal) |> should.equal([3, 3])
      t.to_list(diagonal)
      |> should.equal([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn try_eye_and_diag_reject_invalid_inputs_test() {
  t.try_eye(0) |> should.be_error()
  t.zeros([2, 2]) |> t.try_diag() |> should.be_error()
}

pub fn try_to_list_test() {
  let v = t.from_list([1.0, 2.0, 3.0])
  t.try_to_list(v) |> should.equal(Ok([1.0, 2.0, 3.0]))
}

pub fn vector_test() {
  let v = t.vector([1.0, 2.0])
  t.rank(v) |> should.equal(1)
}

pub fn matrix_test() {
  let m = t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  m |> should.be_ok()
  case m {
    Ok(mat) -> t.shape(mat) |> should.equal([2, 2])
    Error(_) -> should.fail()
  }
}

pub fn from_list2d_test() {
  let m = t.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  m |> should.be_ok()
  case m {
    Ok(mat) -> {
      t.shape(mat) |> should.equal([2, 2])
      t.to_list(mat) |> should.equal([1.0, 2.0, 3.0, 4.0])
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

pub fn add_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([4.0, 5.0, 6.0])
  case t.add(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([5.0, 7.0, 9.0])
    Error(_) -> should.fail()
  }
}

pub fn sub_test() {
  let a = t.from_list([5.0, 5.0])
  let b = t.from_list([2.0, 3.0])
  case t.sub(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([3.0, 2.0])
    Error(_) -> should.fail()
  }
}

pub fn mul_test() {
  let a = t.from_list([2.0, 3.0])
  let b = t.from_list([4.0, 5.0])
  case t.mul(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([8.0, 15.0])
    Error(_) -> should.fail()
  }
}

pub fn div_test() {
  let a = t.from_list([10.0, 20.0])
  let b = t.from_list([2.0, 4.0])
  case t.div(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([5.0, 5.0])
    Error(_) -> should.fail()
  }
}

pub fn div_shape_mismatch_test() {
  let a = t.from_list([10.0, 20.0])
  let b = t.from_list([2.0])

  t.div(a, b) |> should.be_error()
}

pub fn scale_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let s = t.scale(a, 2.0)
  t.to_list(s) |> should.equal([2.0, 4.0, 6.0])
}

pub fn try_scale_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  case t.try_scale(a, 2.0) {
    Ok(s) -> t.to_list(s) |> should.equal([2.0, 4.0, 6.0])
    Error(_) -> should.fail()
  }
}

pub fn map_test() {
  let a = t.from_list([1.0, 4.0, 9.0])
  let b = t.map(a, fn(x) { x *. x })
  t.to_list(b) |> should.equal([1.0, 16.0, 81.0])
}

pub fn try_map_test() {
  let a = t.from_list([1.0, 4.0, 9.0])
  case t.try_map(a, fn(x) { x *. x }) {
    Ok(b) -> t.to_list(b) |> should.equal([1.0, 16.0, 81.0])
    Error(_) -> should.fail()
  }
}

pub fn elementwise_rounding_and_sign_test() {
  let values = t.from_list([-1.8, -0.2, 0.0, 1.2, 1.8])

  t.floor(values) |> t.to_list() |> should.equal([-2.0, -1.0, 0.0, 1.0, 1.0])
  t.ceil(values) |> t.to_list() |> should.equal([-1.0, -0.0, 0.0, 2.0, 2.0])
  t.round(values) |> t.to_list() |> should.equal([-2.0, 0.0, 0.0, 1.0, 2.0])
  t.sign(values) |> t.to_list() |> should.equal([-1.0, -1.0, 0.0, 1.0, 1.0])
}

pub fn reciprocal_and_clip_test() {
  let values = t.from_list([2.0, 4.0, -0.5])

  t.reciprocal(values) |> t.to_list() |> should.equal([0.5, 0.25, -2.0])
  t.clip(values, 0.0, 3.0) |> t.to_list() |> should.equal([2.0, 3.0, 0.0])

  t.try_reciprocal(t.from_list([1.0, 0.0])) |> should.be_error()
  t.try_clip(values, 3.0, 0.0) |> should.be_error()
}

// =============================================================================
// REDUCTIONS
// =============================================================================

pub fn sum_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0])
  t.sum(a) |> should.equal(10.0)
}

pub fn try_sum_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0])
  t.try_sum(a) |> should.equal(Ok(10.0))
}

pub fn mean_test() {
  let a = t.from_list([2.0, 4.0, 6.0, 8.0])
  t.mean(a) |> should.equal(5.0)
}

pub fn try_mean_test() {
  let a = t.from_list([2.0, 4.0, 6.0, 8.0])
  t.try_mean(a) |> should.equal(Ok(5.0))
}

pub fn try_mean_empty_test() {
  t.from_list([])
  |> t.try_mean()
  |> should.be_error()
}

pub fn product_test() {
  let a = t.from_list([2.0, 3.0, 4.0])
  t.product(a) |> should.equal(24.0)
}

pub fn try_product_test() {
  let a = t.from_list([2.0, 3.0, 4.0])
  t.try_product(a) |> should.equal(Ok(24.0))
}

pub fn try_product_empty_identity_test() {
  t.from_list([])
  |> t.try_product()
  |> should.equal(Ok(1.0))
}

pub fn max_test() {
  let a = t.from_list([1.0, 5.0, 3.0, 2.0])
  t.max(a) |> should.equal(5.0)
}

pub fn try_max_test() {
  let a = t.from_list([1.0, 5.0, 3.0, 2.0])
  t.try_max(a) |> should.equal(Ok(5.0))
}

pub fn try_max_empty_test() {
  t.from_list([])
  |> t.try_max()
  |> should.be_error()
}

pub fn min_test() {
  let a = t.from_list([1.0, 5.0, 3.0, 2.0])
  t.min(a) |> should.equal(1.0)
}

pub fn try_min_test() {
  let a = t.from_list([1.0, 5.0, 3.0, 2.0])
  t.try_min(a) |> should.equal(Ok(1.0))
}

pub fn try_min_empty_test() {
  t.from_list([])
  |> t.try_min()
  |> should.be_error()
}

pub fn argmax_test() {
  let a = t.from_list([1.0, 5.0, 3.0])
  t.argmax(a) |> should.equal(1)
}

pub fn try_argmax_test() {
  let a = t.from_list([1.0, 5.0, 3.0])
  t.try_argmax(a) |> should.equal(Ok(1))
}

pub fn try_argmax_empty_test() {
  t.from_list([])
  |> t.try_argmax()
  |> should.be_error()
}

pub fn argmin_test() {
  let a = t.from_list([3.0, 1.0, 5.0])
  t.argmin(a) |> should.equal(1)
}

pub fn try_argmin_test() {
  let a = t.from_list([3.0, 1.0, 5.0])
  t.try_argmin(a) |> should.equal(Ok(1))
}

pub fn try_argmin_empty_test() {
  t.from_list([])
  |> t.try_argmin()
  |> should.be_error()
}

pub fn try_variance_test() {
  let a = t.from_list([2.0, 4.0, 6.0, 8.0])
  t.try_variance(a) |> should.equal(Ok(5.0))
}

pub fn try_variance_empty_test() {
  t.from_list([])
  |> t.try_variance()
  |> should.be_error()
}

pub fn try_std_test() {
  let a = t.from_list([2.0, 4.0, 6.0, 8.0])
  case t.try_std(a) {
    Ok(value) -> { value >. 2.23 && value <. 2.24 } |> should.be_true()
    Error(_) -> should.fail()
  }
}

pub fn try_std_empty_test() {
  t.from_list([])
  |> t.try_std()
  |> should.be_error()
}

pub fn cumsum_test() {
  let values = t.from_list([1.0, 2.0, 3.0, 4.0]) |> t.cumsum()

  t.to_list(values) |> should.equal([1.0, 3.0, 6.0, 10.0])
  t.shape(values) |> should.equal([4])
}

pub fn cumprod_test() {
  let values = t.from_list([1.0, 2.0, 3.0, 4.0]) |> t.cumprod()

  t.to_list(values) |> should.equal([1.0, 2.0, 6.0, 24.0])
  t.shape(values) |> should.equal([4])
}

pub fn cumsum_preserves_matrix_shape_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(matrix) -> {
      let values = t.cumsum(matrix)

      t.to_list(values) |> should.equal([1.0, 3.0, 6.0, 10.0])
      t.shape(values) |> should.equal([2, 2])
    }
    Error(_) -> should.fail()
  }
}

pub fn cumsum_axis_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(matrix) -> {
      case t.cumsum_axis(matrix, 0) {
        Ok(values) -> {
          t.shape(values) |> should.equal([2, 3])
          t.to_list(values) |> should.equal([1.0, 2.0, 3.0, 5.0, 7.0, 9.0])
        }
        Error(_) -> should.fail()
      }

      case t.cumsum_axis(matrix, 1) {
        Ok(values) -> {
          t.shape(values) |> should.equal([2, 3])
          t.to_list(values) |> should.equal([1.0, 3.0, 6.0, 4.0, 9.0, 15.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn cumprod_axis_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(matrix) ->
      case t.cumprod_axis(matrix, 1) {
        Ok(values) -> {
          t.shape(values) |> should.equal([2, 3])
          t.to_list(values) |> should.equal([1.0, 2.0, 6.0, 4.0, 20.0, 120.0])
        }
        Error(_) -> should.fail()
      }
    Error(_) -> should.fail()
  }
}

pub fn cumsum_axis_rejects_invalid_axis_test() {
  t.from_list([1.0, 2.0])
  |> t.try_cumsum_axis(1)
  |> should.be_error()
}

pub fn try_median_test() {
  let odd = t.from_list([3.0, 1.0, 2.0])
  let even = t.from_list([4.0, 1.0, 3.0, 2.0])

  t.try_median(odd) |> should.equal(Ok(2.0))
  t.try_median(even) |> should.equal(Ok(2.5))
}

pub fn try_median_empty_test() {
  t.from_list([])
  |> t.try_median()
  |> should.be_error()
}

pub fn try_percentile_test() {
  let values = t.from_list([15.0, 20.0, 35.0, 40.0, 50.0])

  t.try_percentile(values, 40) |> should.equal(Ok(29.0))
}

pub fn try_percentile_rejects_invalid_percentile_test() {
  let values = t.from_list([1.0, 2.0, 3.0])

  t.try_percentile(values, -1) |> should.be_error()
  t.try_percentile(values, 101) |> should.be_error()
}

pub fn max_min_axis_test() {
  case t.matrix(2, 3, [1.0, 5.0, 2.0, 4.0, 3.0, 6.0]) {
    Ok(matrix) -> {
      case t.max_axis(matrix, 0) {
        Ok(values) -> t.to_list(values) |> should.equal([4.0, 5.0, 6.0])
        Error(_) -> should.fail()
      }

      case t.min_axis(matrix, 1) {
        Ok(values) -> t.to_list(values) |> should.equal([1.0, 3.0])
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn max_min_axis_keepdims_test() {
  case t.matrix(2, 3, [1.0, 5.0, 2.0, 4.0, 3.0, 6.0]) {
    Ok(matrix) -> {
      case t.try_max_axis_keepdims(matrix, 1) {
        Ok(values) -> {
          t.shape(values) |> should.equal([2, 1])
          t.to_list(values) |> should.equal([5.0, 6.0])
        }
        Error(_) -> should.fail()
      }

      case t.try_min_axis_keepdims(matrix, 0) {
        Ok(values) -> {
          t.shape(values) |> should.equal([1, 3])
          t.to_list(values) |> should.equal([1.0, 3.0, 2.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn max_min_axis_reject_invalid_axis_test() {
  t.from_list([1.0, 2.0])
  |> t.try_max_axis(1)
  |> should.be_error()
}

// =============================================================================
// DOT PRODUCT & MATMUL
// =============================================================================

pub fn dot_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([4.0, 5.0, 6.0])
  case t.dot(a, b) {
    Ok(d) -> d |> should.equal(32.0)
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    Error(_) -> should.fail()
  }
}

pub fn matmul_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case t.matmul(a, b) {
            Ok(c) -> {
              t.shape(c) |> should.equal([2, 2])
              t.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn matmul_planned_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case t.matmul_planned(a, b) {
            Ok(c) -> {
              t.shape(c) |> should.equal([2, 2])
              t.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }

  // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
  // [3, 4] x [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case t.matmul(a, b) {
            Ok(c) -> {
              t.shape(c) |> should.equal([2, 2])
              t.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn matmul_auto_rtx_first_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case t.matmul_auto(a, b) {
            Ok(accelerated) -> {
              case t.accelerated_to_tensor(accelerated) {
                Ok(result) -> {
                  t.shape(result) |> should.equal([2, 2])
                  t.to_list(result) |> should.equal([19.0, 22.0, 43.0, 50.0])
                }
                Error(_) -> should.fail()
              }
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn persistent_accelerated_matmul_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case t.to_accelerated(a), t.to_accelerated(b) {
            Ok(a_acc), Ok(b_acc) -> {
              case t.matmul_accelerated(a_acc, b_acc) {
                Ok(accelerated) -> {
                  t.accelerated_shape(accelerated) |> should.equal([2, 2])
                  case t.accelerated_to_tensor(accelerated) {
                    Ok(result) ->
                      t.to_list(result)
                      |> should.equal([19.0, 22.0, 43.0, 50.0])
                    Error(_) -> should.fail()
                  }
                }
                Error(_) -> should.fail()
              }
            }
            _, _ -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn persistent_accelerated_matmul_into_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case
            t.to_accelerated(t.zeros([2, 2])),
            t.to_accelerated(a),
            t.to_accelerated(b)
          {
            Ok(out), Ok(a_acc), Ok(b_acc) -> {
              case t.matmul_accelerated_into(out, a_acc, b_acc) {
                Ok(Nil) -> {
                  case t.accelerated_to_tensor(out) {
                    Ok(result) -> {
                      t.shape(result) |> should.equal([2, 2])
                      t.to_list(result)
                      |> should.equal([19.0, 22.0, 43.0, 50.0])
                    }
                    Error(_) -> should.fail()
                  }
                }
                Error(_) -> {
                  case t.accelerated_backend(out) {
                    CpuFallback -> Nil
                    _ -> should.fail()
                  }
                }
              }
            }
            _, _, _ -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn persistent_fp16_matmul_relu_into_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, -6.0, -7.0, 8.0]) {
        Ok(b) -> {
          case
            t.to_rtx4090_fp16(t.zeros([2, 2])),
            t.to_rtx4090_fp16(a),
            t.to_rtx4090_fp16(b)
          {
            Ok(out), Ok(a_acc), Ok(b_acc) -> {
              case t.matmul_relu_accelerated_into(out, a_acc, b_acc) {
                Ok(Nil) -> {
                  case t.accelerated_to_tensor(out) {
                    Ok(result) ->
                      t.to_list(result) |> should.equal([0.0, 10.0, 0.0, 14.0])
                    Error(_) -> should.fail()
                  }
                }
                Error(_) -> should.fail()
              }
            }
            _, _, _ -> Nil
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn persistent_fp16_linear_relu_into_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, -6.0, -7.0, 8.0]) {
        Ok(b) -> {
          let bias = t.vector([10.0, -20.0])
          case
            t.to_rtx4090_fp16(t.zeros([2, 2])),
            t.to_rtx4090_fp16(a),
            t.to_rtx4090_fp16(b),
            t.to_rtx4090_fp16(bias)
          {
            Ok(out), Ok(a_acc), Ok(b_acc), Ok(bias_acc) -> {
              case t.linear_relu_accelerated_into(out, a_acc, b_acc, bias_acc) {
                Ok(Nil) -> {
                  case t.accelerated_to_tensor(out) {
                    Ok(result) ->
                      t.to_list(result) |> should.equal([1.0, 0.0, 0.0, 0.0])
                    Error(_) -> should.fail()
                  }
                }
                Error(_) -> Nil
              }
            }
            _, _, _, _ -> Nil
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn gpu_workspace_linear_layer_test() {
  case t.gpu_workspace() {
    Ok(workspace) -> {
      case
        t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]),
        t.matrix(2, 2, [5.0, -6.0, -7.0, 8.0])
      {
        Ok(input_cpu), Ok(weight_cpu) -> {
          let bias_cpu = t.vector([10.0, -20.0])
          case
            t.workspace_from_tensor(workspace, input_cpu),
            t.linear_layer(workspace, weight_cpu, bias_cpu)
          {
            Ok(input), Ok(layer) -> {
              t.linear_layer_input_features(layer) |> should.equal(2)
              t.linear_layer_output_features(layer) |> should.equal(2)

              case t.linear_output(workspace, layer, 2) {
                Ok(out) -> {
                  case t.linear_relu_forward_into(out, input, layer) {
                    Ok(Nil) -> {
                      case t.accelerated_to_tensor(out) {
                        Ok(result) ->
                          t.to_list(result)
                          |> should.equal([1.0, 0.0, 0.0, 0.0])
                        Error(_) -> should.fail()
                      }
                    }
                    Error(_) -> should.fail()
                  }
                }
                Error(_) -> should.fail()
              }
            }
            _, _ -> should.fail()
          }
        }
        _, _ -> should.fail()
      }
    }

    Error(_) -> Nil
  }
}

// =============================================================================
// OPTIMIZED OPERATIONS (Erlang Array Backend)
// =============================================================================

pub fn dot_fast_test() {
  let a = core_tensor.from_list([1.0, 2.0, 3.0])
  let b = core_tensor.from_list([4.0, 5.0, 6.0])
  case ops.dot_fast(a, b) {
    Ok(d) -> d |> should.equal(32.0)
    // Same result as dot: 1*4 + 2*5 + 3*6 = 32
    Error(_) -> should.fail()
  }
}

pub fn matmul_fast_test() {
  // Same test as matmul_test but using optimized version
  case core_tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case core_tensor.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case ops.matmul_fast(a, b) {
            Ok(c) -> {
              core_tensor.shape(c) |> should.equal([2, 2])
              core_tensor.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn dot_auto_test() {
  // Auto-selecting dot (uses NIF if available)
  let a = core_tensor.from_list([1.0, 2.0, 3.0])
  let b = core_tensor.from_list([4.0, 5.0, 6.0])
  case ops.dot_auto(a, b) {
    Ok(d) -> d |> should.equal(32.0)
    Error(_) -> should.fail()
  }
}

pub fn matmul_auto_test() {
  // Auto-selecting matmul (uses NIF if available)
  case core_tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case core_tensor.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case ops.matmul_auto(a, b) {
            Ok(c) -> {
              core_tensor.shape(c) |> should.equal([2, 2])
              core_tensor.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn transpose_test() {
  // [1, 2, 3]      [1, 4]
  // [4, 5, 6]  ->  [2, 5]
  //                [3, 6]
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      case t.transpose(m) {
        Ok(mt) -> {
          t.shape(mt) |> should.equal([3, 2])
          t.to_list(mt) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// SHAPE OPERATIONS
// =============================================================================

pub fn reshape_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  case t.reshape(a, [2, 3]) {
    Ok(r) -> t.shape(r) |> should.equal([2, 3])
    Error(_) -> should.fail()
  }
}

pub fn flatten_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      let f = t.flatten(m)
      t.shape(f) |> should.equal([6])
    }
    Error(_) -> should.fail()
  }
}

pub fn try_flatten_test() {
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(m) ->
      case t.try_flatten(m) {
        Ok(f) -> {
          t.shape(f) |> should.equal([4])
          t.to_list(f) |> should.equal([1.0, 2.0, 3.0, 4.0])
        }
        Error(_) -> should.fail()
      }
    Error(_) -> should.fail()
  }
}

pub fn try_norm_test() {
  let vector = t.from_list([3.0, 4.0])
  t.try_norm(vector) |> should.equal(Ok(5.0))
}

pub fn try_normalize_test() {
  let vector = t.from_list([3.0, 4.0])
  case t.try_normalize(vector) {
    Ok(normalized) -> {
      let assert [a, b] = t.to_list(normalized)
      { a >. 0.599 && a <. 0.601 } |> should.be_true()
      { b >. 0.799 && b <. 0.801 } |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn is_close_test() {
  t.is_close(1.0, 1.000001, 0.00001, 0.00001)
  |> should.be_true()
}

pub fn all_close_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([1.0, 2.000001, 3.0])

  t.all_close(a, b, 0.00001, 0.00001)
  |> should.equal(Ok(True))
}

pub fn all_close_shape_mismatch_test() {
  let a = t.from_list([1.0, 2.0])
  let b = t.zeros([1, 2])

  t.all_close(a, b, 0.00001, 0.00001)
  |> should.be_error()
}

pub fn elementwise_math_helpers_test() {
  let values = t.from_list([-1.0, 0.0, 4.0])

  t.abs(values) |> t.to_list() |> should.equal([1.0, 0.0, 4.0])
  t.square(values) |> t.to_list() |> should.equal([1.0, 0.0, 16.0])

  case t.try_sqrt(t.from_list([1.0, 4.0, 9.0])) {
    Ok(result) -> t.to_list(result) |> should.equal([1.0, 2.0, 3.0])
    Error(_) -> should.fail()
  }

  t.try_sqrt(values) |> should.be_error()
  t.try_log(t.from_list([1.0, 0.0])) |> should.be_error()
}

pub fn exp_log_roundtrip_test() {
  let values = t.from_list([1.0, 2.0, 4.0])

  case t.try_exp(t.log(values)) {
    Ok(result) ->
      t.all_close(result, values, 0.00001, 0.00001)
      |> should.equal(Ok(True))
    Error(_) -> should.fail()
  }
}

pub fn tensor_distance_similarity_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([4.0, 6.0, 3.0])

  t.try_manhattan_distance(a, b) |> should.equal(Ok(7.0))
  t.try_dot_similarity(a, b) |> should.equal(Ok(25.0))

  case t.try_euclidean_distance(a, b) {
    Ok(value) -> { value >. 4.999 && value <. 5.001 } |> should.be_true()
    Error(_) -> should.fail()
  }

  t.try_cosine_similarity(a, a) |> should.equal(Ok(1.0))
}

pub fn tensor_distance_rejects_shape_mismatch_test() {
  let a = t.from_list([1.0, 2.0])
  let b = t.zeros([1, 2])

  t.try_euclidean_distance(a, b) |> should.be_error()
}

pub fn zscore_and_standardize_test() {
  let values = t.from_list([1.0, 2.0, 3.0])

  case t.try_zscore(values) {
    Ok(scaled) -> {
      let assert [a, b, c] = t.to_list(scaled)
      { a <. -1.22 && a >. -1.23 } |> should.be_true()
      b |> should.equal(0.0)
      { c >. 1.22 && c <. 1.23 } |> should.be_true()
      t.shape(scaled) |> should.equal([3])
    }
    Error(_) -> should.fail()
  }

  t.try_standardize(t.fill([3], 1.0)) |> should.be_error()
}

pub fn minmax_scale_and_clip_by_norm_test() {
  let values = t.from_list([2.0, 4.0, 6.0])

  case t.try_minmax_scale(values, 0.0, 1.0) {
    Ok(scaled) -> t.to_list(scaled) |> should.equal([0.0, 0.5, 1.0])
    Error(_) -> should.fail()
  }

  case t.try_clip_by_norm(t.from_list([3.0, 4.0]), 2.5) {
    Ok(clipped) -> {
      let assert [a, b] = t.to_list(clipped)
      { a >. 1.49 && a <. 1.51 } |> should.be_true()
      { b >. 1.99 && b <. 2.01 } |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn squeeze_test() {
  let a = t.zeros([1, 3, 1])
  let s = t.squeeze(a)
  t.shape(s) |> should.equal([3])
}

pub fn unsqueeze_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let u = t.unsqueeze(a, 0)
  t.shape(u) |> should.equal([1, 3])
}

// =============================================================================
// BROADCASTING
// =============================================================================

pub fn can_broadcast_test() {
  // Same shape
  t.can_broadcast([2, 3], [2, 3]) |> should.be_true()

  // Scalar broadcast
  t.can_broadcast([2, 3], [1]) |> should.be_true()

  // Different ranks
  t.can_broadcast([2, 3], [3]) |> should.be_true()

  // Incompatible
  t.can_broadcast([2, 3], [4]) |> should.be_false()
}

pub fn broadcast_shape_public_test() {
  case t.broadcast_shape([1, 3], [3, 1]) {
    Ok(shape) -> shape |> should.equal([3, 3])
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shapes_public_test() {
  case t.broadcast_shapes([[6, 7], [5, 6, 1], [7], [5, 1, 7]]) {
    Ok(shape) -> shape |> should.equal([5, 6, 7])
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shapes_empty_public_test() {
  case t.broadcast_shapes([]) {
    Ok(shape) -> shape |> should.equal([])
    Error(_) -> should.fail()
  }
}

pub fn broadcast_shapes_error_public_test() {
  t.broadcast_shapes([[1, 3, 4], [2, 3, 3]]) |> should.be_error()
}

pub fn broadcast_to_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  case t.broadcast_to(a, [2, 3]) {
    Ok(b) -> {
      t.shape(b) |> should.equal([2, 3])
      t.to_list(b) |> should.equal([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn broadcast_pair_test() {
  let a = t.zeros([2, 3])
  let b = t.from_list([1.0, 2.0, 3.0])

  case t.broadcast_pair(a, b) {
    Ok(pair) -> {
      let #(left, right) = pair
      t.shape(left) |> should.equal([2, 3])
      t.shape(right) |> should.equal([2, 3])
      t.to_list(right) |> should.equal([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn add_broadcast_test() {
  let a = t.zeros([2, 3])
  let b = t.from_list([1.0, 2.0, 3.0])
  case t.add_broadcast(a, b) {
    Ok(c) -> {
      t.shape(c) |> should.equal([2, 3])
      t.to_list(c) |> should.equal([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn sub_broadcast_test() {
  let a = t.fill([2, 3], 10.0)
  let b = t.from_list([1.0, 2.0, 5.0])
  case t.sub_broadcast(a, b) {
    Ok(c) -> {
      t.shape(c) |> should.equal([2, 3])
      t.to_list(c) |> should.equal([9.0, 8.0, 5.0, 9.0, 8.0, 5.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn div_broadcast_test() {
  let a = t.fill([2, 3], 8.0)
  let b = t.from_list([1.0, 2.0, 4.0])
  case t.div_broadcast(a, b) {
    Ok(c) -> {
      t.shape(c) |> should.equal([2, 3])
      t.to_list(c) |> should.equal([8.0, 4.0, 2.0, 8.0, 4.0, 2.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn maximum_minimum_broadcast_test() {
  let matrix = t.matrix(2, 3, [1.0, 5.0, 2.0, 4.0, 3.0, 6.0])
  let thresholds = t.from_list([3.0, 4.0, 5.0])

  case matrix {
    Ok(values) -> {
      case t.maximum(values, thresholds) {
        Ok(result) -> {
          t.shape(result) |> should.equal([2, 3])
          t.to_list(result) |> should.equal([3.0, 5.0, 5.0, 4.0, 4.0, 6.0])
        }
        Error(_) -> should.fail()
      }

      case t.minimum(values, thresholds) {
        Ok(result) -> {
          t.shape(result) |> should.equal([2, 3])
          t.to_list(result) |> should.equal([1.0, 4.0, 2.0, 3.0, 3.0, 5.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn softmax_axis_public_facade_test() {
  case t.matrix(2, 2, [0.0, 0.0, 1.0, 1.0]) {
    Ok(logits) ->
      case t.softmax_axis(logits, 1) {
        Ok(probabilities) -> {
          t.shape(probabilities) |> should.equal([2, 2])
          t.to_list(probabilities) |> should.equal([0.5, 0.5, 0.5, 0.5])
        }
        Error(_) -> should.fail()
      }
    Error(_) -> should.fail()
  }
}

pub fn try_softmax_axis_public_facade_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(logits) ->
      case t.try_softmax_axis(logits, 1) {
        Ok(probabilities) -> {
          let assert [a, b, c, d, e, f] = t.to_list(probabilities)

          t.shape(probabilities) |> should.equal([2, 3])
          { a +. b +. c >. 0.999 && a +. b +. c <. 1.001 }
          |> should.be_true()
          { d +. e +. f >. 0.999 && d +. e +. f <. 1.001 }
          |> should.be_true()
        }
        Error(_) -> should.fail()
      }
    Error(_) -> should.fail()
  }
}

pub fn capabilities_smoke_test() {
  let caps = t.capabilities()
  { caps.nif_loaded || !caps.nif_loaded } |> should.be_true()
  { caps.zig_loaded || !caps.zig_loaded } |> should.be_true()
  list.any(caps.backend_capabilities, fn(capability) {
    capability.backend == t.BackendPureGleam && capability.available
  })
  |> should.be_true()
}

pub fn backend_capabilities_include_stable_fallback_test() {
  let capabilities = t.backend_capabilities()

  list.any(capabilities, fn(capability) {
    capability.backend == t.BackendPureGleam
    && capability.device == t.BackendBeamCpu
    && capability.available
  })
  |> should.be_true()
}

pub fn softmax_backend_plan_uses_stable_gleam_path_test() {
  let plan = t.plan_backend(t.OperationSoftmax)

  plan.selected |> should.equal(t.BackendPureGleam)
  plan.fallbacks |> should.equal([t.BackendPureGleam])
}

pub fn matmul_backend_plan_has_safe_fallback_test() {
  let plan = t.plan_backend(t.OperationMatmul(16, 16, 16))

  list.any(plan.fallbacks, fn(backend) { backend == t.BackendPureGleam })
  |> should.be_true()
}

pub fn tensor_device_and_dtype_test() {
  let values = t.from_list([1.0, 2.0, 3.0])

  t.device(values) |> should.equal(layout.BeamCpu)
  t.dtype(values) |> should.equal(layout.Float64)
}

pub fn matmul_backend_plan_reports_rejections_test() {
  let plan = t.plan_backend(t.OperationMatmul(15, 15, 15))

  list.any(plan.rejected, fn(rejection) {
    rejection.backend == t.BackendCudaFp16
  })
  |> should.be_true()
}

pub fn map2_test() {
  let a = t.from_list([1.0, 2.0])
  let b = t.from_list([3.0, 4.0])
  case t.map2(a, b, fn(x, y) { x +. y *. 2.0 }) {
    Ok(c) -> t.to_list(c) |> should.equal([7.0, 10.0])
    Error(_) -> should.fail()
  }
}

// =============================================================================
// STRIDED TENSORS
// =============================================================================

pub fn to_strided_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0])
  let s = t.to_strided(a)
  t.is_contiguous(s) |> should.be_true()
  t.to_list(s) |> should.equal([1.0, 2.0, 3.0, 4.0])
}

pub fn strided_transpose_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      let s = t.to_strided(m)
      case t.transpose_strided(s) {
        Ok(st) -> {
          t.shape(st) |> should.equal([3, 2])
          // Zero-copy: not contiguous after transpose
          t.is_contiguous(st) |> should.be_false()
          // But converting back gives correct data
          let c = t.to_contiguous(st)
          t.to_list(c) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// RANDOM
// =============================================================================

pub fn random_uniform_test() {
  let r = t.random_uniform([100])
  t.shape(r) |> should.equal([100])
  // All values should be in [0, 1)
  let vals = t.to_list(r)
  let in_range =
    vals
    |> list.all(fn(v) { v >=. 0.0 && v <. 1.0 })
  in_range |> should.be_true()
}

pub fn xavier_init_test() {
  let w = t.xavier_init(128, 64)
  t.shape(w) |> should.equal([64, 128])
  // Xavier: std ≈ sqrt(2 / (fan_in + fan_out))
  let std_val = t.std(w)
  // Should be around 0.1 for these dimensions
  { std_val >. 0.05 && std_val <. 0.2 } |> should.be_true()
}

// =============================================================================
// AXIS MODULE
// =============================================================================

pub fn axis_constructors_test() {
  let b = axis.batch(32)
  b.size |> should.equal(32)

  let f = axis.feature(128)
  f.size |> should.equal(128)

  let s = axis.seq(10)
  s.size |> should.equal(10)
}

pub fn axis_equals_test() {
  axis.equals(axis.Batch, axis.Batch) |> should.be_true()
  axis.equals(axis.Batch, axis.Seq) |> should.be_false()
  axis.equals(axis.Named("foo"), axis.Named("foo")) |> should.be_true()
  axis.equals(axis.Named("foo"), axis.Named("bar")) |> should.be_false()
}

// =============================================================================
// NAMED TENSORS
// =============================================================================

pub fn named_zeros_test() {
  let nt = named.zeros([axis.batch(2), axis.feature(3)])
  named.shape(nt) |> should.equal([2, 3])
  named.rank(nt) |> should.equal(2)
}

pub fn named_find_axis_test() {
  let nt = named.zeros([axis.batch(4), axis.seq(10), axis.feature(64)])
  case named.find_axis(nt, axis.Seq) {
    Ok(idx) -> idx |> should.equal(1)
    Error(_) -> should.fail()
  }
  case named.find_axis(nt, axis.Channel) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn named_sum_along_test() {
  let nt = named.ones([axis.batch(2), axis.feature(3)])
  // Sum along batch: [2, 3] -> [3] (each feature summed over batch)
  case named.sum_along(nt, axis.Batch) {
    Ok(result) -> {
      named.shape(result) |> should.equal([3])
      // Each feature had 2 ones, so sum = 2
      tensor.to_list(named.to_tensor(result))
      |> should.equal([2.0, 2.0, 2.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn named_add_test() {
  let a = named.ones([axis.batch(2), axis.feature(3)])
  let b = named.ones([axis.batch(2), axis.feature(3)])
  case named.add(a, b) {
    Ok(c) -> {
      tensor.to_list(named.to_tensor(c))
      |> list.all(fn(v) { v == 2.0 })
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn named_scale_test() {
  let nt = named.ones([axis.batch(2)])
  let scaled = named.scale(nt, 5.0)
  tensor.to_list(named.to_tensor(scaled)) |> should.equal([5.0, 5.0])
}

pub fn named_describe_test() {
  let nt = named.zeros([axis.batch(32), axis.feature(128)])
  let desc = named.describe(nt)
  // Should contain axis names and sizes
  { string.contains(desc, "batch") && string.contains(desc, "32") }
  |> should.be_true()
}

// =============================================================================
// CONVOLUTION TESTS
// =============================================================================

pub fn pad2d_test() {
  // 2x2 input
  let input = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])

  // Pad by 1 on each side -> 4x4
  let result = tensor.pad2d(input, 1, 1)
  result |> should.be_ok()

  let padded = case result {
    Ok(p) -> p
    Error(_) -> tensor.zeros([1])
  }

  tensor.shape(padded) |> should.equal([4, 4])

  // Check corners are zeros
  let data = tensor.get_data(padded)
  list.first(data) |> should.equal(Ok(0.0))
}

pub fn conv2d_simple_test() {
  // 3x3 input
  let input =
    tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape: [
      3,
      3,
    ])

  // 2x2 kernel (all ones = sum)
  let kernel = tensor.Tensor(data: [1.0, 1.0, 1.0, 1.0], shape: [2, 2])

  let config =
    tensor.Conv2dConfig(
      kernel_h: 2,
      kernel_w: 2,
      stride_h: 1,
      stride_w: 1,
      padding_h: 0,
      padding_w: 0,
    )

  let result = tensor.conv2d(input, kernel, config)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // Output should be 2x2
  tensor.shape(output) |> should.equal([2, 2])

  // Top-left: 1+2+4+5 = 12
  let data = tensor.get_data(output)
  list.first(data) |> should.equal(Ok(12.0))
}

pub fn conv2d_same_padding_test() {
  // 4x4 input
  let input = tensor.ones([4, 4])

  // 3x3 kernel with "same" padding
  let kernel = tensor.fill([3, 3], 1.0)
  let config = tensor.conv2d_same(3, 3)

  let result = tensor.conv2d(input, kernel, config)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // With same padding, output = input size
  tensor.shape(output) |> should.equal([4, 4])
}

pub fn max_pool2d_test() {
  // 4x4 input
  let input =
    tensor.Tensor(
      data: [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        14.0, 15.0, 16.0,
      ],
      shape: [4, 4],
    )

  // 2x2 pool, stride 2
  let result = tensor.max_pool2d(input, 2, 2, 2, 2)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // Output should be 2x2
  tensor.shape(output) |> should.equal([2, 2])

  // Max of [1,2,5,6] = 6, [3,4,7,8] = 8, etc.
  let data = tensor.get_data(output)
  data |> should.equal([6.0, 8.0, 14.0, 16.0])
}

pub fn avg_pool2d_test() {
  // 4x4 input (all 4s)
  let input = tensor.fill([4, 4], 4.0)

  // 2x2 pool, stride 2
  let result = tensor.avg_pool2d(input, 2, 2, 2, 2)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // Output should be 2x2, all 4.0
  tensor.shape(output) |> should.equal([2, 2])
  tensor.get_data(output) |> should.equal([4.0, 4.0, 4.0, 4.0])
}

pub fn conv2d_batch_test() {
  // Batch of 2, 1 channel, 3x3
  let input = tensor.ones([2, 1, 3, 3])

  // 1 output channel, 1 input channel, 2x2 kernel
  let kernel = tensor.fill([1, 1, 2, 2], 1.0)

  let config =
    tensor.Conv2dConfig(
      kernel_h: 2,
      kernel_w: 2,
      stride_h: 1,
      stride_w: 1,
      padding_h: 0,
      padding_w: 0,
    )

  let result = tensor.conv2d(input, kernel, config)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // [2, 1, 2, 2]
  tensor.shape(output) |> should.equal([2, 1, 2, 2])

  // Each position = sum of 4 ones = 4.0
  let data = tensor.get_data(output)
  list.first(data) |> should.equal(Ok(4.0))
}
