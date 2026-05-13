import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/layout

pub fn main() {
  gleeunit.main()
}

pub fn stable_creation_and_layout_contract_test() {
  let tensor = t.ones([2, 3])

  t.shape(tensor) |> should.equal([2, 3])
  t.size(tensor) |> should.equal(6)
  t.rank(tensor) |> should.equal(2)
  t.to_list(tensor) |> should.equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
  t.try_to_list(tensor) |> should.equal(Ok([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
  t.try_sum(tensor) |> should.equal(Ok(6.0))
  t.device(tensor) |> should.equal(layout.BeamCpu)
  t.dtype(tensor) |> should.equal(layout.Float64)
  t.linspace(0.0, 1.0, 3) |> t.to_list() |> should.equal([0.0, 0.5, 1.0])
  t.zeros_like(tensor) |> t.shape() |> should.equal([2, 3])
  t.eye(2) |> t.to_list() |> should.equal([1.0, 0.0, 0.0, 1.0])

  case t.try_logspace(1.0, 3.0, 3, 10.0) {
    Ok(logs) -> t.to_list(logs) |> should.equal([10.0, 100.0, 1000.0])
    Error(_) -> should.fail()
  }

  case t.try_diag(t.from_list([2.0, 4.0])) {
    Ok(diagonal) -> t.to_list(diagonal) |> should.equal([2.0, 0.0, 0.0, 4.0])
    Error(_) -> should.fail()
  }

  case t.try_unsqueeze(tensor, 0) {
    Ok(batch) -> t.shape(batch) |> should.equal([1, 2, 3])
    Error(_) -> should.fail()
  }

  case t.try_to_strided(tensor) {
    Ok(strided) -> {
      t.is_contiguous(strided) |> should.be_true()
      case t.try_to_contiguous(strided) {
        Ok(contiguous) ->
          t.to_list(contiguous) |> should.equal(t.to_list(tensor))
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn stable_fallible_unary_contract_test() {
  let tensor = t.from_list([1.0, 2.0, 3.0])

  case t.try_map(tensor, fn(x) { x +. 1.0 }) {
    Ok(mapped) -> t.to_list(mapped) |> should.equal([2.0, 3.0, 4.0])
    Error(_) -> should.fail()
  }

  case t.try_scale(tensor, 2.0) {
    Ok(scaled) -> t.to_list(scaled) |> should.equal([2.0, 4.0, 6.0])
    Error(_) -> should.fail()
  }

  case t.try_add_scalar(tensor, 1.5) {
    Ok(shifted) -> t.to_list(shifted) |> should.equal([2.5, 3.5, 4.5])
    Error(_) -> should.fail()
  }

  case t.try_negate(tensor) {
    Ok(negated) -> t.to_list(negated) |> should.equal([-1.0, -2.0, -3.0])
    Error(_) -> should.fail()
  }

  case t.try_clamp(t.from_list([-1.0, 0.5, 2.0]), 0.0, 1.0) {
    Ok(clamped) -> t.to_list(clamped) |> should.equal([0.0, 0.5, 1.0])
    Error(_) -> should.fail()
  }

  t.is_close(1.0, 1.000001, 0.00001, 0.00001)
  |> should.be_true()
  t.all_close(tensor, tensor, 0.00001, 0.00001)
  |> should.equal(Ok(True))

  t.try_max(tensor) |> should.equal(Ok(3.0))
  t.try_min(tensor) |> should.equal(Ok(1.0))
  t.try_argmax(tensor) |> should.equal(Ok(2))
  t.try_argmin(tensor) |> should.equal(Ok(0))
  t.try_abs(t.from_list([-1.0, 2.0]))
  |> should.equal(Ok(t.from_list([1.0, 2.0])))
  t.try_square(t.from_list([2.0, 3.0]))
  |> should.equal(Ok(t.from_list([4.0, 9.0])))
  t.try_floor(t.from_list([1.8, -1.2]))
  |> should.equal(Ok(t.from_list([1.0, -2.0])))
  t.try_ceil(t.from_list([1.2, -1.8]))
  |> should.equal(Ok(t.from_list([2.0, -1.0])))
  t.try_round(t.from_list([1.2, 1.8]))
  |> should.equal(Ok(t.from_list([1.0, 2.0])))
  t.try_sign(t.from_list([-2.0, 0.0, 3.0]))
  |> should.equal(Ok(t.from_list([-1.0, 0.0, 1.0])))
  t.try_reciprocal(t.from_list([2.0, 4.0]))
  |> should.equal(Ok(t.from_list([0.5, 0.25])))
  t.try_mean(tensor) |> should.equal(Ok(2.0))
  t.try_product(tensor) |> should.equal(Ok(6.0))
  t.try_cumsum(tensor) |> should.equal(Ok(t.from_list([1.0, 3.0, 6.0])))
  t.try_cumprod(tensor) |> should.equal(Ok(t.from_list([1.0, 2.0, 6.0])))
  t.try_cumsum_axis(tensor, 0) |> should.equal(Ok(t.from_list([1.0, 3.0, 6.0])))
  t.try_median(tensor) |> should.equal(Ok(2.0))
  t.try_percentile(tensor, 50) |> should.equal(Ok(2.0))
  t.try_variance(tensor) |> should.equal(Ok(0.6666666666666666))
  t.try_manhattan_distance(tensor, tensor) |> should.equal(Ok(0.0))
  t.try_dot_similarity(tensor, tensor) |> should.equal(Ok(14.0))

  case t.try_flatten(tensor) {
    Ok(flat) -> {
      t.shape(flat) |> should.equal([3])
      t.to_list(flat) |> should.equal([1.0, 2.0, 3.0])
    }
    Error(_) -> should.fail()
  }

  t.try_norm(t.from_list([3.0, 4.0])) |> should.equal(Ok(5.0))
}

pub fn stable_axis_reduction_contract_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(matrix) -> {
      case t.try_sum_axis(matrix, 0) {
        Ok(reduced) -> {
          t.shape(reduced) |> should.equal([3])
          t.to_list(reduced) |> should.equal([5.0, 7.0, 9.0])
        }
        Error(_) -> should.fail()
      }

      case t.try_mean_axis_keepdims(matrix, 1) {
        Ok(reduced) -> {
          t.shape(reduced) |> should.equal([2, 1])
          t.to_list(reduced) |> should.equal([2.0, 5.0])
        }
        Error(_) -> should.fail()
      }

      case t.try_max_axis(matrix, 1) {
        Ok(reduced) -> t.to_list(reduced) |> should.equal([3.0, 6.0])
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn stable_broadcasting_contract_test() {
  let matrix = t.fill([2, 3], 10.0)
  let row = t.from_list([1.0, 2.0, 4.0])

  case t.broadcast_shapes([[6, 7], [5, 6, 1], [7], [5, 1, 7]]) {
    Ok(shape) -> shape |> should.equal([5, 6, 7])
    Error(_) -> should.fail()
  }

  case t.broadcast_pair(matrix, row) {
    Ok(pair) -> {
      let #(left, right) = pair
      t.shape(left) |> should.equal([2, 3])
      t.shape(right) |> should.equal([2, 3])
      t.to_list(right) |> should.equal([1.0, 2.0, 4.0, 1.0, 2.0, 4.0])
    }
    Error(_) -> should.fail()
  }

  case t.sub_broadcast(matrix, row) {
    Ok(result) -> {
      t.shape(result) |> should.equal([2, 3])
      t.to_list(result) |> should.equal([9.0, 8.0, 6.0, 9.0, 8.0, 6.0])
    }
    Error(_) -> should.fail()
  }

  case t.div_broadcast(t.fill([2, 3], 8.0), row) {
    Ok(result) -> {
      t.shape(result) |> should.equal([2, 3])
      t.to_list(result) |> should.equal([8.0, 4.0, 2.0, 8.0, 4.0, 2.0])
    }
    Error(_) -> should.fail()
  }

  case t.maximum(matrix, row) {
    Ok(result) ->
      t.to_list(result)
      |> should.equal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    Error(_) -> should.fail()
  }

  case t.minimum(matrix, row) {
    Ok(result) ->
      t.to_list(result) |> should.equal([1.0, 2.0, 4.0, 1.0, 2.0, 4.0])
    Error(_) -> should.fail()
  }

  case t.greater(matrix, row) {
    Ok(mask) -> t.to_list(mask) |> should.equal([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    Error(_) -> should.fail()
  }

  case t.where(row, matrix, t.zeros([1])) {
    Ok(result) ->
      t.to_list(result) |> should.equal([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    Error(_) -> should.fail()
  }
}

pub fn stable_linear_algebra_contract_test() {
  let a = t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0])

  case a, b {
    Ok(left), Ok(right) ->
      case t.matmul_planned(left, right) {
        Ok(result) -> {
          t.shape(result) |> should.equal([2, 2])
          t.to_list(result) |> should.equal([19.0, 22.0, 43.0, 50.0])
        }
        Error(_) -> should.fail()
      }
    _, _ -> should.fail()
  }
}

pub fn stable_softmax_contract_test() {
  let logits = t.from_list([0.0, 0.0, 1.0, 1.0])

  case t.reshape(logits, [2, 2]) {
    Ok(matrix) ->
      case t.try_softmax_axis(matrix, 1) {
        Ok(probabilities) -> {
          t.shape(probabilities) |> should.equal([2, 2])
          t.to_list(probabilities) |> should.equal([0.5, 0.5, 0.5, 0.5])
        }
        Error(_) -> should.fail()
      }
    Error(_) -> should.fail()
  }
}

pub fn stable_backend_planner_contract_test() {
  let capabilities = t.backend_capabilities()

  list.any(capabilities, fn(capability) {
    capability.backend == t.BackendPureGleam && capability.available
  })
  |> should.be_true()

  let plan = t.plan_backend(t.OperationMatmul(15, 15, 15))

  list.any(plan.rejected, fn(rejection) {
    rejection.backend == t.BackendCudaFp16
  })
  |> should.be_true()
  list.any(plan.fallbacks, fn(backend) { backend == t.BackendPureGleam })
  |> should.be_true()
}
