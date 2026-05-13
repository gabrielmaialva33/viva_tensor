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
  t.device(tensor) |> should.equal(layout.BeamCpu)
  t.dtype(tensor) |> should.equal(layout.Float64)
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
      case t.softmax_axis(matrix, 1) {
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
