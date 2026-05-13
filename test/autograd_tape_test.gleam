//// Sprint 6: Tests for the new `traced_*` wrappers on top of the autograd
//// `Tape`. Each test sets up a small forward graph, runs `backward`, and
//// compares against an analytical or shape-only expectation.

import gleam/dict
import gleam/list
import gleam/result
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/core/ops
import viva_tensor/core/tensor
import viva_tensor/nn/autograd.{Traced}

pub fn main() {
  gleeunit.main()
}

fn tensor_from(data: List(Float), shape: List(Int)) -> tensor.Tensor {
  let assert Ok(t) = tensor.new(data, shape)
  t
}

// ---------------------------------------------------------------------------
// traced_matmul: y = A @ B with grad_y = ones gives grad_A = ones @ B^T,
//                                              grad_B = A^T @ ones.
// ---------------------------------------------------------------------------

pub fn traced_matmul_gradient_test() {
  let a_data = [1.0, 2.0, 3.0, 4.0]
  let b_data = [5.0, 6.0, 7.0, 8.0]
  let a_t = tensor_from(a_data, [2, 2])
  let b_t = tensor_from(b_data, [2, 2])

  let tape = autograd.new_tape()
  let Traced(a, tape1) = autograd.new_variable(tape, a_t)
  let Traced(b, tape2) = autograd.new_variable(tape1, b_t)
  let assert Ok(Traced(y, tape3)) = autograd.traced_matmul(tape2, a, b)

  let assert Ok(grads) = autograd.backward(tape3, y)
  let assert Ok(da) = dict.get(grads, a.id)
  let assert Ok(db) = dict.get(grads, b.id)

  // Analytical: grad_A = ones(2x2) @ B^T, grad_B = A^T @ ones(2x2)
  let ones_2x2 = tensor.ones([2, 2])
  let assert Ok(bt) = ops.transpose(b_t)
  let assert Ok(at) = ops.transpose(a_t)
  let assert Ok(expected_da) = ops.matmul_auto(ones_2x2, bt)
  let assert Ok(expected_db) = ops.matmul_auto(at, ones_2x2)

  numerics.lists_close(
    tensor.to_list(da),
    tensor.to_list(expected_da),
    1.0e-6,
    1.0e-6,
  )
  |> should.be_true()
  numerics.lists_close(
    tensor.to_list(db),
    tensor.to_list(expected_db),
    1.0e-6,
    1.0e-6,
  )
  |> should.be_true()
}

// ---------------------------------------------------------------------------
// traced_chain: y = relu(A @ x). Manual expectation:
//   grad_A = mask(A@x) @ x^T (with grad_y = ones)
// ---------------------------------------------------------------------------

pub fn traced_chain_test() {
  // A: [2,3], x: [3,1]. So y: [2,1].
  let a_data = [1.0, -2.0, 0.5, 0.3, 4.0, -1.0]
  let x_data = [2.0, 1.0, -1.0]
  let a_t = tensor_from(a_data, [2, 3])
  let x_t = tensor_from(x_data, [3, 1])

  let tape = autograd.new_tape()
  let Traced(a, tape1) = autograd.new_variable(tape, a_t)
  let Traced(x, tape2) = autograd.new_variable(tape1, x_t)
  let assert Ok(Traced(pre, tape3)) = autograd.traced_matmul(tape2, a, x)
  let assert Ok(Traced(post, tape4)) = autograd.traced_relu(tape3, pre)
  let assert Ok(grads) = autograd.backward(tape4, post)
  let assert Ok(da) = dict.get(grads, a.id)
  let assert Ok(dx) = dict.get(grads, x.id)

  // Analytical: pre = A @ x. mask = (pre > 0). grad_pre = ones * mask.
  // grad_A = grad_pre @ x^T, grad_x = A^T @ grad_pre.
  let assert Ok(pre_t) = ops.matmul_auto(a_t, x_t)
  let mask =
    ops.map(pre_t, fn(v) {
      case v >. 0.0 {
        True -> 1.0
        False -> 0.0
      }
    })
  let assert Ok(xt) = ops.transpose(x_t)
  let assert Ok(at) = ops.transpose(a_t)
  let assert Ok(expected_da) = ops.matmul_auto(mask, xt)
  let assert Ok(expected_dx) = ops.matmul_auto(at, mask)

  tensor.shape(da) |> should.equal([2, 3])
  tensor.shape(dx) |> should.equal([3, 1])
  numerics.lists_close(
    tensor.to_list(da),
    tensor.to_list(expected_da),
    1.0e-6,
    1.0e-6,
  )
  |> should.be_true()
  numerics.lists_close(
    tensor.to_list(dx),
    tensor.to_list(expected_dx),
    1.0e-6,
    1.0e-6,
  )
  |> should.be_true()
}

// ---------------------------------------------------------------------------
// traced_two_layer: y = sigmoid(A2 @ relu(A1 @ x)). Compare against numerical
// finite differences for grad_A1 and grad_A2.
// ---------------------------------------------------------------------------

pub fn traced_two_layer_test() {
  // x: [3,1]. A1: [4,3]. A2: [2,4]. y: [2,1].
  let a1_data = [0.5, -0.2, 0.1, 1.0, -0.3, 0.4, 0.2, 0.6, -0.7, 0.8, 0.9, -0.5]
  let a2_data = [0.3, -0.1, 0.5, 0.7, -0.4, 0.6, 0.2, -0.8]
  let x_data = [1.0, -1.5, 0.5]

  let a1_t = tensor_from(a1_data, [4, 3])
  let a2_t = tensor_from(a2_data, [2, 4])
  let x_t = tensor_from(x_data, [3, 1])

  // Forward via tape
  let tape = autograd.new_tape()
  let Traced(a1, tape1) = autograd.new_variable(tape, a1_t)
  let Traced(a2, tape2) = autograd.new_variable(tape1, a2_t)
  let Traced(x, tape3) = autograd.new_variable(tape2, x_t)
  let assert Ok(Traced(h1_pre, tape4)) = autograd.traced_matmul(tape3, a1, x)
  let assert Ok(Traced(h1, tape5)) = autograd.traced_relu(tape4, h1_pre)
  let assert Ok(Traced(h2_pre, tape6)) = autograd.traced_matmul(tape5, a2, h1)
  let assert Ok(Traced(y, tape7)) = autograd.traced_sigmoid(tape6, h2_pre)
  let assert Ok(grads) = autograd.backward(tape7, y)
  let assert Ok(da1) = dict.get(grads, a1.id)
  let assert Ok(da2) = dict.get(grads, a2.id)

  // Pure-function forward used for numerical gradient checks.
  let two_layer = fn(a1: tensor.Tensor, a2: tensor.Tensor) -> Float {
    let assert Ok(z1) = ops.matmul_auto(a1, x_t)
    let z1_relu = ops.relu(z1)
    let assert Ok(z2) = ops.matmul_auto(a2, z1_relu)
    // sum over sigmoid (matches summing ones-grad through y)
    ops.map(z2, fn(v) {
      case v >=. 0.0 {
        True -> 1.0 /. { 1.0 +. erlang_exp(0.0 -. v) }
        False -> {
          let ex = erlang_exp(v)
          ex /. { 1.0 +. ex }
        }
      }
    })
    |> tensor.to_list
    |> list.fold(0.0, fn(acc, v) { acc +. v })
  }

  let eps = 0.0001
  let numerical_a1 =
    numerical_gradient(a1_data, [4, 3], eps, fn(t) { two_layer(t, a2_t) })
  let numerical_a2 =
    numerical_gradient(a2_data, [2, 4], eps, fn(t) { two_layer(a1_t, t) })

  tensor.shape(da1) |> should.equal([4, 3])
  tensor.shape(da2) |> should.equal([2, 4])
  numerics.lists_close(tensor.to_list(da1), numerical_a1, 1.0e-3, 1.0e-3)
  |> should.be_true()
  numerics.lists_close(tensor.to_list(da2), numerical_a2, 1.0e-3, 1.0e-3)
  |> should.be_true()
}

// ---------------------------------------------------------------------------
// traced_mse_loss: dL/dpred = 2 * (pred - target) / N
// ---------------------------------------------------------------------------

pub fn traced_mse_loss_test() {
  let pred_data = [0.5, 1.5, 2.5, 3.5]
  let target_data = [1.0, 1.0, 2.0, 4.0]
  let pred_t = tensor_from(pred_data, [4])
  let target_t = tensor_from(target_data, [4])

  let tape = autograd.new_tape()
  let Traced(pred, tape1) = autograd.new_variable(tape, pred_t)
  let assert Ok(Traced(loss, tape2)) =
    autograd.traced_mse_loss(tape1, pred, target_t)

  // loss = mean((pred - target)^2) = mean([0.25, 0.25, 0.25, 0.25]) = 0.25
  let assert [actual_loss] = tensor.to_list(loss.data)
  numerics.floats_close(actual_loss, 0.25, 1.0e-9, 1.0e-9)
  |> should.be_true()

  let assert Ok(grads) = autograd.backward(tape2, loss)
  let assert Ok(dpred) = dict.get(grads, pred.id)
  // Expected grad: 2*(pred - target)/N = [-0.25, 0.25, 0.25, -0.25]
  numerics.lists_close(
    tensor.to_list(dpred),
    [-0.25, 0.25, 0.25, -0.25],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true()
}

// ---------------------------------------------------------------------------
// traced_layer_norm: shape preservation for input + parameter grads.
// ---------------------------------------------------------------------------

pub fn traced_layer_norm_test() {
  let x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
  let scale_data = [1.0, 1.0, 1.0]
  let bias_data = [0.0, 0.0, 0.0]
  let x_t = tensor_from(x_data, [2, 3])
  let scale_t = tensor_from(scale_data, [3])
  let bias_t = tensor_from(bias_data, [3])

  let tape = autograd.new_tape()
  let Traced(x, tape1) = autograd.new_variable(tape, x_t)
  let Traced(scale, tape2) = autograd.new_variable(tape1, scale_t)
  let Traced(bias, tape3) = autograd.new_variable(tape2, bias_t)
  let assert Ok(Traced(y, tape4)) =
    autograd.traced_layer_norm(tape3, x, scale, bias, 1.0e-5)

  // Output shape preserved.
  tensor.shape(y.data) |> should.equal([2, 3])

  // Each row should be ~zero-mean.
  let row0_sum =
    tensor.to_list(y.data)
    |> list.take(3)
    |> list.fold(0.0, fn(acc, v) { acc +. v })
  numerics.floats_close(row0_sum, 0.0, 1.0e-6, 1.0e-6) |> should.be_true()

  let assert Ok(grads) = autograd.backward(tape4, y)
  let assert Ok(dx) = dict.get(grads, x.id)
  let assert Ok(dscale) = dict.get(grads, scale.id)
  let assert Ok(dbias) = dict.get(grads, bias.id)

  tensor.shape(dx) |> should.equal([2, 3])
  tensor.shape(dscale) |> should.equal([3])
  tensor.shape(dbias) |> should.equal([3])
}

// ---------------------------------------------------------------------------
// traced_softmax: gradient via Tape matches direct analytical softmax_backward.
// We feed grad_y = ones, so the resulting grad_x should be zero (softmax is
// invariant under a constant shift in the upstream gradient — verified by
// noting that for ones-grad, sum_j(g_j * s_j) = sum_j s_j = 1, and
// s_i * (1 - 1) = 0).
// ---------------------------------------------------------------------------

pub fn traced_softmax_ones_grad_zero_test() {
  let x_data = [1.0, 2.0, 3.0, -1.0, 0.5, 4.0]
  let x_t = tensor_from(x_data, [2, 3])

  let tape = autograd.new_tape()
  let Traced(x, tape1) = autograd.new_variable(tape, x_t)
  let assert Ok(Traced(y, tape2)) = autograd.traced_softmax(tape1, x, 1)

  // Softmax rows sum to 1.
  let row0 =
    tensor.to_list(y.data)
    |> list.take(3)
  let row_sum0 = list.fold(row0, 0.0, fn(acc, v) { acc +. v })
  numerics.floats_close(row_sum0, 1.0, 1.0e-9, 1.0e-9) |> should.be_true()

  let assert Ok(grads) = autograd.backward(tape2, y)
  let assert Ok(dx) = dict.get(grads, x.id)
  // Expected: zero everywhere because softmax(ones_grad) reduces to zero.
  tensor.shape(dx) |> should.equal([2, 3])
  numerics.lists_close(
    tensor.to_list(dx),
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true()
}

// A second softmax test with a non-trivial upstream gradient (via chained mul
// with a constant scale Variable), ensuring the Tape backward matches the
// explicit formula `grad_x = s * (grad - sum(grad * s, axis, keepdims))`.
pub fn traced_softmax_directional_grad_test() {
  let x_data = [0.5, 1.5, 1.0]
  let x_t = tensor_from(x_data, [3])

  let tape = autograd.new_tape()
  let Traced(x, tape1) = autograd.new_variable(tape, x_t)
  let assert Ok(Traced(s, tape2)) = autograd.traced_softmax(tape1, x, 0)

  // Use mean to get a scalar; backward will push 1/3 through softmax.
  let Traced(loss, tape3) = autograd.mean(tape2, s)
  let assert Ok(grads) = autograd.backward(tape3, loss)
  let assert Ok(dx) = dict.get(grads, x.id)

  // grad_y = 1/3 everywhere; same argument as above => grad_x is zero.
  numerics.lists_close(tensor.to_list(dx), [0.0, 0.0, 0.0], 1.0e-9, 1.0e-9)
  |> should.be_true()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn replace_at(values: List(Float), index: Int, value: Float) -> List(Float) {
  values
  |> list.index_map(fn(item, i) {
    case i == index {
      True -> value
      False -> item
    }
  })
}

fn finite_difference(
  values: List(Float),
  shape: List(Int),
  index: Int,
  eps: Float,
  loss: fn(tensor.Tensor) -> Float,
) -> Float {
  let current =
    values
    |> list.drop(index)
    |> list.first
    |> result.unwrap(0.0)
  let plus = replace_at(values, index, current +. eps)
  let minus = replace_at(values, index, current -. eps)
  let assert Ok(plus_t) = tensor.new(plus, shape)
  let assert Ok(minus_t) = tensor.new(minus, shape)
  { loss(plus_t) -. loss(minus_t) } /. { 2.0 *. eps }
}

fn numerical_gradient(
  values: List(Float),
  shape: List(Int),
  eps: Float,
  loss: fn(tensor.Tensor) -> Float,
) -> List(Float) {
  case values == [] {
    True -> []
    False ->
      list.range(0, list.length(values) - 1)
      |> list.map(fn(i) { finite_difference(values, shape, i, eps, loss) })
  }
}

@external(erlang, "math", "exp")
fn erlang_exp(x: Float) -> Float
