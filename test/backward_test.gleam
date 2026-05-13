//// Numerical gradient tests for `viva_tensor/nn/backward`.
////
//// Each `_backward` function is verified against central-difference finite
//// differences:
////
////   numerical = (L(x + eps) - L(x - eps)) / (2 * eps)
////   analytical = backward(grad_out=ones_like(output), ...)(x)
////
//// where `L(x) = sum(forward(x))` so `dL/dx_i` matches what a sum reduction
//// of the upstream gradient would produce. Tolerances follow PyTorch's
//// `gradcheck` defaults (`rtol = 1e-3`, `atol = 1e-4`) — finite differences
//// of pure-Gleam floats lose precision pretty quickly.

import gleam/float
import gleam/int
import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/nn/activations
import viva_tensor/nn/backward
import viva_tensor/nn/losses

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// HELPERS
// =============================================================================

const eps: Float = 1.0e-4

const rtol: Float = 1.0e-3

const atol: Float = 1.0e-4

/// Build a 1D tensor.
fn vec(xs: List(Float)) -> t.Tensor {
  t.from_list(xs)
}

/// Build a 2D tensor from row-major flat data + dims.
fn mat2(rows: Int, cols: Int, data: List(Float)) -> t.Tensor {
  let assert Ok(m) = t.matrix(rows, cols, data)
  m
}

/// Sum of all elements (used as scalar loss for numerical checks).
fn sum_tensor(x: t.Tensor) -> Float {
  t.sum(x)
}

/// Numerical gradient of `f` at vector `x` via central differences.
fn numerical_grad_1d(
  x: List(Float),
  f: fn(List(Float)) -> Float,
) -> List(Float) {
  list.index_map(x, fn(_, i) {
    let plus = perturb(x, i, eps)
    let minus = perturb(x, i, -1.0 *. eps)
    { f(plus) -. f(minus) } /. { 2.0 *. eps }
  })
}

fn perturb(x: List(Float), index: Int, delta: Float) -> List(Float) {
  list.index_map(x, fn(v, i) {
    case i == index {
      True -> v +. delta
      False -> v
    }
  })
}

fn ones_like(x: t.Tensor) -> t.Tensor {
  t.ones(t.shape(x))
}

// =============================================================================
// ACTIVATION BACKWARDS
// =============================================================================

pub fn relu_backward_test() {
  let x = vec([-1.5, -0.3, 0.4, 1.2, 2.5])
  let g_out = ones_like(x)
  let assert Ok(g_in) = backward.relu_backward(g_out, x)
  let f = fn(xs: List(Float)) -> Float { sum_tensor(activations.relu(vec(xs))) }
  let num = numerical_grad_1d(t.to_list(x), f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}

pub fn relu_backward_at_zero_test() {
  // Subgradient at x = 0 is 0 (PyTorch convention).
  let x = vec([0.0, 0.0, 1.0])
  let g_out = vec([1.0, 1.0, 1.0])
  let assert Ok(g_in) = backward.relu_backward(g_out, x)
  case t.to_list(g_in) {
    [a, b, c] -> {
      numerics.floats_close(a, 0.0, rtol, atol) |> should.be_true
      numerics.floats_close(b, 0.0, rtol, atol) |> should.be_true
      numerics.floats_close(c, 1.0, rtol, atol) |> should.be_true
    }
    _ -> should.fail()
  }
}

pub fn sigmoid_backward_test() {
  let x = vec([-1.5, -0.3, 0.4, 1.2, 2.5])
  let y = activations.sigmoid(x)
  let g_out = ones_like(x)
  let assert Ok(g_in) = backward.sigmoid_backward(g_out, y)
  let f = fn(xs: List(Float)) -> Float {
    sum_tensor(activations.sigmoid(vec(xs)))
  }
  let num = numerical_grad_1d(t.to_list(x), f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}

pub fn tanh_backward_test() {
  let x = vec([-1.2, -0.4, 0.0, 0.6, 1.3])
  let y = activations.tanh(x)
  let g_out = ones_like(x)
  let assert Ok(g_in) = backward.tanh_backward(g_out, y)
  let f = fn(xs: List(Float)) -> Float { sum_tensor(activations.tanh(vec(xs))) }
  let num = numerical_grad_1d(t.to_list(x), f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}

pub fn gelu_backward_test() {
  let x = vec([-1.5, -0.5, 0.2, 0.8, 1.7])
  let g_out = ones_like(x)
  let assert Ok(g_in) = backward.gelu_backward(g_out, x)
  let f = fn(xs: List(Float)) -> Float { sum_tensor(activations.gelu(vec(xs))) }
  let num = numerical_grad_1d(t.to_list(x), f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}

pub fn leaky_relu_backward_test() {
  let slope = 0.1
  // Avoid x = 0 because central-difference straddles the slope discontinuity
  // and would disagree with the analytical subgradient there.
  let x = vec([-2.0, -0.5, 0.3, 0.4, 1.6])
  let g_out = ones_like(x)
  let assert Ok(g_in) = backward.leaky_relu_backward(g_out, x, slope)
  let f = fn(xs: List(Float)) -> Float {
    sum_tensor(activations.leaky_relu(vec(xs), slope))
  }
  let num = numerical_grad_1d(t.to_list(x), f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}

pub fn elu_backward_test() {
  let alpha = 1.0
  let x = vec([-1.5, -0.5, 0.2, 0.8, 1.7])
  let g_out = ones_like(x)
  let assert Ok(g_in) = backward.elu_backward(g_out, x, alpha)
  let f = fn(xs: List(Float)) -> Float {
    sum_tensor(activations.elu(vec(xs), alpha))
  }
  let num = numerical_grad_1d(t.to_list(x), f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}

// =============================================================================
// LOSS BACKWARDS
// =============================================================================

pub fn mse_loss_backward_test() {
  let pred = vec([1.0, 2.0, 3.0, 4.0])
  let target = vec([1.5, 1.8, 3.2, 3.7])
  let g_out = vec([1.0])
  let assert Ok(g_pred) =
    backward.mse_loss_backward(g_out, pred, target, losses.ReductionMean)
  let f = fn(xs: List(Float)) -> Float {
    let assert Ok(loss) = losses.mse_loss(vec(xs), target, losses.ReductionMean)
    sum_tensor(loss)
  }
  let num = numerical_grad_1d(t.to_list(pred), f)
  numerics.lists_close(t.to_list(g_pred), num, rtol, atol)
  |> should.be_true
}

pub fn l1_loss_backward_test() {
  let pred = vec([1.0, 2.5, -0.7, 4.0])
  let target = vec([1.5, 2.0, -1.2, 3.7])
  let g_out = vec([1.0])
  let assert Ok(g_pred) =
    backward.l1_loss_backward(g_out, pred, target, losses.ReductionMean)
  let f = fn(xs: List(Float)) -> Float {
    let assert Ok(loss) = losses.l1_loss(vec(xs), target, losses.ReductionMean)
    sum_tensor(loss)
  }
  let num = numerical_grad_1d(t.to_list(pred), f)
  numerics.lists_close(t.to_list(g_pred), num, rtol, atol)
  |> should.be_true
}

pub fn bce_loss_backward_test() {
  // Keep predictions strictly inside (eps, 1-eps) for a smooth gradient.
  let pred = vec([0.2, 0.4, 0.7, 0.9])
  let target = vec([0.0, 1.0, 1.0, 0.0])
  let g_out = vec([1.0])
  let assert Ok(g_pred) =
    backward.bce_loss_backward(g_out, pred, target, losses.ReductionMean)
  let f = fn(xs: List(Float)) -> Float {
    let assert Ok(loss) = losses.bce_loss(vec(xs), target, losses.ReductionMean)
    sum_tensor(loss)
  }
  let num = numerical_grad_1d(t.to_list(pred), f)
  numerics.lists_close(t.to_list(g_pred), num, rtol, atol)
  |> should.be_true
}

pub fn cross_entropy_loss_backward_test() {
  let logits = mat2(2, 3, [2.0, 1.0, 0.5, 0.3, 2.0, 1.5])
  let targets = vec([0.0, 1.0])
  let g_out = vec([1.0])
  let assert Ok(g_logits) =
    backward.cross_entropy_loss_backward(
      g_out,
      logits,
      targets,
      losses.ReductionMean,
    )
  t.shape(g_logits) |> should.equal([2, 3])

  // Numerical: perturb every logit cell, run forward (mean reduction).
  let logits_flat = t.to_list(logits)
  let f = fn(xs: List(Float)) -> Float {
    let perturbed = mat2(2, 3, xs)
    let assert Ok(loss) =
      losses.cross_entropy_loss(perturbed, targets, losses.ReductionMean)
    sum_tensor(loss)
  }
  let num = numerical_grad_1d(logits_flat, f)
  numerics.lists_close(t.to_list(g_logits), num, rtol, atol)
  |> should.be_true
}

pub fn cross_entropy_softmax_invariance_test() {
  // Adding a constant to all logits doesn't change softmax or its loss.
  let logits = mat2(2, 3, [2.0, 1.0, 0.5, 0.3, 2.0, 1.5])
  let shifted = mat2(2, 3, [12.0, 11.0, 10.5, 10.3, 12.0, 11.5])
  let targets = vec([0.0, 1.0])

  let assert Ok(loss_a) =
    losses.cross_entropy_loss(logits, targets, losses.ReductionMean)
  let assert Ok(loss_b) =
    losses.cross_entropy_loss(shifted, targets, losses.ReductionMean)
  let assert [la] = t.to_list(loss_a)
  let assert [lb] = t.to_list(loss_b)
  numerics.floats_close(la, lb, rtol, atol) |> should.be_true

  let g_out = vec([1.0])
  let assert Ok(g_a) =
    backward.cross_entropy_loss_backward(
      g_out,
      logits,
      targets,
      losses.ReductionMean,
    )
  let assert Ok(g_b) =
    backward.cross_entropy_loss_backward(
      g_out,
      shifted,
      targets,
      losses.ReductionMean,
    )
  numerics.lists_close(t.to_list(g_a), t.to_list(g_b), rtol, atol)
  |> should.be_true
}

// =============================================================================
// LINEAR / MATMUL
// =============================================================================

pub fn linear_backward_test() {
  // input [2, 3], weight [3, 2] -> output [2, 2]
  let input = mat2(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let weight = mat2(3, 2, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
  let assert Ok(output) = t.matmul(input, weight)
  let g_out = ones_like(output)
  let assert Ok(#(g_input, g_weight)) =
    backward.linear_backward(g_out, input, weight)

  // Numerical for input
  let input_flat = t.to_list(input)
  let f_input = fn(xs: List(Float)) -> Float {
    let p = mat2(2, 3, xs)
    let assert Ok(out) = t.matmul(p, weight)
    sum_tensor(out)
  }
  let num_input = numerical_grad_1d(input_flat, f_input)
  numerics.lists_close(t.to_list(g_input), num_input, rtol, atol)
  |> should.be_true

  // Numerical for weight
  let weight_flat = t.to_list(weight)
  let f_weight = fn(xs: List(Float)) -> Float {
    let p = mat2(3, 2, xs)
    let assert Ok(out) = t.matmul(input, p)
    sum_tensor(out)
  }
  let num_weight = numerical_grad_1d(weight_flat, f_weight)
  numerics.lists_close(t.to_list(g_weight), num_weight, rtol, atol)
  |> should.be_true
}

pub fn linear_backward_shape_test() {
  // grad_input shape == input shape, grad_weight shape == weight shape.
  let input = mat2(4, 5, list.repeat(0.5, 20))
  let weight = mat2(5, 3, list.repeat(0.1, 15))
  let assert Ok(output) = t.matmul(input, weight)
  let g_out = ones_like(output)
  let assert Ok(#(g_input, g_weight)) =
    backward.linear_backward(g_out, input, weight)
  t.shape(g_input) |> should.equal(t.shape(input))
  t.shape(g_weight) |> should.equal(t.shape(weight))
}

pub fn matmul_backward_test() {
  let a = mat2(2, 2, [1.0, 2.0, 3.0, 4.0])
  let b = mat2(2, 2, [0.5, -0.5, 1.0, 0.5])
  let assert Ok(c) = t.matmul(a, b)
  let g_out = ones_like(c)
  let assert Ok(#(g_a, g_b)) = backward.matmul_backward(g_out, a, b)

  let a_flat = t.to_list(a)
  let f_a = fn(xs: List(Float)) -> Float {
    let p = mat2(2, 2, xs)
    let assert Ok(o) = t.matmul(p, b)
    sum_tensor(o)
  }
  let num_a = numerical_grad_1d(a_flat, f_a)
  numerics.lists_close(t.to_list(g_a), num_a, rtol, atol)
  |> should.be_true

  let b_flat = t.to_list(b)
  let f_b = fn(xs: List(Float)) -> Float {
    let p = mat2(2, 2, xs)
    let assert Ok(o) = t.matmul(a, p)
    sum_tensor(o)
  }
  let num_b = numerical_grad_1d(b_flat, f_b)
  numerics.lists_close(t.to_list(g_b), num_b, rtol, atol)
  |> should.be_true
}

// =============================================================================
// NORM BACKWARDS
// =============================================================================

// Forward helper that recomputes mean/var inside, so finite-difference works.
fn layer_norm_forward_scalar(
  xs: List(Float),
  _rows: Int,
  cols: Int,
  scale: List(Float),
  bias: List(Float),
  eps_v: Float,
) -> Float {
  let chunks = chunk_every(xs, cols)
  let normalized =
    list.flat_map(chunks, fn(row) {
      let mu = mean_of(row)
      let var = variance_of(row, mu)
      let std = safe_sqrt(var +. eps_v)
      list.map(list.zip(row, list.zip(scale, bias)), fn(t) {
        let #(x, sb) = t
        let #(s, b) = sb
        { x -. mu } /. std *. s +. b
      })
    })
  list.fold(normalized, 0.0, fn(a, x) { a +. x })
}

fn mean_of(xs: List(Float)) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ -> list.fold(xs, 0.0, fn(a, x) { a +. x }) /. int_to_float(n)
  }
}

fn variance_of(xs: List(Float), mean: Float) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ ->
      list.fold(xs, 0.0, fn(a, x) {
        let d = x -. mean
        a +. d *. d
      })
      /. int_to_float(n)
  }
}

fn safe_sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn int_to_float(n: Int) -> Float {
  int.to_float(n)
}

fn chunk_every(items: List(Float), n: Int) -> List(List(Float)) {
  case items {
    [] -> []
    _ -> {
      let head = list.take(items, n)
      let rest = list.drop(items, n)
      [head, ..chunk_every(rest, n)]
    }
  }
}

pub fn layer_norm_backward_test() {
  let rows = 2
  let cols = 4
  let input = mat2(rows, cols, [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 1.5])
  let scale = vec([1.1, 0.9, 1.2, 0.8])
  let bias = vec([0.1, -0.1, 0.2, 0.0])
  let eps_v = 1.0e-5

  // Compute per-row mean/var to pass to backward.
  let row_chunks = chunk_every(t.to_list(input), cols)
  let means = list.map(row_chunks, mean_of)
  let vars = list.map(row_chunks, fn(r) { variance_of(r, mean_of(r)) })
  let mean_t = vec(means)
  let var_t = vec(vars)

  // Build the analytical gradient with grad_out = ones (so sum reduces to identity).
  let g_out = ones_like(input)
  let assert Ok(#(g_input, g_scale, g_bias)) =
    backward.layer_norm_backward(g_out, input, scale, mean_t, var_t, eps_v)

  // Sanity shape checks first.
  t.shape(g_input) |> should.equal([rows, cols])
  t.shape(g_scale) |> should.equal([cols])
  t.shape(g_bias) |> should.equal([cols])

  // Numerical gradient w.r.t. input.
  let input_flat = t.to_list(input)
  let scale_flat = t.to_list(scale)
  let bias_flat = t.to_list(bias)
  let f_input = fn(xs: List(Float)) -> Float {
    layer_norm_forward_scalar(xs, rows, cols, scale_flat, bias_flat, eps_v)
  }
  let num_input = numerical_grad_1d(input_flat, f_input)
  numerics.lists_close(t.to_list(g_input), num_input, rtol, atol)
  |> should.be_true

  // Numerical gradient w.r.t. scale: scale is [cols], perturb each.
  let f_scale = fn(s: List(Float)) -> Float {
    layer_norm_forward_scalar(input_flat, rows, cols, s, bias_flat, eps_v)
  }
  let num_scale = numerical_grad_1d(scale_flat, f_scale)
  numerics.lists_close(t.to_list(g_scale), num_scale, rtol, atol)
  |> should.be_true

  // Numerical gradient w.r.t. bias.
  let f_bias = fn(b: List(Float)) -> Float {
    layer_norm_forward_scalar(input_flat, rows, cols, scale_flat, b, eps_v)
  }
  let num_bias = numerical_grad_1d(bias_flat, f_bias)
  numerics.lists_close(t.to_list(g_bias), num_bias, rtol, atol)
  |> should.be_true
}

// RMS-norm forward as a scalar loss for finite-diff.
fn rms_norm_forward_scalar(
  xs: List(Float),
  cols: Int,
  scale: List(Float),
  eps_v: Float,
) -> Float {
  let chunks = chunk_every(xs, cols)
  let normalized =
    list.flat_map(chunks, fn(row) {
      let ms =
        list.fold(row, 0.0, fn(a, x) { a +. x *. x })
        /. int_to_float(list.length(row))
      let rms = safe_sqrt(ms +. eps_v)
      list.map(list.zip(row, scale), fn(p) {
        let #(x, s) = p
        x /. rms *. s
      })
    })
  list.fold(normalized, 0.0, fn(a, x) { a +. x })
}

pub fn rms_norm_backward_test() {
  let rows = 2
  let cols = 4
  let input = mat2(rows, cols, [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.5, 1.5])
  let scale = vec([1.1, 0.9, 1.2, 0.8])
  let eps_v = 1.0e-6

  // Compute per-row rms to pass to backward.
  let row_chunks = chunk_every(t.to_list(input), cols)
  let rmses =
    list.map(row_chunks, fn(r) {
      let ms =
        list.fold(r, 0.0, fn(a, x) { a +. x *. x })
        /. int_to_float(list.length(r))
      safe_sqrt(ms +. eps_v)
    })
  let rms_t = vec(rmses)

  let g_out = ones_like(input)
  let assert Ok(#(g_input, g_scale)) =
    backward.rms_norm_backward(g_out, input, scale, rms_t, eps_v)
  t.shape(g_input) |> should.equal([rows, cols])
  t.shape(g_scale) |> should.equal([cols])

  let input_flat = t.to_list(input)
  let scale_flat = t.to_list(scale)
  let f_input = fn(xs: List(Float)) -> Float {
    rms_norm_forward_scalar(xs, cols, scale_flat, eps_v)
  }
  let num_input = numerical_grad_1d(input_flat, f_input)
  numerics.lists_close(t.to_list(g_input), num_input, rtol, atol)
  |> should.be_true

  let f_scale = fn(s: List(Float)) -> Float {
    rms_norm_forward_scalar(input_flat, cols, s, eps_v)
  }
  let num_scale = numerical_grad_1d(scale_flat, f_scale)
  numerics.lists_close(t.to_list(g_scale), num_scale, rtol, atol)
  |> should.be_true
}

// =============================================================================
// SOFTMAX BACKWARD
// =============================================================================

pub fn softmax_backward_test() {
  let logits = mat2(2, 3, [1.0, 2.0, 3.0, 0.5, -0.5, 1.5])
  let assert Ok(probs) = t.softmax(logits, 1)
  let g_out = ones_like(probs)
  let assert Ok(g_in) = backward.softmax_backward(g_out, probs, 1)

  // Numerical: perturb each cell of logits, run softmax forward, sum result.
  let logits_flat = t.to_list(logits)
  let f = fn(xs: List(Float)) -> Float {
    let p = mat2(2, 3, xs)
    let assert Ok(s) = t.softmax(p, 1)
    sum_tensor(s)
  }
  let num = numerical_grad_1d(logits_flat, f)
  numerics.lists_close(t.to_list(g_in), num, rtol, atol)
  |> should.be_true
}
