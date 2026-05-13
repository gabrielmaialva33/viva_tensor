import gleam/dict
import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor/core/error.{ShapeMismatch}
import viva_tensor/nn/optim.{
  type Param, AdamState, GradPair, MomentumState, Param, RmspropState,
}
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

// --- Helpers ----------------------------------------------------------------

fn close(actual: Float, expected: Float, tol: Float) -> Bool {
  float.absolute_value(actual -. expected) <. tol
}

fn list_close(actual: List(Float), expected: List(Float), tol: Float) -> Bool {
  list.length(actual) == list.length(expected)
  && list.zip(actual, expected)
  |> list.all(fn(pair) { close(pair.0, pair.1, tol) })
}

fn param_data(param: Param) -> List(Float) {
  tensor.to_list(param.value)
}

fn first_param(params: List(Param)) -> Param {
  let assert [p, ..] = params
  p
}

fn pow_int(base: Float, exp: Int) -> Float {
  case exp {
    0 -> 1.0
    n if n > 0 -> base *. pow_int(base, n - 1)
    _ -> 1.0
  }
}

// --- SGD --------------------------------------------------------------------

pub fn sgd_single_step_test() {
  let opt = optim.sgd(0.1)
  let p = Param("w", tensor.from_list([1.0, 2.0]))
  let g = GradPair("w", tensor.from_list([0.5, -0.5]))

  let assert Ok(#(_opt2, params2)) = optim.step(opt, [p], [g])
  let result = param_data(first_param(params2))

  list_close(result, [0.95, 2.05], 1.0e-9)
  |> should.be_true()
}

pub fn sgd_momentum_two_steps_test() {
  // momentum=0.9, lr=0.1, constant grad g=[1.0]
  // step 1: v = 0 + g = [1.0],  θ = θ - 0.1 * v = [0.9]
  // step 2: v = 0.9*1 + 1.0 = 1.9, θ = 0.9 - 0.1*1.9 = 0.71
  let opt = optim.sgd_momentum(0.1, 0.9)
  let p = Param("w", tensor.from_list([1.0]))
  let g = GradPair("w", tensor.from_list([1.0]))

  let assert Ok(#(opt2, params2)) = optim.step(opt, [p], [g])
  let after1 = param_data(first_param(params2))
  list_close(after1, [0.9], 1.0e-9)
  |> should.be_true()

  // Velocity in state must be [1.0].
  let assert Ok(MomentumState(v1)) = dict.get(opt2.state, "w")
  list_close(tensor.to_list(v1), [1.0], 1.0e-9)
  |> should.be_true()

  let assert Ok(#(_opt3, params3)) = optim.step(opt2, params2, [g])
  let after2 = param_data(first_param(params3))
  list_close(after2, [0.71], 1.0e-9)
  |> should.be_true()
}

// --- RMSprop ----------------------------------------------------------------

pub fn rmsprop_step_test() {
  // alpha=0.9, lr=0.1, eps=1e-8, constant grad g=[1.0]
  // step 1: s = 0.9*0 + 0.1*1 = 0.1,  ratio = 1 / (sqrt(0.1)+eps),
  //          θ = 1 - 0.1 * ratio
  // step 2: s = 0.9*0.1 + 0.1*1 = 0.19, ratio = 1 / (sqrt(0.19)+eps)
  let alpha = 0.9
  let lr = 0.1
  let eps = 1.0e-8
  let opt = optim.rmsprop(lr, alpha, eps)
  let p = Param("w", tensor.from_list([1.0]))
  let g = GradPair("w", tensor.from_list([1.0]))

  let assert Ok(#(opt2, params2)) = optim.step(opt, [p], [g])
  let assert Ok(s1_sqrt) = float.square_root(0.1)
  let expected_after1 = 1.0 -. lr *. 1.0 /. { s1_sqrt +. eps }
  let after1 = param_data(first_param(params2))
  list_close(after1, [expected_after1], 1.0e-6)
  |> should.be_true()

  // square_avg in state should be ~0.1.
  let assert Ok(RmspropState(s_after1)) = dict.get(opt2.state, "w")
  list_close(tensor.to_list(s_after1), [0.1], 1.0e-9)
  |> should.be_true()

  let assert Ok(#(opt3, params3)) = optim.step(opt2, params2, [g])
  let assert Ok(RmspropState(s_after2)) = dict.get(opt3.state, "w")
  list_close(tensor.to_list(s_after2), [0.19], 1.0e-9)
  |> should.be_true()

  let assert Ok(s2_sqrt) = float.square_root(0.19)
  let expected_after2 = expected_after1 -. lr *. 1.0 /. { s2_sqrt +. eps }
  let after2 = param_data(first_param(params3))
  list_close(after2, [expected_after2], 1.0e-6)
  |> should.be_true()
}

// --- Adam -------------------------------------------------------------------

pub fn adam_first_step_test() {
  // Default Adam: beta1=0.9, beta2=0.999, eps=1e-8, lr=0.1
  // grad g=[1.0]
  // m = 0.1 * g = 0.1,  v = 0.001 * g^2 = 0.001
  // m_hat = m / (1 - 0.9^1) = 0.1 / 0.1 = 1.0 = g  (== g / (1 - beta1)*... no: == g)
  // v_hat = v / (1 - 0.999^1) = 0.001 / 0.001 = 1.0
  // step = lr * m_hat / (sqrt(v_hat) + eps) = 0.1 * 1 / (1 + 1e-8) ~ 0.1
  let lr = 0.1
  let opt = optim.adam(lr)
  let p = Param("w", tensor.from_list([1.0]))
  let g = GradPair("w", tensor.from_list([1.0]))

  let assert Ok(#(opt2, params2)) = optim.step(opt, [p], [g])

  // Bias correction property: m_hat at t=1 should equal grad.
  let assert Ok(AdamState(m, v, step)) = dict.get(opt2.state, "w")
  step
  |> should.equal(1)

  let m_val = result_unwrap_float(list.first(tensor.to_list(m)))
  let m_hat = m_val /. { 1.0 -. pow_int(0.9, 1) }
  close(m_hat, 1.0, 1.0e-9)
  |> should.be_true()

  let v_val = result_unwrap_float(list.first(tensor.to_list(v)))
  let v_hat = v_val /. { 1.0 -. pow_int(0.999, 1) }
  close(v_hat, 1.0, 1.0e-9)
  |> should.be_true()

  // Updated param should be 1.0 - lr * 1.0 / (1.0 + 1e-8) ~ 0.9.
  let expected = 1.0 -. lr *. 1.0 /. { 1.0 +. 1.0e-8 }
  let after = param_data(first_param(params2))
  list_close(after, [expected], 1.0e-7)
  |> should.be_true()
}

pub fn adam_two_steps_test() {
  // Verify EMA compounding: after two steps with grad g=1.0,
  // m_2 = beta1 * m_1 + (1 - beta1) * g
  //     = 0.9 * 0.1 + 0.1 * 1.0 = 0.19
  // v_2 = beta2 * v_1 + (1 - beta2) * g^2
  //     = 0.999 * 0.001 + 0.001 * 1.0 = 0.001999
  let opt = optim.adam(0.1)
  let p = Param("w", tensor.from_list([1.0]))
  let g = GradPair("w", tensor.from_list([1.0]))

  let assert Ok(#(opt2, params2)) = optim.step(opt, [p], [g])
  let assert Ok(#(opt3, _params3)) = optim.step(opt2, params2, [g])

  let assert Ok(AdamState(m, v, step)) = dict.get(opt3.state, "w")
  step
  |> should.equal(2)

  list_close(tensor.to_list(m), [0.19], 1.0e-9)
  |> should.be_true()
  list_close(tensor.to_list(v), [0.001999], 1.0e-9)
  |> should.be_true()
}

// --- AdamW ------------------------------------------------------------------

pub fn adamw_two_steps_test() {
  // AdamW: param -= lr * wd * param BEFORE the gradient step.
  // lr=0.1, weight_decay=0.5, beta1=0.9, beta2=0.999, eps=1e-8, grad=[1.0]
  // Step 1:
  //   decay step:    θ' = 1.0 - 0.1 * 0.5 * 1.0 = 0.95
  //   grad step:     θ = 0.95 - 0.1 * 1.0 / (1.0 + 1e-8) ~= 0.85
  // Step 2:
  //   decay step:    θ' = 0.85 - 0.1 * 0.5 * 0.85 = 0.85 * 0.95 = 0.8075
  //   m_2 = 0.19, v_2 = 0.001999
  //   m_hat = 0.19 / (1 - 0.81) = 1.0
  //   v_hat = 0.001999 / (1 - 0.998001) ~= 1.0005 (close to 1.0)
  //   grad step:     θ = 0.8075 - 0.1 * m_hat / (sqrt(v_hat) + eps)
  let lr = 0.1
  let wd = 0.5
  let opt = optim.adamw(lr, wd)
  let p = Param("w", tensor.from_list([1.0]))
  let g = GradPair("w", tensor.from_list([1.0]))

  let assert Ok(#(opt2, params2)) = optim.step(opt, [p], [g])
  let after1 = param_data(first_param(params2))
  let expected1 = 1.0 -. lr *. wd *. 1.0 -. lr *. 1.0 /. { 1.0 +. 1.0e-8 }
  list_close(after1, [expected1], 1.0e-7)
  |> should.be_true()

  let assert Ok(#(_opt3, params3)) = optim.step(opt2, params2, [g])

  // After step 2 the EMAs reach the same values as plain Adam:
  // m_2 = 0.19, v_2 = 0.001999
  let m_hat = 0.19 /. { 1.0 -. pow_int(0.9, 2) }
  let v_hat = 0.001999 /. { 1.0 -. pow_int(0.999, 2) }
  let assert Ok(sqrt_v_hat) = float.square_root(v_hat)
  let theta_after_decay = expected1 -. lr *. wd *. expected1
  let expected2 = theta_after_decay -. lr *. m_hat /. { sqrt_v_hat +. 1.0e-8 }
  let after2 = param_data(first_param(params3))
  list_close(after2, [expected2], 1.0e-6)
  |> should.be_true()
}

// --- Error / Misc -----------------------------------------------------------

pub fn step_returns_error_on_shape_mismatch_test() {
  let opt = optim.sgd(0.1)
  let p = Param("w", tensor.from_list([1.0, 2.0]))
  let g = GradPair("w", tensor.from_list([0.5]))

  case optim.step(opt, [p], [g]) {
    Error(ShapeMismatch(expected: [2], got: [1])) -> Nil
    _ -> panic as "expected ShapeMismatch([2], [1])"
  }
}

pub fn zero_grad_test() {
  let g = GradPair("w", tensor.from_list([0.5, -0.5, 1.0]))
  let zeroed = optim.zero_grad([g])
  let assert [first, ..] = zeroed
  tensor.shape(first.grad)
  |> should.equal([3])
  list_close(tensor.to_list(first.grad), [0.0, 0.0, 0.0], 1.0e-12)
  |> should.be_true()
}

// --- tiny helper ------------------------------------------------------------

fn result_unwrap_float(r: Result(Float, a)) -> Float {
  case r {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}
