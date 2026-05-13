//// Gradient-descent optimizers.
////
//// Pure Gleam, pure functions. No mutation, no autograd integration yet —
//// callers must compute gradients themselves and hand them to `step/3` as a
//// list of `GradPair(name, grad)` paired against a list of
//// `Param(name, value)`. The function returns a new `Optimizer` (with
//// updated per-parameter state) and the updated parameter list.
////
//// Real autograd integration is a follow-up: once `viva_tensor/nn/autograd`
//// exposes a parameter registry, `step` will pluck gradients straight from
//// the tape instead of receiving them as an argument.
////
//// References:
//// - Robbins & Monro (1951). Stochastic approximation. (SGD)
//// - Polyak (1964). Heavy ball method. (SGD + momentum)
//// - Tieleman & Hinton (2012). RMSprop. Coursera lecture 6e.
//// - Kingma & Ba (2015). "Adam: A Method for Stochastic Optimization."
////   https://arxiv.org/abs/1412.6980
//// - Loshchilov & Hutter (2019). "Decoupled Weight Decay Regularization." (AdamW)
////   https://arxiv.org/abs/1711.05101

import gleam/dict.{type Dict}
import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, DimensionError, ShapeMismatch}
import viva_tensor/tensor.{type Tensor}

// --- Public Types -----------------------------------------------------------

/// Optimizer family tag. Used by `step/3` to pick the update rule.
pub type OptimizerKind {
  Sgd
  SgdMomentum
  Rmsprop
  Adam
  Adamw
}

/// A named parameter tensor.
pub type Param {
  Param(name: String, value: Tensor)
}

/// A gradient paired with the name of the parameter it belongs to.
pub type GradPair {
  GradPair(name: String, grad: Tensor)
}

/// Per-parameter state. Different optimizer kinds populate different variants.
///
/// - `EmptyState` — vanilla SGD has no per-parameter state.
/// - `MomentumState` — running velocity buffer for SGD-with-momentum.
/// - `RmspropState` — running average of squared gradients.
/// - `AdamState` — first and second moment estimates plus the step counter
///   used for bias correction.
pub type ParamState {
  EmptyState
  MomentumState(velocity: Tensor)
  RmspropState(square_avg: Tensor)
  AdamState(m: Tensor, v: Tensor, step: Int)
}

/// Optimizer record. Carries hyperparameters and the full per-parameter state
/// dictionary, keyed by parameter name.
pub type Optimizer {
  Optimizer(
    kind: OptimizerKind,
    lr: Float,
    momentum: Float,
    beta1: Float,
    beta2: Float,
    eps: Float,
    weight_decay: Float,
    state: Dict(String, ParamState),
  )
}

// --- Constructors -----------------------------------------------------------

/// Vanilla stochastic gradient descent: `θ ← θ - lr * g`.
///
/// ## Example
///
/// ```gleam
/// let opt = optim.sgd(0.01)
/// ```
pub fn sgd(lr: Float) -> Optimizer {
  Optimizer(
    kind: Sgd,
    lr: lr,
    momentum: 0.0,
    beta1: 0.0,
    beta2: 0.0,
    eps: 0.0,
    weight_decay: 0.0,
    state: dict.new(),
  )
}

/// SGD with momentum (Polyak, 1964):
/// `v ← momentum * v + g`, `θ ← θ - lr * v`.
///
/// ## Example
///
/// ```gleam
/// let opt = optim.sgd_momentum(0.01, 0.9)
/// ```
pub fn sgd_momentum(lr: Float, momentum: Float) -> Optimizer {
  Optimizer(
    kind: SgdMomentum,
    lr: lr,
    momentum: momentum,
    beta1: 0.0,
    beta2: 0.0,
    eps: 0.0,
    weight_decay: 0.0,
    state: dict.new(),
  )
}

/// RMSprop (Tieleman & Hinton, 2012):
/// `s ← alpha * s + (1 - alpha) * g^2`,
/// `θ ← θ - lr * g / (sqrt(s) + eps)`.
///
/// `alpha` plays the same role as Adam's `beta2`; conventional value is 0.99.
///
/// ## Example
///
/// ```gleam
/// let opt = optim.rmsprop(0.001, 0.99, 1.0e-8)
/// ```
pub fn rmsprop(lr: Float, alpha: Float, eps: Float) -> Optimizer {
  Optimizer(
    kind: Rmsprop,
    lr: lr,
    momentum: 0.0,
    beta1: 0.0,
    beta2: alpha,
    eps: eps,
    weight_decay: 0.0,
    state: dict.new(),
  )
}

/// Adam (Kingma & Ba, 2015) with default `beta1=0.9, beta2=0.999, eps=1e-8`.
///
/// Per step `t`:
///   `m ← beta1 * m + (1 - beta1) * g`
///   `v ← beta2 * v + (1 - beta2) * g^2`
///   `m_hat = m / (1 - beta1^t)`
///   `v_hat = v / (1 - beta2^t)`
///   `θ ← θ - lr * m_hat / (sqrt(v_hat) + eps)`.
///
/// ## Example
///
/// ```gleam
/// let opt = optim.adam(0.001)
/// ```
pub fn adam(lr: Float) -> Optimizer {
  Optimizer(
    kind: Adam,
    lr: lr,
    momentum: 0.0,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1.0e-8,
    weight_decay: 0.0,
    state: dict.new(),
  )
}

/// AdamW (Loshchilov & Hutter, 2019). Same as Adam but with decoupled weight
/// decay applied BEFORE the gradient step:
///   `θ ← θ - lr * weight_decay * θ`
///   `θ ← θ - lr * m_hat / (sqrt(v_hat) + eps)`.
///
/// ## Example
///
/// ```gleam
/// let opt = optim.adamw(0.001, 0.01)
/// ```
pub fn adamw(lr: Float, weight_decay: Float) -> Optimizer {
  Optimizer(
    kind: Adamw,
    lr: lr,
    momentum: 0.0,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1.0e-8,
    weight_decay: weight_decay,
    state: dict.new(),
  )
}

// --- Public API -------------------------------------------------------------

/// Apply one optimizer step.
///
/// Validates that every gradient has a matching parameter by name and that
/// shapes line up. On success returns `#(updated_optimizer, updated_params)`.
///
/// Parameters are returned in the same order as the input `params` list.
///
/// ## Example
///
/// ```gleam
/// let p = optim.Param("w", tensor.from_list([1.0, 2.0]))
/// let g = optim.GradPair("w", tensor.from_list([0.5, -0.5]))
/// let opt = optim.sgd(0.1)
/// let assert Ok(#(_opt2, params2)) = optim.step(opt, [p], [g])
/// ```
pub fn step(
  opt: Optimizer,
  params: List(Param),
  grads: List(GradPair),
) -> Result(#(Optimizer, List(Param)), TensorError) {
  let grad_dict = grads_to_dict(grads)
  use _ <- result.try(validate_pairing(params, grad_dict))
  apply_updates(opt, params, grad_dict)
}

/// Return a list of `GradPair` with every gradient tensor replaced by zeros
/// of the same shape. Useful when callers reuse a gradient list as scratch
/// storage between iterations.
///
/// ## Example
///
/// ```gleam
/// let g = optim.GradPair("w", tensor.from_list([0.5, -0.5]))
/// let zeroed = optim.zero_grad([g])
/// ```
pub fn zero_grad(grads: List(GradPair)) -> List(GradPair) {
  list.map(grads, fn(gp) {
    GradPair(name: gp.name, grad: tensor.zeros_like(gp.grad))
  })
}

// --- Internals --------------------------------------------------------------

fn grads_to_dict(grads: List(GradPair)) -> Dict(String, Tensor) {
  list.fold(grads, dict.new(), fn(acc, gp) {
    dict.insert(acc, gp.name, gp.grad)
  })
}

fn validate_pairing(
  params: List(Param),
  grad_dict: Dict(String, Tensor),
) -> Result(Nil, TensorError) {
  let names = list.map(params, fn(p) { p.name })
  // Every grad must reference a known param name.
  let unknown =
    dict.keys(grad_dict)
    |> list.filter(fn(name) { !list.contains(names, name) })
  case unknown {
    [missing, ..] ->
      Error(DimensionError(
        "optim.step: gradient for unknown parameter '" <> missing <> "'",
      ))
    [] -> validate_shapes(params, grad_dict)
  }
}

fn validate_shapes(
  params: List(Param),
  grad_dict: Dict(String, Tensor),
) -> Result(Nil, TensorError) {
  case params {
    [] -> Ok(Nil)
    [p, ..rest] ->
      case dict.get(grad_dict, p.name) {
        Error(_) -> validate_shapes(rest, grad_dict)
        Ok(g) ->
          case tensor.shape(p.value) == tensor.shape(g) {
            False ->
              Error(ShapeMismatch(
                expected: tensor.shape(p.value),
                got: tensor.shape(g),
              ))
            True -> validate_shapes(rest, grad_dict)
          }
      }
  }
}

fn apply_updates(
  opt: Optimizer,
  params: List(Param),
  grad_dict: Dict(String, Tensor),
) -> Result(#(Optimizer, List(Param)), TensorError) {
  do_apply(opt, params, grad_dict, [])
}

fn do_apply(
  opt: Optimizer,
  params: List(Param),
  grad_dict: Dict(String, Tensor),
  acc: List(Param),
) -> Result(#(Optimizer, List(Param)), TensorError) {
  case params {
    [] -> Ok(#(opt, list.reverse(acc)))
    [p, ..rest] ->
      case dict.get(grad_dict, p.name) {
        // No grad supplied for this param → leave it untouched.
        Error(_) -> do_apply(opt, rest, grad_dict, [p, ..acc])
        Ok(g) -> {
          use #(opt2, new_value) <- result.try(update_param(opt, p, g))
          do_apply(opt2, rest, grad_dict, [
            Param(name: p.name, value: new_value),
            ..acc
          ])
        }
      }
  }
}

fn update_param(
  opt: Optimizer,
  param: Param,
  grad: Tensor,
) -> Result(#(Optimizer, Tensor), TensorError) {
  case opt.kind {
    Sgd -> sgd_update(opt, param, grad)
    SgdMomentum -> momentum_update(opt, param, grad)
    Rmsprop -> rmsprop_update(opt, param, grad)
    Adam -> adam_update(opt, param, grad, False)
    Adamw -> adam_update(opt, param, grad, True)
  }
}

// --- SGD --------------------------------------------------------------------

fn sgd_update(
  opt: Optimizer,
  param: Param,
  grad: Tensor,
) -> Result(#(Optimizer, Tensor), TensorError) {
  let scaled = tensor.scale(grad, opt.lr)
  use new_value <- result.try(tensor.sub(param.value, scaled))
  Ok(#(opt, new_value))
}

// --- SGD + momentum ---------------------------------------------------------

fn momentum_update(
  opt: Optimizer,
  param: Param,
  grad: Tensor,
) -> Result(#(Optimizer, Tensor), TensorError) {
  let prev_v = case dict.get(opt.state, param.name) {
    Ok(MomentumState(v)) -> v
    _ -> tensor.zeros_like(param.value)
  }
  // v ← momentum * v + g
  let momentum_term = tensor.scale(prev_v, opt.momentum)
  use new_v <- result.try(tensor.add(momentum_term, grad))
  // θ ← θ - lr * v
  let step_term = tensor.scale(new_v, opt.lr)
  use new_value <- result.try(tensor.sub(param.value, step_term))
  let new_state = dict.insert(opt.state, param.name, MomentumState(new_v))
  Ok(#(Optimizer(..opt, state: new_state), new_value))
}

// --- RMSprop ----------------------------------------------------------------

fn rmsprop_update(
  opt: Optimizer,
  param: Param,
  grad: Tensor,
) -> Result(#(Optimizer, Tensor), TensorError) {
  let prev_s = case dict.get(opt.state, param.name) {
    Ok(RmspropState(s)) -> s
    _ -> tensor.zeros_like(param.value)
  }
  let alpha = opt.beta2
  // s ← alpha * s + (1 - alpha) * g^2
  use g_sq <- result.try(tensor.mul(grad, grad))
  let s_term = tensor.scale(prev_s, alpha)
  let g_term = tensor.scale(g_sq, 1.0 -. alpha)
  use new_s <- result.try(tensor.add(s_term, g_term))
  // update = lr * g / (sqrt(s) + eps)
  let denom = tensor.map(new_s, fn(x) { float_sqrt(x) +. opt.eps })
  use ratio <- result.try(tensor.div(grad, denom))
  let update = tensor.scale(ratio, opt.lr)
  use new_value <- result.try(tensor.sub(param.value, update))
  let new_state = dict.insert(opt.state, param.name, RmspropState(new_s))
  Ok(#(Optimizer(..opt, state: new_state), new_value))
}

// --- Adam / AdamW -----------------------------------------------------------

fn adam_update(
  opt: Optimizer,
  param: Param,
  grad: Tensor,
  decoupled_decay: Bool,
) -> Result(#(Optimizer, Tensor), TensorError) {
  let #(prev_m, prev_v, prev_t) = case dict.get(opt.state, param.name) {
    Ok(AdamState(m, v, t)) -> #(m, v, t)
    _ -> #(tensor.zeros_like(param.value), tensor.zeros_like(param.value), 0)
  }
  let t = prev_t + 1
  // m ← beta1 * m + (1 - beta1) * g
  let m_term = tensor.scale(prev_m, opt.beta1)
  let m_grad = tensor.scale(grad, 1.0 -. opt.beta1)
  use new_m <- result.try(tensor.add(m_term, m_grad))
  // v ← beta2 * v + (1 - beta2) * g^2
  use g_sq <- result.try(tensor.mul(grad, grad))
  let v_term = tensor.scale(prev_v, opt.beta2)
  let v_grad = tensor.scale(g_sq, 1.0 -. opt.beta2)
  use new_v <- result.try(tensor.add(v_term, v_grad))
  // Bias correction.
  let bc1 = 1.0 -. pow_int(opt.beta1, t)
  let bc2 = 1.0 -. pow_int(opt.beta2, t)
  let m_hat = tensor.scale(new_m, 1.0 /. bc1)
  let v_hat = tensor.scale(new_v, 1.0 /. bc2)
  // step = lr * m_hat / (sqrt(v_hat) + eps)
  let denom = tensor.map(v_hat, fn(x) { float_sqrt(x) +. opt.eps })
  use ratio <- result.try(tensor.div(m_hat, denom))
  let step_t = tensor.scale(ratio, opt.lr)
  // AdamW: decoupled weight decay BEFORE the gradient step.
  use base <- result.try(case decoupled_decay {
    True -> {
      let decay = tensor.scale(param.value, opt.lr *. opt.weight_decay)
      tensor.sub(param.value, decay)
    }
    False -> Ok(param.value)
  })
  use new_value <- result.try(tensor.sub(base, step_t))
  let new_state = dict.insert(opt.state, param.name, AdamState(new_m, new_v, t))
  Ok(#(Optimizer(..opt, state: new_state), new_value))
}

// --- Math helpers -----------------------------------------------------------

fn float_sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn pow_int(base: Float, exp: Int) -> Float {
  case float.power(base, int.to_float(exp)) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}
