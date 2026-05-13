//// Parameter initialization helpers for neural network weights.
////
//// All functions in this module are **pure** in the Gleam sense (no
//// side-effect tracking), but the random initializers below rely on the
//// BEAM's `:rand` PRNG via `int.random`. That PRNG is per-process and
//// **non-deterministic across runs** — there is no portable seed argument
//// here. Tests that need reproducibility should be statistical (range,
//// approximate mean/std) rather than bit-for-bit.
////
//// Conventions:
//// - All weight-matrix initializers return shape `[fan_in, fan_out]` so they
////   compose cleanly with `matmul(input, weight)` where `input` has shape
////   `[batch, fan_in]`.
//// - `gain` arguments scale the resulting standard deviation. Use the
////   `*_gain` helpers below to pick the right value for your activation.
////
//// References:
//// - Glorot & Bengio (2010). "Understanding the difficulty of training deep
////   feedforward neural networks."
//// - He et al. (2015). "Delving deep into rectifiers: Surpassing human-level
////   performance on ImageNet classification."
//// - Saxe, McClelland & Ganguli (2014). "Exact solutions to the nonlinear
////   dynamics of learning in deep linear neural networks." (orthogonal init)

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam_community/maths
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/linalg
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Internal helpers -------------------------------------------------------

/// Upper bound used to convert `int.random` output into a `[0.0, 1.0)`
/// uniform float. `2^31 = 2_147_483_648`.
const random_int_bound: Int = 2_147_483_648

/// Same as `random_int_bound` as a `Float`, kept separate so we don't pay
/// the `int.to_float` cost on every sample.
const random_int_bound_f: Float = 2_147_483_648.0

/// Maximum rejection iterations for `truncated_normal`. Picked high enough
/// that the practical rejection rate (`std > (b - a)` collapsing the
/// truncation window) still terminates, while bounded enough to guarantee
/// the function never hangs.
const truncated_normal_max_iters: Int = 100

/// Sample a single `Float` uniformly in `[0.0, 1.0)`.
fn sample_unit() -> Float {
  int.to_float(int.random(random_int_bound)) /. random_int_bound_f
}

/// Sample a single `Float` uniformly in `[low, high)`.
fn sample_uniform(low: Float, high: Float) -> Float {
  low +. sample_unit() *. { high -. low }
}

/// Sample a single standard normal `Float` via the Box-Muller transform.
/// Clamps `u1` away from zero so `log(u1)` stays finite.
fn sample_standard_normal() -> Float {
  let u1 = float.max(sample_unit(), 1.0e-12)
  let u2 = sample_unit()
  let assert Ok(r) = float.square_root(-2.0 *. log_unsafe(u1))
  r *. maths.cos(2.0 *. maths.pi() *. u2)
}

/// `log` that crashes on non-positive input. Only called from
/// `sample_standard_normal`, which clamps the argument.
fn log_unsafe(x: Float) -> Float {
  let assert Ok(v) = float.logarithm(x)
  v
}

/// Total element count for a shape (empty shape -> 1, like `Tensor` core).
fn size_of(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, d) { acc * d })
}

/// Build a `Tensor` of `shape` whose elements are produced by repeatedly
/// calling `gen`.
fn build(shape: List(Int), gen: fn() -> Float) -> Tensor {
  let size = size_of(shape)
  let data =
    list.range(1, size)
    |> list.map(fn(_) { gen() })
  Tensor(data: data, shape: shape)
}

/// Rejection-sample a single truncated-normal value with bounded iterations.
/// Falls back to a uniform sample inside `[a, b]` if the budget is exhausted,
/// so the result still lies in range even for pathological parameters.
fn sample_truncated(
  mean: Float,
  std: Float,
  a: Float,
  b: Float,
  iters_left: Int,
) -> Float {
  case iters_left <= 0 {
    True -> sample_uniform(a, b)
    False -> {
      let x = mean +. std *. sample_standard_normal()
      case x >=. a && x <=. b {
        True -> x
        False -> sample_truncated(mean, std, a, b, iters_left - 1)
      }
    }
  }
}

// --- Deterministic constructors --------------------------------------------

/// All-zeros tensor. Convenience wrapper for `tensor.zeros`.
///
/// ## Use case
/// Bias vectors typically start at zero; dead-weight masks or accumulators.
pub fn zeros(shape: List(Int)) -> Tensor {
  tensor.zeros(shape)
}

/// All-ones tensor. Convenience wrapper for `tensor.ones`.
///
/// ## Use case
/// LayerNorm / BatchNorm scale parameters (`gamma`) usually start at 1.0.
pub fn ones(shape: List(Int)) -> Tensor {
  tensor.ones(shape)
}

/// Constant-filled tensor. Convenience wrapper for `tensor.fill`.
///
/// ## Use case
/// Bias initialization to a small non-zero value (e.g. 0.01) when you want
/// to break symmetry without random noise.
pub fn constant(shape: List(Int), value: Float) -> Tensor {
  tensor.fill(shape, value)
}

/// Identity matrix of shape `[n, n]`. Wrapper around `tensor.eye`.
///
/// ## Use case
/// Orthogonal residual paths; initializing recurrent transitions to act as
/// identity at step zero.
pub fn identity(n: Int) -> Tensor {
  tensor.eye(n)
}

// --- Random distributions --------------------------------------------------

/// Uniform sampling. Each element is drawn independently from `[low, high)`.
///
/// Formula: `x = low + U * (high - low)` where `U ~ Uniform[0, 1)`.
/// `U` is generated as `int.random(2^31) / 2^31`.
///
/// ## Use case
/// Generic random init when you have an explicit interval (e.g. embedding
/// tables initialized to `[-0.05, 0.05)`).
pub fn uniform(shape: List(Int), low: Float, high: Float) -> Tensor {
  build(shape, fn() { sample_uniform(low, high) })
}

/// Normal (Gaussian) sampling. Each element is drawn independently from
/// `N(mean, std^2)` via the Box-Muller transform on uniform samples.
///
/// ## Use case
/// Foundation of most weight initializers. Prefer `xavier_*` or `kaiming_*`
/// over raw `normal` when you have fan-in / fan-out information.
pub fn normal(shape: List(Int), mean: Float, std: Float) -> Tensor {
  build(shape, fn() { mean +. std *. sample_standard_normal() })
}

/// Truncated normal: samples from `N(mean, std^2)` rejected to stay inside
/// `[a, b]`.
///
/// Uses rejection sampling with a budget of
/// `truncated_normal_max_iters` (100) attempts per element to prevent
/// infinite loops on pathological inputs (e.g. `std` much larger than
/// `b - a`). If the budget is exhausted, falls back to a uniform draw
/// inside `[a, b]` so the post-condition `a <= x <= b` still holds.
///
/// ## Use case
/// Transformer attention layers (e.g. BERT, ViT) commonly initialize weights
/// with a truncated normal of `std = 0.02` to avoid the heavy Gaussian tails.
pub fn truncated_normal(
  shape: List(Int),
  mean: Float,
  std: Float,
  a: Float,
  b: Float,
) -> Tensor {
  build(shape, fn() {
    sample_truncated(mean, std, a, b, truncated_normal_max_iters)
  })
}

// --- Variance-scaled initializers ------------------------------------------

/// Xavier (Glorot) uniform init, returns shape `[fan_in, fan_out]`.
///
/// Formula: `U(-a, a)` with `a = sqrt(6 / (fan_in + fan_out))`.
/// Keeps `Var(out) ≈ Var(in)` for tanh/sigmoid-style activations.
///
/// ## Use case
/// Linear / convolutional layers followed by tanh or sigmoid.
pub fn xavier_uniform(fan_in: Int, fan_out: Int) -> Tensor {
  let denom = int.to_float(fan_in + fan_out)
  let assert Ok(a) = float.square_root(6.0 /. denom)
  uniform([fan_in, fan_out], 0.0 -. a, a)
}

/// Xavier (Glorot) normal init, returns shape `[fan_in, fan_out]`.
///
/// Formula: `N(0, std^2)` with `std = sqrt(2 / (fan_in + fan_out))`.
///
/// ## Use case
/// Same as `xavier_uniform`, but if you specifically want Gaussian tails.
pub fn xavier_normal(fan_in: Int, fan_out: Int) -> Tensor {
  let denom = int.to_float(fan_in + fan_out)
  let assert Ok(std) = float.square_root(2.0 /. denom)
  normal([fan_in, fan_out], 0.0, std)
}

/// Kaiming (He) uniform init, returns shape `[fan_in, fan_out]`.
///
/// Formula: `U(-bound, bound)` with `bound = gain * sqrt(3 / fan_in)`.
/// For ReLU, pass `gain = sqrt(2)` (use `relu_gain()`).
///
/// ## Use case
/// Linear / conv layers followed by ReLU and friends. He et al. (2015) showed
/// this preserves activation variance across deep ReLU networks.
pub fn kaiming_uniform(fan_in: Int, fan_out: Int, gain: Float) -> Tensor {
  let assert Ok(s) = float.square_root(3.0 /. int.to_float(fan_in))
  let bound = gain *. s
  uniform([fan_in, fan_out], 0.0 -. bound, bound)
}

/// Kaiming (He) normal init, returns shape `[fan_in, fan_out]`.
///
/// Formula: `N(0, std^2)` with `std = gain * sqrt(1 / fan_in)`.
/// For ReLU, pass `gain = sqrt(2)` (use `relu_gain()`).
///
/// ## Use case
/// Same as `kaiming_uniform`, but Gaussian sampling. Often paired with
/// truncated_normal in practice to control outliers in deeper networks.
pub fn kaiming_normal(fan_in: Int, fan_out: Int, gain: Float) -> Tensor {
  let assert Ok(s) = float.square_root(1.0 /. int.to_float(fan_in))
  let std = gain *. s
  normal([fan_in, fan_out], 0.0, std)
}

/// Orthogonal initialization via QR decomposition of a random Gaussian
/// matrix, returns shape `[rows, cols]`.
///
/// Algorithm:
///   1. Sample `G ~ N(0, 1)` with shape `[max(rows, cols), min(rows, cols)]`
///      so QR (classical Gram-Schmidt requires `m >= n`) is well-defined.
///   2. Compute `Q, _ = qr(G)`; `Q` has orthonormal columns.
///   3. If `rows < cols`, transpose `Q` so the result has orthonormal rows
///      instead. This is documented behaviour: the returned matrix has
///      orthonormal columns when `rows >= cols`, and orthonormal rows when
///      `rows < cols`.
///   4. Multiply by `gain`.
///
/// Returns `Error(TensorError)` only if the underlying QR fails, which
/// should not happen for a Gaussian sample (probability zero of exact rank
/// deficiency).
///
/// ## Use case
/// RNN / LSTM recurrent weights — orthogonal init keeps spectral norm at
/// `gain`, preventing exploding/vanishing gradients across long sequences.
pub fn orthogonal(
  rows: Int,
  cols: Int,
  gain: Float,
) -> Result(Tensor, TensorError) {
  let #(m, n, transpose_result) = case rows < cols {
    True -> #(cols, rows, True)
    False -> #(rows, cols, False)
  }
  let g = normal([m, n], 0.0, 1.0)
  use #(q, _r) <- result.try(linalg.qr(g))
  use oriented <- result.try(case transpose_result {
    True -> tensor.transpose(q)
    False -> Ok(q)
  })
  let scaled =
    tensor.to_list(oriented)
    |> list.map(fn(x) { x *. gain })
  Ok(Tensor(data: scaled, shape: [rows, cols]))
}

// --- Activation gain helpers -----------------------------------------------

/// Recommended multiplicative gain for layers followed by ReLU.
/// Equal to `sqrt(2)`.
pub fn relu_gain() -> Float {
  let assert Ok(g) = float.square_root(2.0)
  g
}

/// Recommended gain for layers followed by Leaky ReLU with the given
/// `negative_slope`. Equal to `sqrt(2 / (1 + slope^2))`.
pub fn leaky_relu_gain(negative_slope: Float) -> Float {
  let denom = 1.0 +. negative_slope *. negative_slope
  let assert Ok(g) = float.square_root(2.0 /. denom)
  g
}

/// Recommended gain for layers followed by `tanh`. Empirical value `5/3`
/// from the PyTorch literature.
pub fn tanh_gain() -> Float {
  5.0 /. 3.0
}

/// Identity gain (`1.0`) for layers followed by a linear activation.
pub fn linear_gain() -> Float {
  1.0
}

/// Recommended gain for layers followed by `sigmoid`. Equal to `1.0` — the
/// convention used by PyTorch's `calculate_gain` since the sigmoid
/// derivative at zero already sits at `0.25`, which keeps variance bounded.
pub fn sigmoid_gain() -> Float {
  1.0
}
