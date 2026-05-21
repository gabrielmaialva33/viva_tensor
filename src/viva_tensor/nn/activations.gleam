//// Activation functions for neural networks (forward only).
////
//// All activations are pure element-wise (or axis-wise, for softmax) functions
//// over a `Tensor`. They allocate a new tensor and never mutate the input.
////
//// References:
//// - LeCun, Bengio, Hinton (2015). "Deep learning." Nature.
//// - Klambauer et al. (2017). "Self-Normalizing Neural Networks." (SELU)
//// - Hendrycks & Gimpel (2016). "Gaussian Error Linear Units (GELUs)."
//// - Ramachandran et al. (2017). "Searching for Activation Functions." (Swish)
//// - Misra (2019). "Mish: A Self Regularized Non-Monotonic Activation Function."
////
//// Numerical stability notes are baked into each function — `sigmoid`,
//// `softplus`, `softmax`, and `log_softmax` use the standard "shift by max"
//// or "split by sign" tricks to avoid overflow.

import gleam/float
import gleam/list
import gleam/result
import viva_math/scalar as vm_scalar
import viva_tensor/core/error.{type TensorError, DimensionError}
import viva_tensor/core/layout_math
import viva_tensor/core/tensor_axis
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Constants --------------------------------------------------------------

/// SELU scale constant (Klambauer et al. 2017, table 1).
const selu_scale: Float = 1.0507009873554804934193349852946

/// SELU alpha constant (Klambauer et al. 2017, table 1).
const selu_alpha: Float = 1.6732632423543772848170429916717

// --- Element-wise activations -----------------------------------------------

/// Sigmoid activation: `1 / (1 + exp(-x))`.
///
/// Uses a numerically stable split: `exp(x) / (1 + exp(x))` for negative `x`
/// to avoid `exp(-x)` overflow. Output range: `(0, 1)`.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([0.0])
/// activations.sigmoid(t)
/// // -> Tensor([0.5], [1])
/// ```
pub fn sigmoid(t: Tensor) -> Tensor {
  tensor.map(t, sigmoid_scalar)
}

fn sigmoid_scalar(x: Float) -> Float {
  case x >=. 0.0 {
    True -> 1.0 /. { 1.0 +. float.exponential(0.0 -. x) }
    False -> {
      let ex = float.exponential(x)
      ex /. { 1.0 +. ex }
    }
  }
}

/// Hyperbolic tangent activation: `tanh(x)`.
///
/// Output range: `(-1, 1)`. Delegates to `viva_math/scalar.tanh`,
/// which is numerically stable across the entire float range.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([0.0])
/// activations.tanh(t)
/// // -> Tensor([0.0], [1])
/// ```
pub fn tanh(t: Tensor) -> Tensor {
  tensor.map(t, vm_scalar.tanh)
}

/// Rectified Linear Unit: `max(0, x)`.
///
/// The default nonlinearity for most modern feedforward and convolutional
/// networks. Cheap, well-behaved gradient, no saturation for `x > 0`.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([-1.0, 0.0, 2.0])
/// activations.relu(t)
/// // -> Tensor([0.0, 0.0, 2.0], [3])
/// ```
pub fn relu(t: Tensor) -> Tensor {
  tensor.map(t, fn(x) { float.max(x, 0.0) })
}

/// Leaky ReLU: `x` if `x > 0` else `negative_slope * x`.
///
/// Default `negative_slope` is typically `0.01`. Keeps a small gradient for
/// negative inputs so neurons cannot fully die during training.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([-1.0, 2.0])
/// activations.leaky_relu(t, 0.01)
/// // -> Tensor([-0.01, 2.0], [2])
/// ```
pub fn leaky_relu(t: Tensor, negative_slope: Float) -> Tensor {
  tensor.map(t, fn(x) {
    case x >. 0.0 {
      True -> x
      False -> negative_slope *. x
    }
  })
}

/// Exponential Linear Unit: `x` if `x > 0` else `alpha * (exp(x) - 1)`.
///
/// Smoothly saturates for large negative inputs to `-alpha` and has mean
/// activations closer to zero than ReLU. Default `alpha` is `1.0`.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([-1.0, 2.0])
/// activations.elu(t, 1.0)
/// // -> Tensor([-0.6321..., 2.0], [2])
/// ```
pub fn elu(t: Tensor, alpha: Float) -> Tensor {
  tensor.map(t, fn(x) { elu_scalar(x, alpha) })
}

fn elu_scalar(x: Float, alpha: Float) -> Float {
  case x >. 0.0 {
    True -> x
    False -> alpha *. { float.exponential(x) -. 1.0 }
  }
}

/// Scaled Exponential Linear Unit (SELU): `scale * elu(x, alpha)`.
///
/// Uses the canonical constants from Klambauer et al. (2017):
/// `scale = 1.0507...`, `alpha = 1.6732...`. With standard input
/// normalization this induces self-normalizing activations.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([1.0])
/// activations.selu(t)
/// // -> Tensor([1.0507...], [1])
/// ```
pub fn selu(t: Tensor) -> Tensor {
  tensor.map(t, fn(x) { selu_scale *. elu_scalar(x, selu_alpha) })
}

/// Gaussian Error Linear Unit (GELU): `0.5 * x * (1 + erf(x / sqrt(2)))`.
///
/// This module uses the **exact** formulation because `gleam_community/maths`
/// exports `erf`. Output is smooth and matches PyTorch's default GELU.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([1.0])
/// activations.gelu(t)
/// // -> Tensor([0.8413...], [1])
/// ```
pub fn gelu(t: Tensor) -> Tensor {
  // Delegate scalar form to viva_math/scalar.gelu (exact form using erf).
  tensor.map(t, vm_scalar.gelu)
}

/// Swish (a.k.a. SiLU): `x * sigmoid(x)`.
///
/// Smooth, non-monotonic, self-gated activation. Used in EfficientNet and
/// many transformer FFN variants under the name SiLU.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([0.0])
/// activations.swish(t)
/// // -> Tensor([0.0], [1])
/// ```
pub fn swish(t: Tensor) -> Tensor {
  tensor.map(t, fn(x) { x *. sigmoid_scalar(x) })
}

/// Mish: `x * tanh(softplus(x))` where `softplus(x) = log(1 + exp(x))`.
///
/// Smooth alternative to ReLU/Swish with non-monotonic behaviour near zero.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([0.0])
/// activations.mish(t)
/// // -> Tensor([0.0], [1])
/// ```
pub fn mish(t: Tensor) -> Tensor {
  // Delegate scalar form to viva_math/scalar.mish.
  tensor.map(t, vm_scalar.mish)
}

/// Softplus: `log(1 + exp(x))`.
///
/// Numerically stable via `max(x, 0) + log(1 + exp(-|x|))`, which avoids
/// overflow for large positive `x` and underflow for large negative `x`.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([0.0])
/// activations.softplus(t)
/// // -> Tensor([0.6931...], [1])
/// ```
pub fn softplus(t: Tensor) -> Tensor {
  // Delegate scalar form to viva_math/scalar.softplus.
  tensor.map(t, vm_scalar.softplus)
}

// --- Axis-wise activations --------------------------------------------------

/// Softmax along `axis`: `exp(x - max) / sum(exp(x - max))`.
///
/// Subtracts the per-slice maximum before exponentiating to avoid overflow.
/// Each slice along `axis` sums to `1.0` in the output.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([1.0, 2.0, 3.0])
/// let assert Ok(p) = activations.softmax(t, 0)
/// // tensor.to_list(p) ~ [0.0900, 0.2447, 0.6652]
/// ```
pub fn softmax(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  tensor.softmax_axis(t, axis)
}

/// Log-softmax along `axis`: `x - max - log(sum(exp(x - max)))`.
///
/// Computed via the log-sum-exp trick so values stay finite even when raw
/// `softmax` outputs would underflow to zero.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([1.0, 2.0, 3.0])
/// let assert Ok(lp) = activations.log_softmax(t, 0)
/// // sum(exp(lp)) ~ 1.0
/// ```
pub fn log_softmax(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(t)
  let rnk = list.length(shp)

  case axis >= 0 && axis < rnk {
    False -> Error(DimensionError("Invalid axis for log_softmax"))
    True -> {
      use axis_size <- result.try(tensor_axis.axis_size(shp, axis))
      let inner_size = layout_math.size(list.drop(shp, axis + 1))

      case axis_size <= 0 {
        True -> Ok(Tensor(data: [], shape: shp))
        False -> {
          let data = tensor.to_list(t)
          use result_data <- result.try(tensor_axis.axis_transform_data(
            data,
            tensor.size(t),
            axis_size,
            inner_size,
            log_softmax_slice,
          ))
          Ok(Tensor(data: result_data, shape: shp))
        }
      }
    }
  }
}

fn log_softmax_slice(values: List(Float)) -> List(Float) {
  case values {
    [] -> []
    [first, ..rest] -> {
      let max_v = list.fold(rest, first, fn(acc, v) { float.max(acc, v) })
      let shifted = list.map(values, fn(v) { v -. max_v })
      let sum_exp =
        list.fold(shifted, 0.0, fn(acc, s) { acc +. float.exponential(s) })
      let log_sum = case float.logarithm(sum_exp) {
        Ok(l) -> l
        Error(_) -> 0.0
      }
      list.map(shifted, fn(s) { s -. log_sum })
    }
  }
}

// --- Hard / clamped activations ---------------------------------------------

/// HardSwish: `x * relu6(x + 3) / 6` where `relu6(x) = min(max(0, x), 6)`.
///
/// Piecewise-linear approximation of Swish used in efficient CV models
/// (MobileNetV3, etc.). Cheap on hardware without smooth activation ops.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([0.0])
/// activations.hardswish(t)
/// // -> Tensor([0.0], [1])
/// ```
pub fn hardswish(t: Tensor) -> Tensor {
  tensor.map(t, fn(x) {
    let r6 = float.min(float.max(x +. 3.0, 0.0), 6.0)
    x *. r6 /. 6.0
  })
}

/// HardTanh: `clamp(x, min_val, max_val)`.
///
/// Piecewise-linear clamp; identity in the interior, flat outside.
///
/// ## Example
///
/// ```gleam
/// let t = tensor.from_list([-2.0, 0.5, 3.0])
/// activations.hardtanh(t, -1.0, 1.0)
/// // -> Tensor([-1.0, 0.5, 1.0], [3])
/// ```
pub fn hardtanh(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  tensor.map(t, fn(x) { float.min(float.max(x, min_val), max_val) })
}
