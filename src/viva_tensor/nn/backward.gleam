//// Standalone backward (gradient) functions for the v1 NN surface.
////
//// These are **pure functions**, not autograd nodes. Each `*_backward`
//// receives `grad_out` (the upstream gradient) plus whatever forward
//// inputs/outputs are needed to compute the local Jacobian-vector product,
//// and returns gradients with respect to each *differentiable* input.
//// Constant inputs (e.g. classification targets, hyperparameters) do not
//// receive gradients.
////
//// A few of the activation backwards take the **output** of the forward
//// pass (`sigmoid`, `tanh`) and a few of the norm backwards take saved
//// statistics (`mean`, `variance`, `rms`). Docstrings flag this with a
//// "Saved from forward" note. The future autograd `Tape` will keep these
//// values alive between forward and backward.
////
//// Math conventions match PyTorch's `torch.autograd.gradcheck`:
////   reduction=`mean` divides by `N` (total elements) where applicable;
////   reduction=`sum` and `none` do not.
////
//// Pure Gleam, no NIFs, no `Tape` integration in this module.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam_community/maths
import viva_tensor/core/error.{
  type TensorError, DimensionError, InvalidShape, OperandShapeMismatch,
  RankMismatch, ShapeMismatch,
}
import viva_tensor/core/ffi
import viva_tensor/nn/losses.{type Reduction, ReductionMean, ReductionNone}
import viva_tensor/tensor.{type Tensor, Tensor}

// =============================================================================
// CONSTANTS
// =============================================================================

const inv_sqrt2: Float = 0.7071067811865475

const sqrt_2_over_pi: Float = 0.7978845608028654

// =============================================================================
// ELEMENT-WISE ACTIVATION BACKWARDS
// =============================================================================

/// Backward for `relu`. Gradient: `grad_in = grad_out * (input > 0 ? 1 : 0)`.
///
/// At `x = 0` we pick the subgradient `0`, matching PyTorch's convention.
///
/// Arguments:
/// - `grad_out`: upstream gradient (freshly passed).
/// - `input`: original forward input (freshly passed; not the output).
pub fn relu_backward(
  grad_out: Tensor,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  ensure_same_shape("relu_backward", grad_out, input)
  |> result.try(fn(_) {
    elementwise2(grad_out, input, fn(g, x) {
      case x >. 0.0 {
        True -> g
        False -> 0.0
      }
    })
  })
}

/// Backward for `sigmoid`. Gradient: `grad_in = grad_out * output * (1 - output)`.
///
/// Saved from forward: `output` is the sigmoid result `sig(x)`, not the
/// original `x`. This is how PyTorch caches the activation so the backward
/// only needs one buffer per layer.
pub fn sigmoid_backward(
  grad_out: Tensor,
  output: Tensor,
) -> Result(Tensor, TensorError) {
  ensure_same_shape("sigmoid_backward", grad_out, output)
  |> result.try(fn(_) {
    elementwise2(grad_out, output, fn(g, y) { g *. y *. { 1.0 -. y } })
  })
}

/// Backward for `tanh`. Gradient: `grad_in = grad_out * (1 - output^2)`.
///
/// Saved from forward: `output = tanh(x)`. Again, no need to keep the
/// original `x` around.
pub fn tanh_backward(
  grad_out: Tensor,
  output: Tensor,
) -> Result(Tensor, TensorError) {
  ensure_same_shape("tanh_backward", grad_out, output)
  |> result.try(fn(_) {
    elementwise2(grad_out, output, fn(g, y) { g *. { 1.0 -. y *. y } })
  })
}

/// Backward for exact `gelu`.
///
/// Formula:
/// `grad_in = grad_out * 0.5 * (1 + erf(x / sqrt(2))
///                              + x * sqrt(2/pi) * exp(-x^2 / 2))`.
///
/// Uses the input `x` (freshly passed) because the closed-form involves
/// both `erf` and an additional `x * pdf(x)` correction term.
pub fn gelu_backward(
  grad_out: Tensor,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  ensure_same_shape("gelu_backward", grad_out, input)
  |> result.try(fn(_) {
    elementwise2(grad_out, input, fn(g, x) {
      let phi_part = 1.0 +. maths.erf(x *. inv_sqrt2)
      let pdf_part = x *. sqrt_2_over_pi *. float.exponential(-0.5 *. x *. x)
      g *. 0.5 *. { phi_part +. pdf_part }
    })
  })
}

/// Backward for `leaky_relu`.
///
/// Gradient: `grad_in = grad_out * (input > 0 ? 1 : negative_slope)`.
///
/// Subgradient at `x = 0` is `negative_slope` (PyTorch convention).
pub fn leaky_relu_backward(
  grad_out: Tensor,
  input: Tensor,
  negative_slope: Float,
) -> Result(Tensor, TensorError) {
  ensure_same_shape("leaky_relu_backward", grad_out, input)
  |> result.try(fn(_) {
    elementwise2(grad_out, input, fn(g, x) {
      case x >. 0.0 {
        True -> g
        False -> g *. negative_slope
      }
    })
  })
}

/// Backward for `elu`.
///
/// Gradient: `grad_in = grad_out * (input > 0 ? 1 : alpha * exp(input))`.
///
/// Takes the original `input` (freshly passed).
pub fn elu_backward(
  grad_out: Tensor,
  input: Tensor,
  alpha: Float,
) -> Result(Tensor, TensorError) {
  ensure_same_shape("elu_backward", grad_out, input)
  |> result.try(fn(_) {
    elementwise2(grad_out, input, fn(g, x) {
      case x >. 0.0 {
        True -> g
        False -> g *. alpha *. float.exponential(x)
      }
    })
  })
}

// =============================================================================
// LOSS BACKWARDS
// =============================================================================

/// Backward for `mse_loss`. Returns gradient w.r.t. `prediction` only.
///
/// `target` is treated as a constant (no gradient).
///
/// Formula (per element):
/// - `ReductionMean`: `grad_pred = grad_out * 2 * (pred - target) / N`
/// - `ReductionSum` / `ReductionNone`: `grad_pred = grad_out * 2 * (pred - target)`
///
/// `grad_out` for `Mean`/`Sum` is expected to be a scalar tensor (shape
/// `[1]`); for `ReductionNone` it has the same shape as `prediction`.
pub fn mse_loss_backward(
  grad_out: Tensor,
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(ensure_same_shape("mse_loss_backward", prediction, target))
  use pred_data <- result.try(tensor.try_to_list(prediction))
  use target_data <- result.try(tensor.try_to_list(target))
  use scale <- result.try(loss_scale(grad_out, prediction, reduction))
  let n_inv = reduction_inv_n(prediction, reduction)
  let pred_shape = tensor.shape(prediction)
  case reduction {
    ReductionNone -> {
      use grad_data <- result.try(tensor.try_to_list(grad_out))
      let out =
        list.map(list.zip(grad_data, list.zip(pred_data, target_data)), fn(t) {
          let #(g, rest) = t
          let #(p, y) = rest
          g *. 2.0 *. { p -. y }
        })
      Ok(Tensor(data: out, shape: pred_shape))
    }
    _ -> {
      let out =
        list.map(list.zip(pred_data, target_data), fn(pair) {
          let #(p, y) = pair
          scale *. 2.0 *. { p -. y } *. n_inv
        })
      Ok(Tensor(data: out, shape: pred_shape))
    }
  }
}

/// Backward for `l1_loss`. Returns gradient w.r.t. `prediction` only.
///
/// Formula (per element):
/// - `ReductionMean`: `grad_pred = grad_out * sign(pred - target) / N`
/// - `ReductionSum` / `ReductionNone`: `grad_pred = grad_out * sign(pred - target)`
///
/// `sign(0)` is `0` (matches PyTorch's subgradient choice).
pub fn l1_loss_backward(
  grad_out: Tensor,
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(ensure_same_shape("l1_loss_backward", prediction, target))
  use pred_data <- result.try(tensor.try_to_list(prediction))
  use target_data <- result.try(tensor.try_to_list(target))
  let pred_shape = tensor.shape(prediction)
  case reduction {
    ReductionNone -> {
      use grad_data <- result.try(tensor.try_to_list(grad_out))
      let out =
        list.map(list.zip(grad_data, list.zip(pred_data, target_data)), fn(t) {
          let #(g, rest) = t
          let #(p, y) = rest
          g *. sign(p -. y)
        })
      Ok(Tensor(data: out, shape: pred_shape))
    }
    _ -> {
      use scale <- result.try(loss_scale(grad_out, prediction, reduction))
      let n_inv = reduction_inv_n(prediction, reduction)
      let out =
        list.map(list.zip(pred_data, target_data), fn(pair) {
          let #(p, y) = pair
          scale *. sign(p -. y) *. n_inv
        })
      Ok(Tensor(data: out, shape: pred_shape))
    }
  }
}

/// Backward for `bce_loss`. Returns gradient w.r.t. `prediction` only.
///
/// Formula (per element):
/// - `ReductionMean`: `grad_pred = grad_out * (pred - target) / (pred * (1 - pred) * N)`
/// - `ReductionSum` / `ReductionNone`: drop the `/N` factor.
///
/// `prediction` is clamped to `[eps, 1 - eps]` (eps = 1e-7) to stay in
/// lock-step with the forward pass.
pub fn bce_loss_backward(
  grad_out: Tensor,
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(ensure_same_shape("bce_loss_backward", prediction, target))
  use pred_data <- result.try(tensor.try_to_list(prediction))
  use target_data <- result.try(tensor.try_to_list(target))
  let eps = 1.0e-7
  let one_minus_eps = 1.0 -. eps
  let pred_shape = tensor.shape(prediction)
  case reduction {
    ReductionNone -> {
      use grad_data <- result.try(tensor.try_to_list(grad_out))
      let out =
        list.map(list.zip(grad_data, list.zip(pred_data, target_data)), fn(t) {
          let #(g, rest) = t
          let #(p, y) = rest
          let p_c = float.clamp(p, min: eps, max: one_minus_eps)
          g *. { p_c -. y } /. { p_c *. { 1.0 -. p_c } }
        })
      Ok(Tensor(data: out, shape: pred_shape))
    }
    _ -> {
      use scale <- result.try(loss_scale(grad_out, prediction, reduction))
      let n_inv = reduction_inv_n(prediction, reduction)
      let out =
        list.map(list.zip(pred_data, target_data), fn(pair) {
          let #(p, y) = pair
          let p_c = float.clamp(p, min: eps, max: one_minus_eps)
          scale *. { p_c -. y } /. { p_c *. { 1.0 -. p_c } } *. n_inv
        })
      Ok(Tensor(data: out, shape: pred_shape))
    }
  }
}

/// Backward for `cross_entropy_loss`. Returns gradient w.r.t. `logits`.
///
/// Formula:
/// `grad_logits = grad_out * (softmax(logits) - one_hot(targets)) / batch`
/// for `ReductionMean`, omit `/batch` for `ReductionSum`. For
/// `ReductionNone` (per-row loss), `grad_out` is expected to have shape
/// `[batch]` and the division by `batch` is dropped.
///
/// `targets` is treated as a constant. Shape: `logits` is `[batch, C]`,
/// `targets` is `[batch]` (float-encoded class indices), returned grad
/// has shape `[batch, C]`.
pub fn cross_entropy_loss_backward(
  grad_out: Tensor,
  logits: Tensor,
  targets: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  let logits_shape = tensor.shape(logits)
  use #(batch, num_classes) <- result.try(case logits_shape {
    [b, c] -> Ok(#(b, c))
    _ ->
      Error(RankMismatch(
        operation: "cross_entropy_loss_backward",
        expected_rank: 2,
        got_shape: logits_shape,
      ))
  })

  let targets_shape = tensor.shape(targets)
  use _ <- result.try(case targets_shape {
    [t_batch] if t_batch == batch -> Ok(Nil)
    _ ->
      Error(OperandShapeMismatch(
        operation: "cross_entropy_loss_backward",
        operand: "targets",
        expected: "[batch]",
        got: targets_shape,
      ))
  })

  use softmaxed <- result.try(tensor.softmax_axis(logits, 1))
  use sm_data <- result.try(tensor.try_to_list(softmaxed))
  use target_data <- result.try(tensor.try_to_list(targets))

  let inv_batch = case reduction {
    ReductionMean -> 1.0 /. int.to_float(batch)
    _ -> 1.0
  }

  case reduction {
    ReductionNone -> {
      // grad_out is [batch], one upstream per row.
      use grad_data <- result.try(tensor.try_to_list(grad_out))
      let rows = chunk_every(sm_data, num_classes)
      let zipped = list.zip(rows, list.zip(target_data, grad_data))
      let grad_rows =
        list.map(zipped, fn(t) {
          let #(row, rest) = t
          let #(target_f, g) = rest
          let class_idx = float.round(target_f)
          list.index_map(row, fn(p, i) {
            let one_hot = case i == class_idx {
              True -> 1.0
              False -> 0.0
            }
            g *. { p -. one_hot }
          })
        })
      Ok(Tensor(data: list.flatten(grad_rows), shape: [batch, num_classes]))
    }
    _ -> {
      use scale <- result.try(grad_scalar_value(grad_out))
      let rows = chunk_every(sm_data, num_classes)
      let zipped = list.zip(rows, target_data)
      let grad_rows =
        list.map(zipped, fn(pair) {
          let #(row, target_f) = pair
          let class_idx = float.round(target_f)
          list.index_map(row, fn(p, i) {
            let one_hot = case i == class_idx {
              True -> 1.0
              False -> 0.0
            }
            scale *. { p -. one_hot } *. inv_batch
          })
        })
      Ok(Tensor(data: list.flatten(grad_rows), shape: [batch, num_classes]))
    }
  }
}

// =============================================================================
// LINEAR / MATMUL BACKWARDS
// =============================================================================

/// Backward for a linear layer `output = input @ weight`.
///
/// Returns `#(grad_input, grad_weight)`:
/// - `grad_input  = grad_out @ weight^T`
/// - `grad_weight = input^T @ grad_out`
///
/// Both `input` and `weight` are freshly passed (the future autograd `Tape`
/// will save them). This module does **not** model an explicit bias term:
/// gradient w.r.t. a bias would be `sum(grad_out, axis=0)`.
///
/// Shape contract:
/// - `input`  : `[batch, in_features]`
/// - `weight` : `[in_features, out_features]`
/// - `grad_out`: `[batch, out_features]`
pub fn linear_backward(
  grad_out: Tensor,
  input: Tensor,
  weight: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  matmul_backward(grad_out, input, weight)
}

/// Backward for matrix multiplication `output = a @ b`.
///
/// Returns `#(grad_a, grad_b)`:
/// - `grad_a = grad_out @ b^T`
/// - `grad_b = a^T @ grad_out`
///
/// Same math as `linear_backward`, exposed under the matmul name for the
/// user-facing `viva_tensor.matmul`.
pub fn matmul_backward(
  grad_out: Tensor,
  a: Tensor,
  b: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  use b_t <- result.try(tensor.transpose(b))
  use a_t <- result.try(tensor.transpose(a))
  use grad_a <- result.try(tensor.matmul(grad_out, b_t))
  use grad_b <- result.try(tensor.matmul(a_t, grad_out))
  Ok(#(grad_a, grad_b))
}

// =============================================================================
// NORM BACKWARDS
// =============================================================================

/// Backward for `layer_norm` over the last dimension.
///
/// Forward formula (recap):
///   `x_hat = (x - mean) / sqrt(var + eps)`
///   `y     = x_hat * scale + bias`
///
/// Saved from forward:
/// - `mean`     : shape == input shape with the last dim collapsed (size
///                 = product of all dims except the last). One mean per slice.
/// - `variance` : same shape as `mean`. One variance per slice.
///
/// Freshly passed: `grad_out`, `input`, `scale`, `eps`.
///
/// Returns `#(grad_input, grad_scale, grad_bias)` where `grad_scale`/`grad_bias`
/// have the same shape as `scale` (`[D]`) and are reduced over all batch dims.
///
/// Math (per slice of size `D` along the last axis):
///   `g     = grad_out * scale`
///   `m1    = mean(g)`
///   `m2    = mean(g * x_hat)`
///   `grad_x = (g - m1 - x_hat * m2) / std`
pub fn layer_norm_backward(
  grad_out: Tensor,
  input: Tensor,
  scale: Tensor,
  mean: Tensor,
  variance: Tensor,
  eps: Float,
) -> Result(#(Tensor, Tensor, Tensor), TensorError) {
  let input_shape = tensor.shape(input)
  use _ <- result.try(case tensor.shape(grad_out) == input_shape {
    True -> Ok(Nil)
    False ->
      Error(ShapeMismatch(expected: input_shape, got: tensor.shape(grad_out)))
  })

  use d <- result.try(last_dim_of(input_shape))
  let scale_shape = tensor.shape(scale)
  use _ <- result.try(case scale_shape == [d] {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(expected: [d], got: scale_shape))
  })

  let outer = product(input_shape) / int.max(d, 1)
  use _ <- result.try(check_stat_size("layer_norm_backward.mean", mean, outer))
  use _ <- result.try(check_stat_size(
    "layer_norm_backward.variance",
    variance,
    outer,
  ))

  use data <- result.try(tensor.try_to_list(input))
  use grad_data <- result.try(tensor.try_to_list(grad_out))
  use scale_data <- result.try(tensor.try_to_list(scale))
  use mean_data <- result.try(tensor.try_to_list(mean))
  use var_data <- result.try(tensor.try_to_list(variance))

  let input_rows = chunk_every(data, d)
  let grad_rows = chunk_every(grad_data, d)
  let combined =
    list.zip(input_rows, list.zip(grad_rows, list.zip(mean_data, var_data)))

  let init_acc = #([], list.repeat(0.0, d), list.repeat(0.0, d))
  let #(rev_grad_x, grad_scale_data, grad_bias_data) =
    list.fold(combined, init_acc, fn(acc, row) {
      let #(rev_gx, gs, gb) = acc
      let #(x_row, rest1) = row
      let #(g_row, stats) = rest1
      let #(mu, var) = stats
      let std = safe_sqrt(var +. eps)
      let inv_std = 1.0 /. std
      // x_hat for this slice
      let x_hat = list.map(x_row, fn(x) { { x -. mu } *. inv_std })
      // g_scaled = grad_out * scale (element-wise per feature)
      let g_scaled =
        list.map(list.zip(g_row, scale_data), fn(p) {
          let #(g, s) = p
          g *. s
        })
      let d_f = int.to_float(d)
      let m1 = list.fold(g_scaled, 0.0, fn(a, v) { a +. v }) /. d_f
      let m2 =
        list.fold(list.zip(g_scaled, x_hat), 0.0, fn(a, p) {
          let #(gs2, xh) = p
          a +. gs2 *. xh
        })
        /. d_f
      // grad_x slice
      let grad_x_slice =
        list.map(list.zip(g_scaled, x_hat), fn(p) {
          let #(gs2, xh) = p
          { gs2 -. m1 -. xh *. m2 } *. inv_std
        })
      // grad_scale += grad_out * x_hat   (per feature)
      let new_gs =
        list.map(list.zip(gs, list.zip(g_row, x_hat)), fn(t) {
          let #(acc_v, rest2) = t
          let #(g, xh) = rest2
          acc_v +. g *. xh
        })
      // grad_bias += grad_out (per feature)
      let new_gb =
        list.map(list.zip(gb, g_row), fn(p) {
          let #(acc_v, g) = p
          acc_v +. g
        })
      #([grad_x_slice, ..rev_gx], new_gs, new_gb)
    })

  let grad_x_data =
    rev_grad_x
    |> list.reverse
    |> list.flatten
  Ok(#(
    Tensor(data: grad_x_data, shape: input_shape),
    Tensor(data: grad_scale_data, shape: [d]),
    Tensor(data: grad_bias_data, shape: [d]),
  ))
}

/// Backward for `rms_norm` over the last dimension.
///
/// Forward formula:
///   `rms = sqrt(mean(x^2) + eps)`
///   `y   = x / rms * scale`
///
/// Saved from forward:
/// - `rms` : one value per slice, shape product == `input` shape minus
///           last dim.
///
/// Freshly passed: `grad_out`, `input`, `scale`, `eps` (unused arithmetically
/// because `rms` already contains it; kept in the signature for symmetry
/// with the LayerNorm backward and easier callsite ergonomics).
///
/// Returns `#(grad_input, grad_scale)`:
///   `g       = grad_out * scale`
///   `dot     = sum(g * x) / D`   (`D` = size of last dim)
///   `grad_x  = (g - x * dot / (rms^2)) / rms`
///   `grad_s  = sum(grad_out * x / rms)`  (reduced over batch dims)
pub fn rms_norm_backward(
  grad_out: Tensor,
  input: Tensor,
  scale: Tensor,
  rms: Tensor,
  _eps: Float,
) -> Result(#(Tensor, Tensor), TensorError) {
  // `_eps` retained for signature symmetry; the caller is expected to have
  // baked it into `rms` already.
  let input_shape = tensor.shape(input)
  use _ <- result.try(case tensor.shape(grad_out) == input_shape {
    True -> Ok(Nil)
    False ->
      Error(ShapeMismatch(expected: input_shape, got: tensor.shape(grad_out)))
  })
  use d <- result.try(last_dim_of(input_shape))
  let scale_shape = tensor.shape(scale)
  use _ <- result.try(case scale_shape == [d] {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(expected: [d], got: scale_shape))
  })
  let outer = product(input_shape) / int.max(d, 1)
  use _ <- result.try(check_stat_size("rms_norm_backward.rms", rms, outer))

  use data <- result.try(tensor.try_to_list(input))
  use grad_data <- result.try(tensor.try_to_list(grad_out))
  use scale_data <- result.try(tensor.try_to_list(scale))
  use rms_data <- result.try(tensor.try_to_list(rms))

  let input_rows = chunk_every(data, d)
  let grad_rows = chunk_every(grad_data, d)
  let combined = list.zip(input_rows, list.zip(grad_rows, rms_data))

  let init_acc = #([], list.repeat(0.0, d))
  let #(rev_grad_x, grad_scale_data) =
    list.fold(combined, init_acc, fn(acc, row) {
      let #(rev_gx, gs) = acc
      let #(x_row, rest) = row
      let #(g_row, r) = rest
      let inv_r = 1.0 /. r
      let inv_r2 = inv_r *. inv_r
      let g_scaled =
        list.map(list.zip(g_row, scale_data), fn(p) {
          let #(g, s) = p
          g *. s
        })
      let d_f = int.to_float(d)
      let dot =
        list.fold(list.zip(g_scaled, x_row), 0.0, fn(a, p) {
          let #(gs2, x) = p
          a +. gs2 *. x
        })
        /. d_f
      let grad_x_slice =
        list.map(list.zip(g_scaled, x_row), fn(p) {
          let #(gs2, x) = p
          { gs2 -. x *. dot *. inv_r2 } *. inv_r
        })
      // grad_scale_i += grad_out_i * x_i / rms
      let new_gs =
        list.map(list.zip(gs, list.zip(g_row, x_row)), fn(t) {
          let #(acc_v, rest2) = t
          let #(g, x) = rest2
          acc_v +. g *. x *. inv_r
        })
      #([grad_x_slice, ..rev_gx], new_gs)
    })

  let grad_x_data =
    rev_grad_x
    |> list.reverse
    |> list.flatten
  Ok(#(
    Tensor(data: grad_x_data, shape: input_shape),
    Tensor(data: grad_scale_data, shape: [d]),
  ))
}

// =============================================================================
// SOFTMAX BACKWARD
// =============================================================================

/// Backward for `softmax` along `axis`.
///
/// Formula along the axis:
///   `grad_in[i] = output[i] * (grad_out[i] - sum(grad_out * output))`
///
/// Saved from forward: `output` (the softmax result, NOT the original input).
///
/// Supports any rank with valid `axis`. The implementation walks slices of
/// length `axis_size` separated by `inner_size`, matching the forward in
/// `viva_tensor/core/tensor_axis`.
pub fn softmax_backward(
  grad_out: Tensor,
  output: Tensor,
  axis: Int,
) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(output)
  use _ <- result.try(case tensor.shape(grad_out) == shp {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(expected: shp, got: tensor.shape(grad_out)))
  })
  let rnk = list.length(shp)
  case axis >= 0 && axis < rnk {
    False -> Error(DimensionError("Invalid axis for softmax_backward"))
    True -> {
      let axis_size = nth_dim(shp, axis)
      let inner =
        shp
        |> list.drop(axis + 1)
        |> product
      let outer =
        shp
        |> list.take(axis)
        |> product

      case axis_size <= 0 {
        True -> Ok(Tensor(data: [], shape: shp))
        False -> {
          use grad_data <- result.try(tensor.try_to_list(grad_out))
          use out_data <- result.try(tensor.try_to_list(output))
          let buffer =
            softmax_backward_data(grad_data, out_data, outer, axis_size, inner)
          Ok(Tensor(data: buffer, shape: shp))
        }
      }
    }
  }
}

// Walk slices laid out as [outer, axis, inner] and write per-element grad.
//
// For each (outer_idx, inner_idx) pair, the slice along the axis has length
// `axis_size`. We compute `s = sum(grad * out)` over the slice once, then
// emit `out[i] * (grad[i] - s)` for each element.
fn softmax_backward_data(
  grad: List(Float),
  out: List(Float),
  outer: Int,
  axis_size: Int,
  inner: Int,
) -> List(Float) {
  let grad_arr = ffi.list_to_array(grad)
  let out_arr = ffi.list_to_array(out)
  range_int(0, outer - 1)
  |> list.flat_map(fn(o) {
    let outer_offset = o * axis_size * inner
    // Pre-compute per-(inner) sum_g_y across the axis dim.
    let sums =
      range_int(0, inner - 1)
      |> list.map(fn(inner_idx) {
        range_int(0, axis_size - 1)
        |> list.fold(0.0, fn(acc, k) {
          let idx = outer_offset + k * inner + inner_idx
          acc +. ffi.array_get(grad_arr, idx) *. ffi.array_get(out_arr, idx)
        })
      })
    let sums_arr = ffi.list_to_array(sums)
    // Now emit per-position output in flat order [axis, inner].
    range_int(0, axis_size - 1)
    |> list.flat_map(fn(k) {
      range_int(0, inner - 1)
      |> list.map(fn(inner_idx) {
        let idx = outer_offset + k * inner + inner_idx
        let g = ffi.array_get(grad_arr, idx)
        let y = ffi.array_get(out_arr, idx)
        let s = ffi.array_get(sums_arr, inner_idx)
        y *. { g -. s }
      })
    })
  })
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

fn elementwise2(
  a: Tensor,
  b: Tensor,
  f: fn(Float, Float) -> Float,
) -> Result(Tensor, TensorError) {
  use a_data <- result.try(tensor.try_to_list(a))
  use b_data <- result.try(tensor.try_to_list(b))
  let out =
    list.zip(a_data, b_data)
    |> list.map(fn(p) {
      let #(x, y) = p
      f(x, y)
    })
  Ok(Tensor(data: out, shape: tensor.shape(a)))
}

fn ensure_same_shape(
  _op: String,
  a: Tensor,
  b: Tensor,
) -> Result(Nil, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> Ok(Nil)
    False ->
      Error(ShapeMismatch(expected: tensor.shape(a), got: tensor.shape(b)))
  }
}

fn loss_scale(
  grad_out: Tensor,
  prediction: Tensor,
  reduction: Reduction,
) -> Result(Float, TensorError) {
  case reduction {
    ReductionNone -> {
      case tensor.shape(grad_out) == tensor.shape(prediction) {
        True -> Ok(1.0)
        False ->
          Error(ShapeMismatch(
            expected: tensor.shape(prediction),
            got: tensor.shape(grad_out),
          ))
      }
    }
    _ -> grad_scalar_value(grad_out)
  }
}

fn grad_scalar_value(grad_out: Tensor) -> Result(Float, TensorError) {
  case tensor.try_to_list(grad_out) {
    Ok([v]) -> Ok(v)
    Ok(_) ->
      Error(InvalidShape(
        "expected scalar (shape [1]) grad_out for reduced loss backward",
      ))
    Error(e) -> Error(e)
  }
}

fn reduction_inv_n(prediction: Tensor, reduction: Reduction) -> Float {
  case reduction {
    ReductionMean -> {
      let n = tensor.size(prediction)
      case n <= 0 {
        True -> 1.0
        False -> 1.0 /. int.to_float(n)
      }
    }
    _ -> 1.0
  }
}

fn sign(x: Float) -> Float {
  case x >. 0.0, x <. 0.0 {
    True, _ -> 1.0
    _, True -> -1.0
    _, _ -> 0.0
  }
}

fn safe_sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn last_dim_of(shape: List(Int)) -> Result(Int, TensorError) {
  case list.last(shape) {
    Ok(d) -> Ok(d)
    Error(_) -> Error(InvalidShape("expected non-empty shape, got []"))
  }
}

fn nth_dim(shape: List(Int), idx: Int) -> Int {
  case shape, idx {
    [], _ -> 0
    [d, ..], 0 -> d
    [_, ..rest], i -> nth_dim(rest, i - 1)
  }
}

fn product(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(a, b) { a * b })
}

fn check_stat_size(
  _op: String,
  t: Tensor,
  expected: Int,
) -> Result(Nil, TensorError) {
  let n = tensor.size(t)
  case n == expected {
    True -> Ok(Nil)
    False ->
      Error(InvalidShape(
        "expected "
        <> int.to_string(expected)
        <> " entries, got "
        <> int.to_string(n),
      ))
  }
}

fn chunk_every(items: List(Float), n: Int) -> List(List(Float)) {
  case n <= 0 {
    True -> []
    False -> do_chunk_every(items, n, [])
  }
}

fn do_chunk_every(
  items: List(Float),
  n: Int,
  acc: List(List(Float)),
) -> List(List(Float)) {
  case items {
    [] -> list.reverse(acc)
    _ -> {
      let chunk = list.take(items, n)
      let rest = list.drop(items, n)
      do_chunk_every(rest, n, [chunk, ..acc])
    }
  }
}

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}
