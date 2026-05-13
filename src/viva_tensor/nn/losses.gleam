//// Common neural-network loss functions.
////
//// Forward-pass-only in v1: each loss returns a tensor produced from the
//// prediction/target pair. Autograd backward passes are intentionally NOT
//// wired up in this round — the existing `viva_tensor/nn/autograd` module
//// has its own surface and the loss closures will be added there in a
//// follow-up commit.
////
//// All losses follow the same reduction convention as PyTorch:
//// - `ReductionNone` returns the per-element loss tensor (same shape as
////   `prediction`).
//// - `ReductionMean` returns a 1-element tensor with the arithmetic mean.
//// - `ReductionSum` returns a 1-element tensor with the sum.
////
//// ```gleam
//// import viva_tensor as t
//// import viva_tensor/nn/losses
////
//// let pred = t.from_list([0.9, 0.2, 0.7])
//// let target = t.from_list([1.0, 0.0, 1.0])
//// let assert Ok(loss) = losses.bce_loss(pred, target, losses.ReductionMean)
//// ```

import gleam/float
import gleam/list
import gleam/result
import viva_tensor/core/error.{
  type TensorError, IndexOutOfBounds, InvalidShape, OperandShapeMismatch,
  RankMismatch, ShapeMismatch,
}
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Public types -----------------------------------------------------------

/// How to reduce a per-element loss tensor to a scalar (or keep it as-is).
///
/// Matches PyTorch's `reduction` argument:
/// - `ReductionNone` keeps the per-element shape.
/// - `ReductionMean` averages every element to a single scalar.
/// - `ReductionSum` sums every element to a single scalar.
pub type Reduction {
  ReductionNone
  ReductionMean
  ReductionSum
}

// --- Constants --------------------------------------------------------------

/// Clamp epsilon for `bce_loss`. Picked to match the default used by
/// `torch.nn.functional.binary_cross_entropy` and Keras' BCE.
const bce_eps: Float = 1.0e-7

// --- Mean Squared Error -----------------------------------------------------

/// Mean Squared Error: `mean((prediction - target)^2)`.
///
/// Per-element loss is `(pred_i - target_i)^2`. The `reduction` argument
/// controls how those per-element losses are aggregated.
///
/// Errors with `ShapeMismatch` when the two tensors have different shapes.
///
/// ```gleam
/// import viva_tensor as t
/// import viva_tensor/nn/losses
///
/// let pred = t.from_list([1.0, 2.0, 3.0])
/// let target = t.from_list([1.5, 2.5, 2.0])
/// let assert Ok(loss) = losses.mse_loss(pred, target, losses.ReductionMean)
/// // t.to_list(loss) == [0.5]
/// ```
pub fn mse_loss(
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  use elementwise <- result.try(squared_error(prediction, target))
  Ok(reduce(elementwise, reduction))
}

// --- Mean Absolute Error / L1 ----------------------------------------------

/// L1 / Mean Absolute Error: `mean(|prediction - target|)`.
///
/// Per-element loss is `|pred_i - target_i|`. Robust to outliers compared to
/// MSE; the gradient is constant in magnitude on either side of zero.
///
/// Errors with `ShapeMismatch` when the two tensors have different shapes.
///
/// ```gleam
/// import viva_tensor as t
/// import viva_tensor/nn/losses
///
/// let pred = t.from_list([1.0, 2.0, 3.0])
/// let target = t.from_list([1.5, 1.5, 4.0])
/// let assert Ok(loss) = losses.l1_loss(pred, target, losses.ReductionMean)
/// // t.to_list(loss) == [0.6666666666666666]
/// ```
pub fn l1_loss(
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  use elementwise <- result.try(absolute_error(prediction, target))
  Ok(reduce(elementwise, reduction))
}

// --- Binary Cross-Entropy ---------------------------------------------------

/// Binary Cross-Entropy: `-[target * log(pred) + (1 - target) * log(1 - pred)]`.
///
/// `prediction` is clamped to `[eps, 1 - eps]` (eps = 1e-7) before the log
/// to avoid `log(0)` blowing up. `target` must contain values in `[0, 1]`.
///
/// Errors:
/// - `ShapeMismatch` if the two tensors have different shapes.
/// - `InvalidShape` if any `target` element is outside `[0, 1]`.
///
/// ```gleam
/// import viva_tensor as t
/// import viva_tensor/nn/losses
///
/// let pred = t.from_list([0.9, 0.2])
/// let target = t.from_list([1.0, 0.0])
/// let assert Ok(loss) = losses.bce_loss(pred, target, losses.ReductionMean)
/// ```
pub fn bce_loss(
  prediction: Tensor,
  target: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  let pred_shape = tensor.shape(prediction)
  let target_shape = tensor.shape(target)
  use _ <- result.try(case pred_shape == target_shape {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(expected: pred_shape, got: target_shape))
  })

  use pred_data <- result.try(tensor.try_to_list(prediction))
  use target_data <- result.try(tensor.try_to_list(target))

  use _ <- result.try(validate_probability_targets(target_data))

  let one_minus_eps = 1.0 -. bce_eps
  let per_element =
    list.zip(pred_data, target_data)
    |> list.map(fn(pair) {
      let #(p, y) = pair
      let p_clamped = float.clamp(p, min: bce_eps, max: one_minus_eps)
      let one_minus_p = 1.0 -. p_clamped
      let one_minus_y = 1.0 -. y
      0.0 -. { y *. ffi_log(p_clamped) +. one_minus_y *. ffi_log(one_minus_p) }
    })

  let elementwise = Tensor(data: per_element, shape: pred_shape)
  Ok(reduce(elementwise, reduction))
}

// --- Softmax Cross-Entropy (multiclass) -------------------------------------

/// Softmax cross-entropy with integer-valued class targets.
///
/// `logits` has shape `[batch, num_classes]`. `targets` has shape `[batch]`
/// and contains the index of the correct class for each row, encoded as a
/// float. The per-row loss is
///
/// ```text
/// loss_i = -logits[i, target_i] + logsumexp(logits[i, :])
/// ```
///
/// which is the numerically stable form of `-log(softmax(logits)[target])`.
///
/// Errors:
/// - `RankMismatch` if `logits` is not 2D.
/// - `OperandShapeMismatch` if `targets` is not 1D with matching batch size.
/// - `IndexOutOfBounds` if any target value is outside `[0, num_classes)`.
///
/// ```gleam
/// import viva_tensor as t
/// import viva_tensor/nn/losses
///
/// let assert Ok(logits) = t.from_list2d([[2.0, 1.0], [0.5, 2.5]])
/// let targets = t.from_list([0.0, 1.0])
/// let assert Ok(loss) =
///   losses.cross_entropy_loss(logits, targets, losses.ReductionMean)
/// ```
pub fn cross_entropy_loss(
  logits: Tensor,
  targets: Tensor,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  let logits_shape = tensor.shape(logits)
  use #(batch, num_classes) <- result.try(case logits_shape {
    [b, c] -> Ok(#(b, c))
    _ ->
      Error(RankMismatch(
        operation: "cross_entropy_loss",
        expected_rank: 2,
        got_shape: logits_shape,
      ))
  })

  let targets_shape = tensor.shape(targets)
  use _ <- result.try(case targets_shape {
    [t_batch] if t_batch == batch -> Ok(Nil)
    _ ->
      Error(OperandShapeMismatch(
        operation: "cross_entropy_loss",
        operand: "targets",
        expected: "[batch]",
        got: targets_shape,
      ))
  })

  use logits_data <- result.try(tensor.try_to_list(logits))
  use target_data <- result.try(tensor.try_to_list(targets))

  let rows = chunk_every(logits_data, num_classes)
  let zipped = list.zip(rows, target_data)

  use per_row <- result.try(
    list_try_map(zipped, fn(pair) {
      let #(row, target_f) = pair
      let class_idx = float.round(target_f)
      case class_idx < 0 || class_idx >= num_classes {
        True -> Error(IndexOutOfBounds(index: class_idx, size: num_classes))
        False -> {
          let assert Ok(logit_for_target) = list_at(row, class_idx)
          let lse = logsumexp(row)
          Ok(0.0 -. logit_for_target +. lse)
        }
      }
    }),
  )

  let elementwise = Tensor(data: per_row, shape: [batch])
  Ok(reduce(elementwise, reduction))
}

// --- Huber loss (smooth L1) -------------------------------------------------

/// Huber loss: quadratic for small errors, linear for large ones.
///
/// For each element, with `e = pred - target`:
/// - if `|e| < delta`: `0.5 * e^2`
/// - else: `delta * (|e| - 0.5 * delta)`
///
/// This is the loss popularized in robust regression: gentle on inliers,
/// hard cap on the gradient for outliers.
///
/// Errors with `ShapeMismatch` when the two tensors have different shapes.
///
/// ```gleam
/// import viva_tensor as t
/// import viva_tensor/nn/losses
///
/// let pred = t.from_list([0.0, 5.0])
/// let target = t.from_list([0.1, 0.0])
/// let assert Ok(loss) =
///   losses.huber_loss(pred, target, 1.0, losses.ReductionMean)
/// ```
pub fn huber_loss(
  prediction: Tensor,
  target: Tensor,
  delta: Float,
  reduction: Reduction,
) -> Result(Tensor, TensorError) {
  use diff <- result.try(tensor.sub(prediction, target))
  use diff_data <- result.try(tensor.try_to_list(diff))
  let half_delta = 0.5 *. delta
  let per_element =
    list.map(diff_data, fn(e) {
      let abs_e = float.absolute_value(e)
      case abs_e <. delta {
        True -> 0.5 *. e *. e
        False -> delta *. { abs_e -. half_delta }
      }
    })
  let elementwise = Tensor(data: per_element, shape: tensor.shape(prediction))
  Ok(reduce(elementwise, reduction))
}

// --- Internal helpers -------------------------------------------------------

fn squared_error(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use diff <- result.try(tensor.sub(a, b))
  Ok(tensor.square(diff))
}

fn absolute_error(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use diff <- result.try(tensor.sub(a, b))
  Ok(tensor.abs(diff))
}

fn reduce(elementwise: Tensor, reduction: Reduction) -> Tensor {
  case reduction {
    ReductionNone -> elementwise
    ReductionMean -> Tensor(data: [tensor.mean(elementwise)], shape: [1])
    ReductionSum -> Tensor(data: [tensor.sum(elementwise)], shape: [1])
  }
}

fn validate_probability_targets(
  targets: List(Float),
) -> Result(Nil, TensorError) {
  case list.all(targets, fn(y) { y >=. 0.0 && y <=. 1.0 }) {
    True -> Ok(Nil)
    False ->
      Error(InvalidShape("bce_loss: target contains values outside [0, 1]"))
  }
}

fn logsumexp(row: List(Float)) -> Float {
  case row {
    [] -> 0.0
    [first, ..rest] -> {
      let max_val = list.fold(rest, first, fn(acc, x) { float.max(acc, x) })
      let sum_exp =
        list.fold(row, 0.0, fn(acc, x) { acc +. ffi_exp(x -. max_val) })
      max_val +. ffi_log(sum_exp)
    }
  }
}

fn chunk_every(items: List(Float), size: Int) -> List(List(Float)) {
  case items {
    [] -> []
    _ -> {
      let #(head, rest) = take_split(items, size, [])
      [head, ..chunk_every(rest, size)]
    }
  }
}

fn take_split(
  items: List(Float),
  n: Int,
  acc: List(Float),
) -> #(List(Float), List(Float)) {
  case n, items {
    0, _ -> #(list.reverse(acc), items)
    _, [] -> #(list.reverse(acc), [])
    _, [x, ..rest] -> take_split(rest, n - 1, [x, ..acc])
  }
}

fn list_at(items: List(Float), index: Int) -> Result(Float, Nil) {
  case items, index {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], _ -> list_at(rest, index - 1)
  }
}

fn list_try_map(
  items: List(a),
  f: fn(a) -> Result(b, TensorError),
) -> Result(List(b), TensorError) {
  do_list_try_map(items, f, [])
}

fn do_list_try_map(
  items: List(a),
  f: fn(a) -> Result(b, TensorError),
  acc: List(b),
) -> Result(List(b), TensorError) {
  case items {
    [] -> Ok(list.reverse(acc))
    [head, ..rest] ->
      case f(head) {
        Ok(value) -> do_list_try_map(rest, f, [value, ..acc])
        Error(err) -> Error(err)
      }
  }
}

// Wrappers around float ops to avoid leaking gleam_community/maths inside
// the hot path — these are inlined by the compiler.
fn ffi_log(x: Float) -> Float {
  let assert Ok(v) = float.logarithm(x)
  v
}

fn ffi_exp(x: Float) -> Float {
  float.exponential(x)
}
