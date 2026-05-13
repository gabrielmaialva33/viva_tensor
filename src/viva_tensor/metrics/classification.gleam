//// Classification metrics: accuracy, confusion matrix, precision, recall,
//// F1-score, top-k accuracy, and per-class IoU.
////
//// All inputs are tensors of class indices (`Float` values that round to
//// non-negative integers in `0..num_classes`). The metrics return scalar
//// floats or per-class lists, and use the shared `TensorError` type so they
//// compose with the rest of the API.

import gleam/float
import gleam/int
import gleam/list
import gleam/order
import gleam/result
import viva_tensor/core/error.{
  type TensorError, IndexOutOfBounds, InvalidShape, ShapeMismatch,
}
import viva_tensor/tensor.{type Tensor}

// --- Types ------------------------------------------------------------------

/// Averaging strategy for multi-class precision / recall / F1.
///
/// * `Micro` — aggregate true-positives and the matching false-positives /
///   false-negatives globally before computing the ratio.
/// * `Macro` — compute the per-class metric and return the unweighted mean.
/// * `Weighted` — like `Macro`, but each class is weighted by its support
///   (number of true samples in the target tensor).
pub type Average {
  Micro
  Macro
  Weighted
}

// --- Accuracy ---------------------------------------------------------------

/// Classification accuracy.
///
/// Formula: `accuracy = (1/N) * sum_i [pred_i == target_i]`.
///
/// Both inputs must be 1D class-index tensors with identical shape.
pub fn accuracy(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  use #(preds, tgts) <- result.try(pair_indices(predictions, targets))
  let n = list.length(preds)
  case n {
    0 -> Error(InvalidShape("accuracy: empty inputs"))
    _ -> {
      let matches =
        list.zip(preds, tgts)
        |> list.fold(0, fn(acc, pair) {
          let #(p, t) = pair
          case p == t {
            True -> acc + 1
            False -> acc
          }
        })
      Ok(int.to_float(matches) /. int.to_float(n))
    }
  }
}

// --- Confusion matrix -------------------------------------------------------

/// Confusion matrix of shape `[num_classes, num_classes]` where
/// `cm[true, pred]` counts how many samples with true class `true` were
/// predicted as `pred`.
///
/// Both inputs are 1D and must have the same length. Returns
/// `IndexOutOfBounds` if any class index is `>= num_classes`.
pub fn confusion_matrix(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(case num_classes > 0 {
    True -> Ok(Nil)
    False -> Error(InvalidShape("confusion_matrix: num_classes must be > 0"))
  })
  use #(preds, tgts) <- result.try(pair_indices(predictions, targets))
  use counts <- result.try(build_counts(preds, tgts, num_classes))
  tensor.matrix(num_classes, num_classes, counts)
}

// --- Precision / Recall / F1 ------------------------------------------------

/// Precision: `TP / (TP + FP)` per class, aggregated by `average`.
///
/// * `Micro`: `sum(TP) / (sum(TP) + sum(FP))`.
/// * `Macro`: unweighted mean of per-class precision.
/// * `Weighted`: mean of per-class precision weighted by class support
///   (counts in `targets`).
pub fn precision(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
  average: Average,
) -> Result(Float, TensorError) {
  use stats <- result.try(class_stats(predictions, targets, num_classes))
  let ClassStats(tp, fp, _fn, support) = stats
  Ok(aggregate(tp, fp, support, average))
}

/// Recall: `TP / (TP + FN)` per class, aggregated by `average`.
///
/// Aggregation modes mirror `precision`.
pub fn recall(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
  average: Average,
) -> Result(Float, TensorError) {
  use stats <- result.try(class_stats(predictions, targets, num_classes))
  let ClassStats(tp, _fp, fn_, support) = stats
  Ok(aggregate(tp, fn_, support, average))
}

/// F1 score: harmonic mean of precision and recall.
///
/// Per-class: `f1_c = 2 * P_c * R_c / (P_c + R_c)`, returning 0.0 when
/// `P_c + R_c = 0`. Aggregation follows `average` (`Micro` returns the
/// micro-precision == micro-recall == micro-F1 identity).
pub fn f1(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
  average: Average,
) -> Result(Float, TensorError) {
  use stats <- result.try(class_stats(predictions, targets, num_classes))
  let ClassStats(tp, fp, fn_, support) = stats
  case average {
    Micro -> {
      // micro-F1 == micro-precision == micro-recall
      let sum_tp = int_sum(tp)
      let sum_fp = int_sum(fp)
      let sum_fn = int_sum(fn_)
      Ok(safe_ratio(
        int.to_float(sum_tp),
        int.to_float(sum_tp + { sum_fp + sum_fn } / 2),
      ))
    }
    Macro -> {
      let per_class = per_class_f1(tp, fp, fn_)
      Ok(mean_floats(per_class))
    }
    Weighted -> {
      let per_class = per_class_f1(tp, fp, fn_)
      Ok(weighted_mean(per_class, support))
    }
  }
}

// --- Top-K accuracy ---------------------------------------------------------

/// Top-K accuracy: proportion of rows whose true class is among the top `k`
/// indices of the logits (ties broken by index order).
///
/// `logits` is `[batch, num_classes]`; `targets` is `[batch]`.
pub fn top_k_accuracy(
  logits: Tensor,
  targets: Tensor,
  k: Int,
) -> Result(Float, TensorError) {
  case k > 0 {
    True -> Ok(Nil)
    False -> Error(InvalidShape("top_k_accuracy: k must be > 0"))
  }
  |> result.try(fn(_) {
    case tensor.shape(logits) {
      [batch, num_classes] -> Ok(#(batch, num_classes))
      other ->
        Error(InvalidShape(
          "top_k_accuracy: logits must be 2D, got "
          <> error.shape_to_string(other),
        ))
    }
  })
  |> result.try(fn(dims) {
    let #(batch, num_classes) = dims
    case tensor.shape(targets) {
      [n] if n == batch -> Ok(#(batch, num_classes))
      other -> Error(ShapeMismatch(expected: [batch], got: other))
    }
  })
  |> result.try(fn(dims) {
    let #(batch, num_classes) = dims
    use logit_data <- result.try(tensor.try_to_list(logits))
    use target_data <- result.try(tensor.try_to_list(targets))
    use target_idx <- result.try(to_indices(target_data))
    let effective_k = case k > num_classes {
      True -> num_classes
      False -> k
    }
    let rows = chunk_rows(logit_data, num_classes)
    let hits =
      list.zip(rows, target_idx)
      |> list.fold(0, fn(acc, pair) {
        let #(row, t) = pair
        case row_topk_contains(row, effective_k, t) {
          True -> acc + 1
          False -> acc
        }
      })
    case batch {
      0 -> Error(InvalidShape("top_k_accuracy: empty batch"))
      _ -> Ok(int.to_float(hits) /. int.to_float(batch))
    }
  })
}

// --- IoU --------------------------------------------------------------------

/// Per-class intersection-over-union: `IoU_c = TP_c / (TP_c + FP_c + FN_c)`.
///
/// Classes that never appear in either input contribute `0.0`.
pub fn iou_per_class(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(List(Float), TensorError) {
  use stats <- result.try(class_stats(predictions, targets, num_classes))
  let ClassStats(tp, fp, fn_, _support) = stats
  let triples = list.zip(tp, list.zip(fp, fn_))
  let ious =
    list.map(triples, fn(triple) {
      let #(tp_c, #(fp_c, fn_c)) = triple
      let denom = tp_c + fp_c + fn_c
      case denom {
        0 -> 0.0
        _ -> int.to_float(tp_c) /. int.to_float(denom)
      }
    })
  Ok(ious)
}

/// Mean IoU across all classes.
///
/// Formula: `mIoU = (1/C) * sum_c IoU_c`.
pub fn mean_iou(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(Float, TensorError) {
  use ious <- result.try(iou_per_class(predictions, targets, num_classes))
  Ok(mean_floats(ious))
}

// --- Internal helpers -------------------------------------------------------

type ClassStats {
  ClassStats(tp: List(Int), fp: List(Int), fn_: List(Int), support: List(Int))
}

/// Validate equal 1D shapes and materialize both as integer index lists.
fn pair_indices(
  predictions: Tensor,
  targets: Tensor,
) -> Result(#(List(Int), List(Int)), TensorError) {
  let pred_shape = tensor.shape(predictions)
  let target_shape = tensor.shape(targets)
  case pred_shape == target_shape {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(expected: target_shape, got: pred_shape))
  }
  |> result.try(fn(_) {
    case pred_shape {
      [_] -> Ok(Nil)
      _ ->
        Error(InvalidShape(
          "expected 1D tensors, got " <> error.shape_to_string(pred_shape),
        ))
    }
  })
  |> result.try(fn(_) {
    use pred_data <- result.try(tensor.try_to_list(predictions))
    use target_data <- result.try(tensor.try_to_list(targets))
    use preds <- result.try(to_indices(pred_data))
    use tgts <- result.try(to_indices(target_data))
    Ok(#(preds, tgts))
  })
}

fn to_indices(xs: List(Float)) -> Result(List(Int), TensorError) {
  list.try_map(xs, fn(x) {
    let i = float.round(x)
    case i >= 0 {
      True -> Ok(i)
      False -> Error(IndexOutOfBounds(index: i, size: 0))
    }
  })
}

fn build_counts(
  preds: List(Int),
  tgts: List(Int),
  num_classes: Int,
) -> Result(List(Float), TensorError) {
  // counts is a flat row-major list of length num_classes * num_classes
  let size = num_classes * num_classes
  let zeros = list.repeat(0, size)
  let pairs = list.zip(tgts, preds)
  list.try_fold(pairs, zeros, fn(acc, pair) {
    let #(t, p) = pair
    case t >= num_classes {
      True -> Error(IndexOutOfBounds(index: t, size: num_classes))
      False ->
        case p >= num_classes {
          True -> Error(IndexOutOfBounds(index: p, size: num_classes))
          False -> Ok(increment_at(acc, t * num_classes + p))
        }
    }
  })
  |> result.map(fn(ints) { list.map(ints, int.to_float) })
}

fn increment_at(xs: List(Int), idx: Int) -> List(Int) {
  list.index_map(xs, fn(value, i) {
    case i == idx {
      True -> value + 1
      False -> value
    }
  })
}

fn class_stats(
  predictions: Tensor,
  targets: Tensor,
  num_classes: Int,
) -> Result(ClassStats, TensorError) {
  case num_classes > 0 {
    True -> Ok(Nil)
    False -> Error(InvalidShape("num_classes must be > 0"))
  }
  |> result.try(fn(_) { pair_indices(predictions, targets) })
  |> result.try(fn(pair) {
    let #(preds, tgts) = pair
    let zeros = list.repeat(0, num_classes)
    let folded =
      list.try_fold(
        list.zip(tgts, preds),
        #(zeros, zeros, zeros, zeros),
        fn(acc, pair) {
          let #(t, p) = pair
          let #(tp, fp, fn_, support) = acc
          case t >= num_classes {
            True -> Error(IndexOutOfBounds(index: t, size: num_classes))
            False ->
              case p >= num_classes {
                True -> Error(IndexOutOfBounds(index: p, size: num_classes))
                False -> {
                  let support2 = increment_at(support, t)
                  case t == p {
                    True -> Ok(#(increment_at(tp, t), fp, fn_, support2))
                    False ->
                      Ok(#(
                        tp,
                        increment_at(fp, p),
                        increment_at(fn_, t),
                        support2,
                      ))
                  }
                }
              }
          }
        },
      )
    use stats <- result.try(folded)
    let #(tp, fp, fn_, support) = stats
    Ok(ClassStats(tp: tp, fp: fp, fn_: fn_, support: support))
  })
}

fn aggregate(
  tp: List(Int),
  other: List(Int),
  support: List(Int),
  average: Average,
) -> Float {
  case average {
    Micro -> {
      let sum_tp = int_sum(tp)
      let sum_other = int_sum(other)
      safe_ratio(int.to_float(sum_tp), int.to_float(sum_tp + sum_other))
    }
    Macro -> {
      let per_class = per_class_ratio(tp, other)
      mean_floats(per_class)
    }
    Weighted -> {
      let per_class = per_class_ratio(tp, other)
      weighted_mean(per_class, support)
    }
  }
}

fn per_class_ratio(tp: List(Int), other: List(Int)) -> List(Float) {
  list.zip(tp, other)
  |> list.map(fn(pair) {
    let #(tp_c, other_c) = pair
    safe_ratio(int.to_float(tp_c), int.to_float(tp_c + other_c))
  })
}

fn per_class_f1(tp: List(Int), fp: List(Int), fn_: List(Int)) -> List(Float) {
  list.zip(tp, list.zip(fp, fn_))
  |> list.map(fn(triple) {
    let #(tp_c, #(fp_c, fn_c)) = triple
    let p = safe_ratio(int.to_float(tp_c), int.to_float(tp_c + fp_c))
    let r = safe_ratio(int.to_float(tp_c), int.to_float(tp_c + fn_c))
    case p +. r >. 0.0 {
      True -> 2.0 *. p *. r /. { p +. r }
      False -> 0.0
    }
  })
}

fn safe_ratio(num: Float, denom: Float) -> Float {
  case denom >. 0.0 {
    True -> num /. denom
    False -> 0.0
  }
}

fn int_sum(xs: List(Int)) -> Int {
  list.fold(xs, 0, fn(acc, v) { acc + v })
}

fn mean_floats(xs: List(Float)) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ -> list.fold(xs, 0.0, fn(acc, v) { acc +. v }) /. int.to_float(n)
  }
}

fn weighted_mean(values: List(Float), weights: List(Int)) -> Float {
  let total = int_sum(weights)
  case total {
    0 -> 0.0
    _ -> {
      let weighted =
        list.zip(values, weights)
        |> list.fold(0.0, fn(acc, pair) {
          let #(v, w) = pair
          acc +. v *. int.to_float(w)
        })
      weighted /. int.to_float(total)
    }
  }
}

fn chunk_rows(data: List(Float), cols: Int) -> List(List(Float)) {
  case data {
    [] -> []
    _ -> {
      let row = list.take(data, cols)
      let rest = list.drop(data, cols)
      [row, ..chunk_rows(rest, cols)]
    }
  }
}

/// Returns True when `target` is among the top-`k` indices in `row`
/// (descending order, ties broken by lower index winning).
fn row_topk_contains(row: List(Float), k: Int, target: Int) -> Bool {
  let indexed = list.index_map(row, fn(value, idx) { #(value, idx) })
  let sorted =
    list.sort(indexed, fn(a, b) {
      let #(va, ia) = a
      let #(vb, ib) = b
      case float.compare(vb, va) {
        order.Eq -> int.compare(ia, ib)
        ord -> ord
      }
    })
  sorted
  |> list.take(k)
  |> list.any(fn(pair) {
    let #(_, idx) = pair
    idx == target
  })
}
