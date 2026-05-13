//// Regression metrics: MAE, MSE, RMSE, R², and MAPE.
////
//// All functions expect `predictions` and `targets` to share the same
//// shape; they materialize the data via `tensor.try_to_list`, so any
//// tensor variant (dense, strided, native) works.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{
  type TensorError, DimensionError, InvalidShape, ShapeMismatch,
}
import viva_tensor/tensor.{type Tensor}

/// Mean Absolute Error.
///
/// Formula: `MAE = (1/N) * sum_i |pred_i - target_i|`.
pub fn mean_absolute_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  use pairs <- result.try(materialize_pairs(predictions, targets))
  case pairs {
    [] -> Error(InvalidShape("mean_absolute_error: empty inputs"))
    _ -> {
      let total =
        list.fold(pairs, 0.0, fn(acc, pair) {
          let #(p, t) = pair
          acc +. float.absolute_value(p -. t)
        })
      Ok(total /. int.to_float(list.length(pairs)))
    }
  }
}

/// Mean Squared Error.
///
/// Formula: `MSE = (1/N) * sum_i (pred_i - target_i)^2`.
pub fn mean_squared_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  use pairs <- result.try(materialize_pairs(predictions, targets))
  case pairs {
    [] -> Error(InvalidShape("mean_squared_error: empty inputs"))
    _ -> {
      let total =
        list.fold(pairs, 0.0, fn(acc, pair) {
          let #(p, t) = pair
          let d = p -. t
          acc +. d *. d
        })
      Ok(total /. int.to_float(list.length(pairs)))
    }
  }
}

/// Root Mean Squared Error.
///
/// Formula: `RMSE = sqrt(MSE)`.
pub fn root_mean_squared_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  use mse_value <- result.try(mean_squared_error(predictions, targets))
  case float.square_root(mse_value) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(DimensionError("root_mean_squared_error: negative MSE"))
  }
}

/// Coefficient of determination (R²).
///
/// Formula:
/// ```
/// SS_res = sum_i (target_i - pred_i)^2
/// SS_tot = sum_i (target_i - mean(target))^2
/// R^2    = 1 - SS_res / SS_tot
/// ```
/// Returns `1.0` when `SS_tot == 0` and the predictions match exactly,
/// `0.0` when `SS_tot == 0` but predictions disagree (degenerate target).
pub fn r_squared(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  use pairs <- result.try(materialize_pairs(predictions, targets))
  case pairs {
    [] -> Error(InvalidShape("r_squared: empty inputs"))
    _ -> {
      let targets_only = list.map(pairs, fn(pair) { pair.1 })
      let mean_t =
        list.fold(targets_only, 0.0, fn(acc, v) { acc +. v })
        /. int.to_float(list.length(targets_only))
      let ss_res =
        list.fold(pairs, 0.0, fn(acc, pair) {
          let #(p, t) = pair
          let d = t -. p
          acc +. d *. d
        })
      let ss_tot =
        list.fold(targets_only, 0.0, fn(acc, t) {
          let d = t -. mean_t
          acc +. d *. d
        })
      case ss_tot >. 0.0 {
        True -> Ok(1.0 -. ss_res /. ss_tot)
        False ->
          case ss_res >. 0.0 {
            True -> Ok(0.0)
            False -> Ok(1.0)
          }
      }
    }
  }
}

/// Mean Absolute Percentage Error.
///
/// Formula: `MAPE = (100/N) * sum_i |pred_i - target_i| / |target_i|`.
///
/// Returns `InvalidShape("MAPE: target contains zero values")` if any
/// target is exactly `0.0`.
pub fn mean_absolute_percentage_error(
  predictions: Tensor,
  targets: Tensor,
) -> Result(Float, TensorError) {
  use pairs <- result.try(materialize_pairs(predictions, targets))
  case pairs {
    [] -> Error(InvalidShape("mean_absolute_percentage_error: empty inputs"))
    _ -> {
      let has_zero =
        list.any(pairs, fn(pair) {
          let #(_, t) = pair
          t == 0.0
        })
      case has_zero {
        True -> Error(InvalidShape("MAPE: target contains zero values"))
        False -> {
          let total =
            list.fold(pairs, 0.0, fn(acc, pair) {
              let #(p, t) = pair
              acc +. float.absolute_value(p -. t) /. float.absolute_value(t)
            })
          Ok(100.0 *. total /. int.to_float(list.length(pairs)))
        }
      }
    }
  }
}

// --- Helpers ----------------------------------------------------------------

fn materialize_pairs(
  predictions: Tensor,
  targets: Tensor,
) -> Result(List(#(Float, Float)), TensorError) {
  let pred_shape = tensor.shape(predictions)
  let target_shape = tensor.shape(targets)
  case pred_shape == target_shape {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(expected: target_shape, got: pred_shape))
  }
  |> result.try(fn(_) {
    use pred_data <- result.try(tensor.try_to_list(predictions))
    use target_data <- result.try(tensor.try_to_list(targets))
    Ok(list.zip(pred_data, target_data))
  })
}
