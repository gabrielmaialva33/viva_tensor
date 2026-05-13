//// Numerical comparison helpers for reference tests.
////
//// Mirrors NumPy's `np.allclose`:
////   |actual - expected| <= atol + rtol * |expected|
////
//// Tests that need to compare a viva_tensor `Tensor` against a NumPy reference
//// (shape + flat data) should go through `assert_close` here so failure
//// messages stay consistent across the suite.

import gleam/float
import gleam/int
import gleam/list
import gleam/string
import gleeunit/should
import viva_tensor as t

/// Assert `actual` is element-wise close to NumPy's `expected_data` / shape.
///
/// Uses the same tolerance rule as `np.allclose`:
///   `|expected - actual| <= atol + rtol * |expected|`
///
/// Both shape and length mismatches fail with a descriptive message before any
/// element-wise comparison happens, so callers can rely on a clean diff once
/// shapes match.
pub fn assert_close(
  actual: t.Tensor,
  expected_data: List(Float),
  expected_shape: List(Int),
  rtol: Float,
  atol: Float,
) -> Nil {
  let actual_shape = t.shape(actual)
  case actual_shape == expected_shape {
    True -> Nil
    False -> {
      let msg =
        "shape mismatch: actual="
        <> shape_to_string(actual_shape)
        <> " expected="
        <> shape_to_string(expected_shape)
      // force failure with a useful label
      should.equal(msg, "")
    }
  }

  let actual_data = t.to_list(actual)
  case list.length(actual_data) == list.length(expected_data) {
    True -> Nil
    False -> {
      let msg =
        "length mismatch: actual="
        <> int.to_string(list.length(actual_data))
        <> " expected="
        <> int.to_string(list.length(expected_data))
      should.equal(msg, "")
    }
  }

  let pairs = list.zip(actual_data, expected_data)
  case first_mismatch(pairs, rtol, atol, 0) {
    Ok(Nil) -> Nil
    Error(detail) -> should.equal(detail, "")
  }
}

/// Same tolerance rule but for a single scalar result (e.g. `sum`, `mean`).
pub fn assert_scalar_close(
  actual: Float,
  expected: Float,
  rtol: Float,
  atol: Float,
) -> Nil {
  case close(actual, expected, rtol, atol) {
    True -> Nil
    False -> {
      let detail =
        "scalar mismatch: actual="
        <> float.to_string(actual)
        <> " expected="
        <> float.to_string(expected)
        <> " (rtol="
        <> float.to_string(rtol)
        <> " atol="
        <> float.to_string(atol)
        <> ")"
      should.equal(detail, "")
    }
  }
}

fn first_mismatch(
  pairs: List(#(Float, Float)),
  rtol: Float,
  atol: Float,
  index: Int,
) -> Result(Nil, String) {
  case pairs {
    [] -> Ok(Nil)
    [#(a, e), ..rest] ->
      case close(a, e, rtol, atol) {
        True -> first_mismatch(rest, rtol, atol, index + 1)
        False -> {
          let detail =
            "element ["
            <> int.to_string(index)
            <> "] mismatch: actual="
            <> float.to_string(a)
            <> " expected="
            <> float.to_string(e)
            <> " |diff|="
            <> float.to_string(float.absolute_value(a -. e))
            <> " tol="
            <> float.to_string(atol +. rtol *. float.absolute_value(e))
          Error(detail)
        }
      }
  }
}

fn close(actual: Float, expected: Float, rtol: Float, atol: Float) -> Bool {
  let diff = float.absolute_value(actual -. expected)
  let bound = atol +. rtol *. float.absolute_value(expected)
  diff <=. bound
}

fn shape_to_string(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), ", ") <> "]"
}
