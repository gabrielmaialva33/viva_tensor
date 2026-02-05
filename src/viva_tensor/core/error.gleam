//// Centralized Error Types - All tensor operation errors
////
//// This module provides a unified error type for all tensor operations,
//// ensuring consistent error handling across the library.
////
//// ## Usage
////
//// ```gleam
//// import viva_tensor/core/error.{type TensorError}
////
//// pub fn my_operation(t: Tensor) -> Result(Tensor, TensorError) {
////   case validate(t) {
////     True -> Ok(transform(t))
////     False -> Error(error.InvalidShape("Must be 2D"))
////   }
//// }
//// ```
////
//// ## Error Formatting
////
//// ```gleam
//// case ops.add(a, b) {
////   Ok(result) -> result
////   Error(e) -> panic as error.to_string(e)
//// }
//// ```

import gleam/int
import gleam/list
import gleam/string

// =============================================================================
// TENSOR ERRORS
// =============================================================================

/// Comprehensive error type for all tensor operations.
///
/// Each variant provides detailed context for debugging:
/// - `ShapeMismatch` - Two tensors have incompatible shapes
/// - `InvalidShape` - Shape specification is invalid
/// - `DimensionError` - Axis out of bounds or dimension issue
/// - `BroadcastError` - Shapes cannot be broadcast together
/// - `IndexOutOfBounds` - Element access outside valid range
/// - `DtypeError` - Data type incompatibility
pub type TensorError {
  /// Shape mismatch between two tensors.
  ///
  /// ## Example
  /// Trying to add [2, 3] tensor with [4, 5] tensor.
  ShapeMismatch(expected: List(Int), got: List(Int))

  /// Invalid shape specification.
  ///
  /// ## Example
  /// Data size doesn't match shape dimensions.
  InvalidShape(reason: String)

  /// Dimension-related error (axis out of bounds, etc.).
  ///
  /// ## Example
  /// Accessing axis 3 on a 2D tensor.
  DimensionError(reason: String)

  /// Broadcasting incompatibility.
  ///
  /// ## Example
  /// Cannot broadcast [2, 3] with [4, 5].
  BroadcastError(shape_a: List(Int), shape_b: List(Int))

  /// Index out of bounds.
  ///
  /// ## Example
  /// Accessing index 10 in tensor of size 5.
  IndexOutOfBounds(index: Int, size: Int)

  /// Invalid dtype for operation.
  ///
  /// ## Example
  /// Using INT8 operation on Float32 tensor.
  DtypeError(reason: String)
}

// =============================================================================
// ERROR FORMATTING
// =============================================================================

/// Convert error to human-readable string.
///
/// ## Examples
///
/// ```gleam
/// error.to_string(ShapeMismatch([2, 3], [4, 5]))
/// // -> "Shape mismatch: expected [2, 3], got [4, 5]"
/// ```
pub fn to_string(error: TensorError) -> String {
  case error {
    ShapeMismatch(expected, got) ->
      "Shape mismatch: expected "
      <> shape_to_string(expected)
      <> ", got "
      <> shape_to_string(got)

    InvalidShape(reason) -> "Invalid shape: " <> reason

    DimensionError(reason) -> "Dimension error: " <> reason

    BroadcastError(a, b) ->
      "Cannot broadcast shapes "
      <> shape_to_string(a)
      <> " and "
      <> shape_to_string(b)

    IndexOutOfBounds(index, size) ->
      "Index "
      <> int.to_string(index)
      <> " out of bounds for size "
      <> int.to_string(size)

    DtypeError(reason) -> "Dtype error: " <> reason
  }
}

/// Format shape as string [d0, d1, ...].
///
/// ## Examples
///
/// ```gleam
/// error.shape_to_string([2, 3, 4])
/// // -> "[2, 3, 4]"
/// ```
pub fn shape_to_string(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), ", ") <> "]"
}
