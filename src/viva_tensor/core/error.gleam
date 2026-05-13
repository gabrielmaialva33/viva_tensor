//// Centralized error types for tensor operations.
////
//// Philosophy: fail fast, fail loud, fail informatively.
////
//// Each error variant carries enough context to debug the issue without
//// printf-debugging. ShapeMismatch tells you both shapes. IndexOutOfBounds
//// tells you the index AND the size. No more "index out of bounds" with no context.
////
//// Why a single error type instead of operation-specific ones?
//// - Simpler API (one Result type everywhere)
//// - Easy to convert to user-facing messages
//// - Pattern matching still works for specific handling

import gleam/int
import gleam/list
import gleam/string

/// All the ways tensor operations can fail.
/// Tried to keep it minimal but expressive.
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

  /// Axis index out of bounds for the tensor's rank.
  ///
  /// Carries the operation name (e.g. `"sum_axis"`), the requested
  /// axis index, and the tensor's rank so callers know which dimension
  /// to fix.
  AxisOutOfBounds(operation: String, axis: Int, rank: Int)

  /// Operation requires a specific rank but received a different one.
  ///
  /// ## Example
  /// `get_row` requires a 2D tensor; passing a 3D tensor produces
  /// `RankMismatch("get_row", 2, [3, 4, 5])`.
  RankMismatch(operation: String, expected_rank: Int, got_shape: List(Int))

  /// An operand has the wrong shape for an operation that takes
  /// multiple tensors.
  ///
  /// ## Example
  /// `linear_relu`'s `input` doesn't match the weight matrix:
  /// `OperandShapeMismatch("linear_relu", "input", "[batch, in_features]", [4, 5])`.
  OperandShapeMismatch(
    operation: String,
    operand: String,
    expected: String,
    got: List(Int),
  )

  /// Slice start/length lists must have the same rank as the tensor
  /// they index into.
  ///
  /// ## Example
  /// `SliceArityMismatch(tensor_shape: [4, 5, 6], start: [0, 1], lengths: [2, 3])`
  /// tells the caller they passed 2 indices for a 3-D tensor.
  SliceArityMismatch(
    tensor_shape: List(Int),
    start: List(Int),
    lengths: List(Int),
  )

  /// Three-operand backend mismatch (typically `out = a OP b` where
  /// each tensor must live on the same accelerated backend).
  ///
  /// Backend names are passed as strings to avoid a circular
  /// dependency with `viva_tensor/native/cuda`.
  BackendMismatch(operation: String, out: String, lhs: String, rhs: String)

  /// Native NIF library is not loaded (no `priv/viva_tensor_zig.so`).
  ///
  /// Carries the operation that needed the native backend so callers
  /// know which feature degraded.
  NifNotLoaded(operation: String)
}

// --- Formatting -------------------------------------------------------------

/// Human-readable error message. Useful for debugging.
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

    AxisOutOfBounds(operation, axis, rank) ->
      operation
      <> ": axis "
      <> int.to_string(axis)
      <> " is out of bounds for tensor of rank "
      <> int.to_string(rank)
      <> " (valid: 0.."
      <> int.to_string(rank - 1)
      <> ")"

    RankMismatch(operation, expected_rank, got_shape) ->
      operation
      <> " requires a "
      <> int.to_string(expected_rank)
      <> "D tensor, got rank "
      <> int.to_string(list.length(got_shape))
      <> " with shape "
      <> shape_to_string(got_shape)

    OperandShapeMismatch(operation, operand, expected, got) ->
      operation
      <> ": "
      <> operand
      <> " has shape "
      <> shape_to_string(got)
      <> ", expected "
      <> expected

    SliceArityMismatch(tensor_shape, start, lengths) ->
      "slice: start and length must have rank "
      <> int.to_string(list.length(tensor_shape))
      <> " (tensor shape "
      <> shape_to_string(tensor_shape)
      <> "); got start="
      <> shape_to_string(start)
      <> ", length="
      <> shape_to_string(lengths)

    BackendMismatch(operation, out, lhs, rhs) ->
      operation
      <> ": backend mismatch — out="
      <> out
      <> ", lhs="
      <> lhs
      <> ", rhs="
      <> rhs

    NifNotLoaded(operation) ->
      operation
      <> ": native NIF library not loaded (build with `make zig` or `gleam build` after configuring the Zig toolchain)"
  }
}

/// Pretty-print a shape like [2, 3, 4].
pub fn shape_to_string(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), ", ") <> "]"
}
