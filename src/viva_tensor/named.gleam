//// Named Tensor - Tensors with semantic axis names
////
//// Wrap tensors with named axes for clearer, safer operations.
//// Instead of sum(t, axis: 0), write sum(t, along: Batch)

import gleam/int
import gleam/list
import gleam/string
import viva_tensor/axis.{
  type Axis, type AxisSpec, Anon, AxisSpec, equals as axis_equals,
  to_string as axis_to_string,
}
import viva_tensor/tensor.{type Tensor, type TensorError}

// =============================================================================
// TYPES
// =============================================================================

/// Tensor with named axes
pub type NamedTensor {
  NamedTensor(
    /// Underlying data tensor
    data: Tensor,
    /// Axis specifications (names + sizes, in order)
    axes: List(AxisSpec),
  )
}

/// Error types for named tensor operations
pub type NamedTensorError {
  /// Axis not found
  AxisNotFound(name: Axis)
  /// Duplicate axis name
  DuplicateAxis(name: Axis)
  /// Axis mismatch in operation
  AxisMismatch(expected: Axis, got: Axis)
  /// Size mismatch for same axis
  SizeMismatch(axis: Axis, expected: Int, got: Int)
  /// Cannot broadcast axes
  BroadcastErr(reason: String)
  /// Underlying tensor error
  TensorErr(TensorError)
  /// Invalid operation
  InvalidOp(reason: String)
}

// =============================================================================
// CONSTRUCTORS
// =============================================================================

/// Create named tensor from data and axis specs
pub fn new(
  data: Tensor,
  axes: List(AxisSpec),
) -> Result(NamedTensor, NamedTensorError) {
  let data_rank = tensor.rank(data)
  let axes_count = list.length(axes)

  case data_rank == axes_count {
    False ->
      Error(InvalidOp(
        "Axis count ("
        <> int.to_string(axes_count)
        <> ") doesn't match tensor rank ("
        <> int.to_string(data_rank)
        <> ")",
      ))
    True -> {
      case validate_sizes(data.shape, axes) {
        Error(e) -> Error(e)
        Ok(_) -> {
          case validate_unique_names(axes) {
            Error(e) -> Error(e)
            Ok(_) -> Ok(NamedTensor(data: data, axes: axes))
          }
        }
      }
    }
  }
}

/// Create from tensor with inferred anonymous axes
pub fn from_tensor(t: Tensor) -> NamedTensor {
  let axes = list.map(t.shape, fn(size) { AxisSpec(name: Anon, size: size) })
  NamedTensor(data: t, axes: axes)
}

/// Create named tensor of zeros
pub fn zeros(axes: List(AxisSpec)) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.zeros(shape)
  NamedTensor(data: data, axes: axes)
}

/// Create named tensor of ones
pub fn ones(axes: List(AxisSpec)) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.ones(shape)
  NamedTensor(data: data, axes: axes)
}

/// Create named tensor with random values [0, 1)
pub fn random(axes: List(AxisSpec)) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.random_uniform(shape)
  NamedTensor(data: data, axes: axes)
}

/// Create named tensor with normal distribution
pub fn randn(axes: List(AxisSpec), mean: Float, std: Float) -> NamedTensor {
  let shape = list.map(axes, fn(a) { a.size })
  let data = tensor.random_normal(shape, mean, std)
  NamedTensor(data: data, axes: axes)
}

// =============================================================================
// AXIS OPERATIONS
// =============================================================================

/// Find axis index by name
pub fn find_axis(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  find_axis_in_list(t.axes, name, 0)
}

fn find_axis_in_list(
  axes: List(AxisSpec),
  name: Axis,
  idx: Int,
) -> Result(Int, NamedTensorError) {
  case axes {
    [] -> Error(AxisNotFound(name))
    [first, ..rest] -> {
      case axis_equals(first.name, name) {
        True -> Ok(idx)
        False -> find_axis_in_list(rest, name, idx + 1)
      }
    }
  }
}

/// Get axis size by name
pub fn axis_size(t: NamedTensor, name: Axis) -> Result(Int, NamedTensorError) {
  case find_axis(t, name) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case list_at(t.axes, idx) {
        Error(_) -> Error(AxisNotFound(name))
        Ok(spec) -> Ok(spec.size)
      }
    }
  }
}

/// Check if tensor has axis
pub fn has_axis(t: NamedTensor, name: Axis) -> Bool {
  case find_axis(t, name) {
    Ok(_) -> True
    Error(_) -> False
  }
}

/// Get all axis names
pub fn axis_names(t: NamedTensor) -> List(Axis) {
  list.map(t.axes, fn(a) { a.name })
}

/// Get shape as list
pub fn shape(t: NamedTensor) -> List(Int) {
  t.data.shape
}

/// Get rank (number of dimensions)
pub fn rank(t: NamedTensor) -> Int {
  list.length(t.axes)
}

/// Total number of elements
pub fn size(t: NamedTensor) -> Int {
  tensor.size(t.data)
}

/// Rename an axis
pub fn rename_axis(
  t: NamedTensor,
  from: Axis,
  to: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, from) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      let new_axes =
        list.index_map(t.axes, fn(spec, i) {
          case i == idx {
            True -> AxisSpec(..spec, name: to)
            False -> spec
          }
        })
      Ok(NamedTensor(..t, axes: new_axes))
    }
  }
}

/// Add a new axis of size 1
pub fn unsqueeze(t: NamedTensor, name: Axis, position: Int) -> NamedTensor {
  let new_spec = AxisSpec(name: name, size: 1)
  let #(before, after) = list.split(t.axes, position)
  let new_axes = list.flatten([before, [new_spec], after])
  let new_data = tensor.unsqueeze(t.data, position)
  NamedTensor(data: new_data, axes: new_axes)
}

/// Remove axis of size 1 by name
pub fn squeeze(
  t: NamedTensor,
  name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, name) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case list_at(t.axes, idx) {
        Error(_) -> Error(AxisNotFound(name))
        Ok(spec) -> {
          case spec.size == 1 {
            False -> Error(InvalidOp("Cannot squeeze axis with size != 1"))
            True -> {
              case tensor.squeeze_axis(t.data, idx) {
                Error(e) -> Error(TensorErr(e))
                Ok(squeezed) -> {
                  let new_axes = remove_at(t.axes, idx)
                  Ok(NamedTensor(data: squeezed, axes: new_axes))
                }
              }
            }
          }
        }
      }
    }
  }
}

// =============================================================================
// OPERATIONS
// =============================================================================

/// Sum along named axis
pub fn sum_along(
  t: NamedTensor,
  axis_name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, axis_name) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case tensor.sum_axis(t.data, idx) {
        Error(e) -> Error(TensorErr(e))
        Ok(summed) -> {
          let new_axes = remove_at(t.axes, idx)
          Ok(NamedTensor(data: summed, axes: new_axes))
        }
      }
    }
  }
}

/// Mean along named axis
pub fn mean_along(
  t: NamedTensor,
  axis_name: Axis,
) -> Result(NamedTensor, NamedTensorError) {
  case find_axis(t, axis_name) {
    Error(e) -> Error(e)
    Ok(idx) -> {
      case tensor.mean_axis(t.data, idx) {
        Error(e) -> Error(TensorErr(e))
        Ok(meaned) -> {
          let new_axes = remove_at(t.axes, idx)
          Ok(NamedTensor(data: meaned, axes: new_axes))
        }
      }
    }
  }
}

/// Element-wise add (same axes required)
pub fn add(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  case axis.specs_equal(a.axes, b.axes) {
    False -> Error(InvalidOp("Axes don't match"))
    True -> {
      case tensor.add(a.data, b.data) {
        Error(e) -> Error(TensorErr(e))
        Ok(result) -> Ok(NamedTensor(data: result, axes: a.axes))
      }
    }
  }
}

/// Element-wise mul (same axes required)
pub fn mul(
  a: NamedTensor,
  b: NamedTensor,
) -> Result(NamedTensor, NamedTensorError) {
  case axis.specs_equal(a.axes, b.axes) {
    False -> Error(InvalidOp("Axes don't match"))
    True -> {
      case tensor.mul(a.data, b.data) {
        Error(e) -> Error(TensorErr(e))
        Ok(result) -> Ok(NamedTensor(data: result, axes: a.axes))
      }
    }
  }
}

/// Scale by constant
pub fn scale(t: NamedTensor, s: Float) -> NamedTensor {
  NamedTensor(data: tensor.scale(t.data, s), axes: t.axes)
}

/// Map function over elements
pub fn map(t: NamedTensor, f: fn(Float) -> Float) -> NamedTensor {
  NamedTensor(data: tensor.map(t.data, f), axes: t.axes)
}

// =============================================================================
// CONVERSION
// =============================================================================

/// Convert to plain tensor (drop names)
pub fn to_tensor(t: NamedTensor) -> Tensor {
  t.data
}

/// Pretty print tensor info
pub fn describe(t: NamedTensor) -> String {
  let axes_str =
    t.axes
    |> list.map(fn(a) { axis_to_string(a.name) <> ":" <> int.to_string(a.size) })
    |> string.join(", ")

  "NamedTensor[" <> axes_str <> "]"
}

// =============================================================================
// HELPERS
// =============================================================================

fn validate_sizes(
  shape: List(Int),
  axes: List(AxisSpec),
) -> Result(Nil, NamedTensorError) {
  case shape, axes {
    [], [] -> Ok(Nil)
    [s, ..s_rest], [a, ..a_rest] -> {
      case s == a.size {
        True -> validate_sizes(s_rest, a_rest)
        False -> Error(SizeMismatch(a.name, a.size, s))
      }
    }
    _, _ -> Error(InvalidOp("Shape and axes length mismatch"))
  }
}

fn validate_unique_names(
  axes: List(AxisSpec),
) -> Result(Nil, NamedTensorError) {
  let named_axes =
    list.filter(axes, fn(a) {
      case a.name {
        Anon -> False
        _ -> True
      }
    })
  let names = list.map(named_axes, fn(a) { a.name })
  case has_duplicates(names) {
    True -> Error(DuplicateAxis(Anon))
    False -> Ok(Nil)
  }
}

fn has_duplicates(items: List(Axis)) -> Bool {
  case items {
    [] -> False
    [first, ..rest] -> {
      case list.any(rest, fn(x) { axis_equals(x, first) }) {
        True -> True
        False -> has_duplicates(rest)
      }
    }
  }
}

fn list_at(lst: List(a), idx: Int) -> Result(a, Nil) {
  lst
  |> list.drop(idx)
  |> list.first
}

fn remove_at(lst: List(a), idx: Int) -> List(a) {
  lst
  |> list.index_map(fn(item, i) { #(item, i) })
  |> list.filter(fn(pair) { pair.1 != idx })
  |> list.map(fn(pair) { pair.0 })
}
