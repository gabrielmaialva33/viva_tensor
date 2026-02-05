//// Core Tensor Type - N-dimensional arrays for numerical computing
////
//// Design principles:
//// - Opaque type prevents invalid state construction
//// - Smart constructors validate inputs
//// - NumPy-inspired API with strides for zero-copy views
//// - O(1) random access via Erlang :array
////
//// ## Example
//// ```gleam
//// import viva_tensor/core/tensor
////
//// // Create tensors with validated constructors
//// let a = tensor.zeros([2, 3])
//// let b = tensor.ones([2, 3])
////
//// // Operations return Results for safety
//// use c <- result.try(tensor.add(a, b))
//// use d <- result.try(tensor.matmul(a, tensor.transpose(b)))
//// ```

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/ffi.{type ErlangArray}

// =============================================================================
// OPAQUE TENSOR TYPE
// =============================================================================

/// Tensor with NumPy-style strides for zero-copy views
///
/// Opaque type - can only be constructed via module functions.
/// This ensures all tensors are in a valid state.
///
/// Internal representation:
/// - Dense: List-backed for simple operations
/// - Strided: Array-backed for O(1) access and zero-copy views
pub opaque type Tensor {
  /// Dense tensor - simple list storage
  Dense(data: List(Float), shape: List(Int))
  /// Strided tensor - O(1) access with view support
  Strided(
    storage: ErlangArray,
    shape: List(Int),
    strides: List(Int),
    offset: Int,
  )
}

// =============================================================================
// SMART CONSTRUCTORS
// =============================================================================

/// Create tensor from data and shape with validation
pub fn new(data: List(Float), shape: List(Int)) -> Result(Tensor, TensorError) {
  let expected_size = compute_size(shape)
  let actual_size = list.length(data)

  case expected_size == actual_size {
    True -> Ok(Dense(data: data, shape: shape))
    False ->
      Error(error.InvalidShape(
        "Data size "
        <> int.to_string(actual_size)
        <> " doesn't match shape "
        <> error.shape_to_string(shape)
        <> " (expected "
        <> int.to_string(expected_size)
        <> ")",
      ))
  }
}

/// Create tensor of zeros
pub fn zeros(shape: List(Int)) -> Tensor {
  let size = compute_size(shape)
  Dense(data: list.repeat(0.0, size), shape: shape)
}

/// Create tensor of ones
pub fn ones(shape: List(Int)) -> Tensor {
  let size = compute_size(shape)
  Dense(data: list.repeat(1.0, size), shape: shape)
}

/// Create tensor filled with a value
pub fn fill(shape: List(Int), value: Float) -> Tensor {
  let size = compute_size(shape)
  Dense(data: list.repeat(value, size), shape: shape)
}

/// Create 1D tensor (vector) from list
pub fn from_list(data: List(Float)) -> Tensor {
  Dense(data: data, shape: [list.length(data)])
}

/// Create 2D tensor (matrix) from list of lists
pub fn from_list2d(rows: List(List(Float))) -> Result(Tensor, TensorError) {
  case rows {
    [] -> Ok(Dense(data: [], shape: [0, 0]))
    [first, ..rest] -> {
      let cols = list.length(first)
      let valid = list.all(rest, fn(row) { list.length(row) == cols })

      case valid {
        False -> Error(error.InvalidShape("Rows have different lengths"))
        True -> {
          let data = list.flatten(rows)
          let num_rows = list.length(rows)
          Ok(Dense(data: data, shape: [num_rows, cols]))
        }
      }
    }
  }
}

/// Create vector (alias for from_list)
pub fn vector(data: List(Float)) -> Tensor {
  from_list(data)
}

/// Create matrix with explicit dimensions
pub fn matrix(
  rows rows: Int,
  cols cols: Int,
  data data: List(Float),
) -> Result(Tensor, TensorError) {
  new(data, [rows, cols])
}

/// Create identity matrix
pub fn eye(n: Int) -> Tensor {
  let data =
    list.range(0, n - 1)
    |> list.flat_map(fn(i) {
      list.range(0, n - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  Dense(data: data, shape: [n, n])
}

/// Create tensor with values from start to end (exclusive)
pub fn arange(start: Float, end: Float, step: Float) -> Tensor {
  let data = arange_loop(start, end, step, [])
  from_list(list.reverse(data))
}

fn arange_loop(
  current: Float,
  end: Float,
  step: Float,
  acc: List(Float),
) -> List(Float) {
  case current >=. end {
    True -> acc
    False -> arange_loop(current +. step, end, step, [current, ..acc])
  }
}

/// Create linearly spaced tensor
pub fn linspace(start: Float, end: Float, num: Int) -> Tensor {
  case num <= 1 {
    True -> from_list([start])
    False -> {
      let step = { end -. start } /. int.to_float(num - 1)
      let data =
        list.range(0, num - 1)
        |> list.map(fn(i) { start +. int.to_float(i) *. step })
      from_list(data)
    }
  }
}

// =============================================================================
// RANDOM CONSTRUCTORS
// =============================================================================

/// Tensor with uniform random values in [0, 1)
pub fn random_uniform(shape: List(Int)) -> Tensor {
  let size = compute_size(shape)
  let data =
    list.range(1, size)
    |> list.map(fn(_) { ffi.random_uniform() })
  Dense(data: data, shape: shape)
}

/// Tensor with normal random values (Box-Muller approximation)
pub fn random_normal(
  shape shape: List(Int),
  mean mean: Float,
  std std: Float,
) -> Tensor {
  let size = compute_size(shape)
  let data =
    list.range(1, size)
    |> list.map(fn(_) {
      let u1 = float.max(ffi.random_uniform(), 0.0001)
      let u2 = ffi.random_uniform()
      let z = ffi.sqrt(-2.0 *. ffi.log(u1)) *. ffi.cos(2.0 *. ffi.pi *. u2)
      mean +. z *. std
    })
  Dense(data: data, shape: shape)
}

/// Xavier/Glorot initialization for neural network weights
pub fn xavier_init(fan_in fan_in: Int, fan_out fan_out: Int) -> Tensor {
  let limit = ffi.sqrt(6.0 /. int.to_float(fan_in + fan_out))
  let data =
    list.range(1, fan_in * fan_out)
    |> list.map(fn(_) {
      let r = ffi.random_uniform()
      r *. 2.0 *. limit -. limit
    })
  Dense(data: data, shape: [fan_out, fan_in])
}

/// He initialization for ReLU networks
pub fn he_init(fan_in fan_in: Int, fan_out fan_out: Int) -> Tensor {
  let std = ffi.sqrt(2.0 /. int.to_float(fan_in))
  random_normal(shape: [fan_out, fan_in], mean: 0.0, std: std)
}

// =============================================================================
// ACCESSORS (Read-only access to internal state)
// =============================================================================

/// Get tensor shape
pub fn shape(t: Tensor) -> List(Int) {
  case t {
    Dense(_, s) -> s
    Strided(_, s, _, _) -> s
  }
}

/// Get tensor data as list
pub fn to_list(t: Tensor) -> List(Float) {
  case t {
    Dense(data, _) -> data
    Strided(storage, shp, strides, offset) -> {
      let total_size = compute_size(shp)
      list.range(0, total_size - 1)
      |> list.map(fn(flat_idx) {
        let indices = flat_to_multi(flat_idx, shp)
        let idx =
          list.zip(indices, strides)
          |> list.fold(offset, fn(acc, pair) {
            let #(i, s) = pair
            acc + i * s
          })
        ffi.array_get(storage, idx)
      })
    }
  }
}

/// Total number of elements
pub fn size(t: Tensor) -> Int {
  compute_size(shape(t))
}

/// Number of dimensions (rank)
pub fn rank(t: Tensor) -> Int {
  list.length(shape(t))
}

/// Get specific dimension size
pub fn dim(t: Tensor, axis: Int) -> Result(Int, TensorError) {
  list_at(shape(t), axis)
  |> result.map_error(fn(_) {
    error.DimensionError("Axis " <> int.to_string(axis) <> " out of bounds")
  })
}

/// Number of rows (for 2D tensors)
pub fn rows(t: Tensor) -> Int {
  case shape(t) {
    [r, ..] -> r
    [] -> 0
  }
}

/// Number of columns (for 2D tensors)
pub fn cols(t: Tensor) -> Int {
  case shape(t) {
    [_, c, ..] -> c
    [n] -> n
    [] -> 0
  }
}

// =============================================================================
// ELEMENT ACCESS
// =============================================================================

/// Get element by linear index
pub fn get(t: Tensor, index: Int) -> Result(Float, TensorError) {
  case t {
    Dense(data, _) ->
      list_at_float(data, index)
      |> result.map_error(fn(_) { error.IndexOutOfBounds(index, size(t)) })

    Strided(storage, shp, strides, offset) -> {
      let indices = flat_to_multi(index, shp)
      let flat_idx =
        list.zip(indices, strides)
        |> list.fold(offset, fn(acc, pair) {
          let #(i, s) = pair
          acc + i * s
        })
      Ok(ffi.array_get(storage, flat_idx))
    }
  }
}

/// Get element by 2D coordinates
pub fn get2d(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  case shape(t) {
    [_rows, num_cols] -> get(t, row * num_cols + col)
    _ -> Error(error.DimensionError("Tensor is not 2D"))
  }
}

/// Get matrix row as vector
pub fn get_row(t: Tensor, row_idx: Int) -> Result(Tensor, TensorError) {
  case shape(t) {
    [num_rows, num_cols] -> {
      case row_idx >= 0 && row_idx < num_rows {
        True -> {
          let data = to_list(t)
          let start = row_idx * num_cols
          let row_data =
            data
            |> list.drop(start)
            |> list.take(num_cols)
          Ok(from_list(row_data))
        }
        False -> Error(error.IndexOutOfBounds(row_idx, num_rows))
      }
    }
    _ -> Error(error.DimensionError("Tensor is not 2D"))
  }
}

/// Get matrix column as vector
pub fn get_col(t: Tensor, col_idx: Int) -> Result(Tensor, TensorError) {
  case shape(t) {
    [num_rows, num_cols] -> {
      case col_idx >= 0 && col_idx < num_cols {
        True -> {
          let col_data =
            list.range(0, num_rows - 1)
            |> list.filter_map(fn(row) { get2d(t, row, col_idx) })
          Ok(from_list(col_data))
        }
        False -> Error(error.IndexOutOfBounds(col_idx, num_cols))
      }
    }
    _ -> Error(error.DimensionError("Tensor is not 2D"))
  }
}

// =============================================================================
// STRIDED TENSOR OPERATIONS (Zero-copy)
// =============================================================================

/// Convert to strided tensor for O(1) access
pub fn to_strided(t: Tensor) -> Tensor {
  case t {
    Strided(_, _, _, _) -> t
    Dense(data, shp) -> {
      let storage = ffi.list_to_array(data)
      let strides = compute_strides(shp)
      Strided(storage: storage, shape: shp, strides: strides, offset: 0)
    }
  }
}

/// Convert strided tensor back to dense (materializes the view)
pub fn to_dense(t: Tensor) -> Tensor {
  case t {
    Dense(_, _) -> t
    Strided(_, _, _, _) -> {
      let data = to_list(t)
      Dense(data: data, shape: shape(t))
    }
  }
}

/// Alias for to_dense - ensure contiguous memory layout
pub fn to_contiguous(t: Tensor) -> Tensor {
  to_dense(t)
}

/// Check if tensor has contiguous memory layout
pub fn is_contiguous(t: Tensor) -> Bool {
  case t {
    Dense(_, _) -> True
    Strided(_, shp, strides, _) -> {
      let expected_strides = compute_strides(shp)
      strides == expected_strides
    }
  }
}

/// Zero-copy transpose (just swap strides and shape)
pub fn transpose_strided(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    Dense(_, shp) -> {
      case shp {
        [_m, _n] -> {
          let strided = to_strided(t)
          transpose_strided(strided)
        }
        _ -> Error(error.DimensionError("Transpose requires 2D tensor"))
      }
    }
    Strided(storage, shp, strides, offset) -> {
      case shp, strides {
        [m, n], [s0, s1] -> {
          Ok(Strided(
            storage: storage,
            shape: [n, m],
            strides: [s1, s0],
            offset: offset,
          ))
        }
        _, _ -> Error(error.DimensionError("Transpose requires 2D tensor"))
      }
    }
  }
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

fn compute_size(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, dim) { acc * dim })
}

fn compute_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(strides, _) =
    list.fold(reversed, #([], 1), fn(acc, dim) {
      let #(s, running) = acc
      #([running, ..s], running * dim)
    })
  strides
}

fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(indices, _) =
    list.fold(reversed, #([], flat), fn(acc, dim) {
      let #(idxs, remaining) = acc
      let idx = remaining % dim
      let next = remaining / dim
      #([idx, ..idxs], next)
    })
  indices
}

fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  list_at(lst, index)
}
