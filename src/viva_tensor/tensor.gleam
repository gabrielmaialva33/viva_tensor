//// Tensor - N-dimensional arrays for numerical computing.
////
//// This module is the implementation-facing tensor API used by the stable
//// package facade in `viva_tensor`. It keeps storage variants explicit while
//// the top-level module provides the preferred import surface for users.
////
//// Design: NumPy-inspired with strides for zero-copy views.
//// Uses Erlang :array for O(1) access + strides for efficient transpose/reshape.
//// Optional native acceleration is used for large hot paths when the NIF is
//// available.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam_community/maths
import viva_tensor/core/error.{
  BroadcastError, DimensionError, IndexOutOfBounds, InvalidShape, ShapeMismatch,
}
import viva_tensor/core/ffi.{type ErlangArray, type NativeTensorRef}
import viva_tensor/core/layout_math
import viva_tensor/layout

// --- Types ---

/// Re-export TensorError so other modules can reference tensor.TensorError.
pub type TensorError =
  error.TensorError

/// Tensor with NumPy-style strides for zero-copy views
/// - storage: contiguous data buffer (Erlang array for O(1) access)
/// - shape: dimensions [d0, d1, ..., dn]
/// - strides: bytes to skip for each dimension [s0, s1, ..., sn]
/// - offset: starting position in storage (for views/slices)
///
/// Convolution helpers use NCHW layout by convention.
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
  StridedTensor(
    storage: ErlangArray,
    shape: List(Int),
    strides: List(Int),
    offset: Int,
  )
  NativeTensor(ref: NativeTensorRef, shape: List(Int))
}

// --- Constructors ---

/// Create tensor of zeros
pub fn zeros(shape: List(Int)) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  Tensor(data: list.repeat(0.0, size), shape: shape)
}

/// Create tensor of ones
pub fn ones(shape: List(Int)) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  Tensor(data: list.repeat(1.0, size), shape: shape)
}

/// Create tensor filled with value
pub fn fill(shape: List(Int), value: Float) -> Tensor {
  let size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  Tensor(data: list.repeat(value, size), shape: shape)
}

/// Create tensor from list (1D)
pub fn from_list(data: List(Float)) -> Tensor {
  Tensor(data: data, shape: [list.length(data)])
}

/// Create 2D tensor (matrix) from list of lists
pub fn from_list2d(rows: List(List(Float))) -> Result(Tensor, TensorError) {
  case rows {
    [] -> Ok(Tensor(data: [], shape: [0, 0]))
    [first, ..rest] -> {
      let cols = list.length(first)
      let valid = list.all(rest, fn(row) { list.length(row) == cols })

      case valid {
        False -> Error(InvalidShape("Rows have different lengths"))
        True -> {
          let data = list.flatten(rows)
          let num_rows = list.length(rows)
          Ok(Tensor(data: data, shape: [num_rows, cols]))
        }
      }
    }
  }
}

/// Create vector (1D tensor)
pub fn vector(data: List(Float)) -> Tensor {
  from_list(data)
}

/// Create a 1D tensor with evenly spaced values over a closed interval.
pub fn try_linspace(
  start: Float,
  stop: Float,
  steps: Int,
) -> Result(Tensor, TensorError) {
  case steps {
    n if n <= 0 -> Error(InvalidShape("linspace requires steps > 0"))
    1 -> Ok(from_list([start]))
    _ -> {
      case maths.linear_space(start, stop, steps, True) {
        Ok(data) -> Ok(from_list(data))
        Error(_) -> Error(InvalidShape("linspace requires steps > 0"))
      }
    }
  }
}

/// Create a 1D tensor with evenly spaced values over a closed interval.
pub fn linspace(start: Float, stop: Float, steps: Int) -> Tensor {
  try_linspace(start, stop, steps)
  |> result.unwrap(from_list([]))
}

/// Create a 1D tensor with logarithmically spaced values.
pub fn try_logspace(
  start: Float,
  stop: Float,
  steps: Int,
  base: Float,
) -> Result(Tensor, TensorError) {
  case steps <= 0 || base <=. 0.0 {
    True -> Error(InvalidShape("logspace requires steps > 0 and base > 0"))
    False -> {
      case maths.logarithmic_space(start, stop, steps, True, base) {
        Ok(data) -> Ok(from_list(data))
        Error(_) ->
          Error(InvalidShape("logspace requires steps > 0 and base > 0"))
      }
    }
  }
}

/// Create a 1D tensor with logarithmically spaced values.
pub fn logspace(start: Float, stop: Float, steps: Int, base: Float) -> Tensor {
  try_logspace(start, stop, steps, base)
  |> result.unwrap(from_list([]))
}

/// Create matrix (2D tensor) with explicit dimensions
pub fn matrix(
  rows: Int,
  cols: Int,
  data: List(Float),
) -> Result(Tensor, TensorError) {
  let expected_size = rows * cols
  let actual_size = list.length(data)

  case expected_size == actual_size {
    True -> Ok(Tensor(data: data, shape: [rows, cols]))
    False ->
      Error(InvalidShape(
        "Expected "
        <> int.to_string(expected_size)
        <> " elements, got "
        <> int.to_string(actual_size),
      ))
  }
}

/// Wrap an existing native NIF resource as a Tensor.
///
/// The caller is responsible for passing the correct shape for the resource.
pub fn from_native_ref(ref: NativeTensorRef, shape: List(Int)) -> Tensor {
  NativeTensor(ref: ref, shape: shape)
}

/// Extract the native NIF resource when this tensor is native-backed.
pub fn native_ref(t: Tensor) -> Result(NativeTensorRef, Nil) {
  case t {
    NativeTensor(ref, _) -> Ok(ref)
    _ -> Error(Nil)
  }
}

/// Check whether this tensor stores data in native NIF memory.
pub fn is_native(t: Tensor) -> Bool {
  case t {
    NativeTensor(_, _) -> True
    _ -> False
  }
}

/// Create a native-backed tensor of zeros.
pub fn native_zeros(shape: List(Int)) -> Result(Tensor, TensorError) {
  case ffi.nt_zeros(shape) {
    Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
    Error(reason) -> Error(InvalidShape(reason))
  }
}

/// Create a native-backed tensor of ones.
pub fn native_ones(shape: List(Int)) -> Result(Tensor, TensorError) {
  case ffi.nt_ones(shape) {
    Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
    Error(reason) -> Error(InvalidShape(reason))
  }
}

/// Create a native-backed tensor filled with a value.
pub fn native_fill(
  shape: List(Int),
  value: Float,
) -> Result(Tensor, TensorError) {
  case ffi.nt_fill(shape, value) {
    Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
    Error(reason) -> Error(InvalidShape(reason))
  }
}

/// Create a native-backed tensor from row-major list data.
pub fn native_from_list(
  data: List(Float),
  shape: List(Int),
) -> Result(Tensor, TensorError) {
  let expected_size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let actual_size = list.length(data)

  case expected_size == actual_size {
    False ->
      Error(InvalidShape(
        "Expected "
        <> int.to_string(expected_size)
        <> " elements, got "
        <> int.to_string(actual_size),
      ))
    True -> {
      case ffi.nt_from_list(data, shape) {
        Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
        Error(reason) -> Error(InvalidShape(reason))
      }
    }
  }
}

// --- Properties ---

/// Get tensor shape
pub fn shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, s) -> s
    StridedTensor(_, s, _, _) -> s
    NativeTensor(_, s) -> s
  }
}

/// Extract data as a list, preserving materialization failures.
pub fn try_to_list(t: Tensor) -> Result(List(Float), TensorError) {
  case t {
    Tensor(data, _) -> Ok(data)
    NativeTensor(ref, _) -> {
      case ffi.nt_to_list(ref) {
        Ok(data) -> Ok(data)
        Error(reason) ->
          Error(DimensionError(
            "Native tensor materialization failed: " <> reason,
          ))
      }
    }
    StridedTensor(storage, shape, strides, offset) -> {
      let total_size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
      let data =
        list.range(0, total_size - 1)
        |> list.map(fn(flat_idx) {
          let indices = flat_to_multi(flat_idx, shape)
          let idx =
            list.zip(indices, strides)
            |> list.fold(offset, fn(acc, pair) {
              let #(i, s) = pair
              acc + i * s
            })
          ffi.array_get(storage, idx)
        })
      Ok(data)
    }
  }
}

/// Extract data as list from any tensor variant.
///
/// Prefer `try_to_list` in fallible code paths so native materialization errors
/// are not hidden.
pub fn get_data(t: Tensor) -> List(Float) {
  try_to_list(t)
  |> result.unwrap([])
}

/// Total number of elements
pub fn size(t: Tensor) -> Int {
  case t {
    Tensor(data, _) -> list.length(data)
    NativeTensor(_, shape) -> list.fold(shape, 1, fn(acc, dim) { acc * dim })
    StridedTensor(_, shape, _, _) ->
      list.fold(shape, 1, fn(acc, dim) { acc * dim })
  }
}

/// Number of dimensions (rank)
pub fn rank(t: Tensor) -> Int {
  list.length(t.shape)
}

/// Inspect the canonical tensor layout metadata.
pub fn layout(t: Tensor) -> layout.TensorLayout {
  case t {
    Tensor(_, shape) ->
      layout.TensorLayout(
        storage: layout.DenseStorage,
        device: layout.BeamCpu,
        dtype: layout.Float64,
        shape: shape,
        strides: compute_strides(shape),
        offset: 0,
        size: size(t),
        rank: list.length(shape),
        contiguous: True,
      )

    StridedTensor(_, shape, strides, offset) ->
      layout.TensorLayout(
        storage: layout.StridedStorage,
        device: layout.BeamCpu,
        dtype: layout.Float64,
        shape: shape,
        strides: strides,
        offset: offset,
        size: size(t),
        rank: list.length(shape),
        contiguous: strides == compute_strides(shape),
      )

    NativeTensor(_, shape) ->
      layout.TensorLayout(
        storage: layout.NativeStorage,
        device: layout.NativeCpu,
        dtype: layout.Float64,
        shape: shape,
        strides: compute_strides(shape),
        offset: 0,
        size: size(t),
        rank: list.length(shape),
        contiguous: True,
      )
  }
}

/// Specific dimension
pub fn dim(t: Tensor, axis: Int) -> Result(Int, TensorError) {
  list_at_int(t.shape, axis)
  |> result.map_error(fn(_) {
    DimensionError("Axis " <> int.to_string(axis) <> " out of bounds")
  })
}

/// Return number of rows (for matrices)
pub fn rows(t: Tensor) -> Int {
  case t.shape {
    [r, ..] -> r
    [] -> 0
  }
}

/// Return number of columns (for matrices)
pub fn cols(t: Tensor) -> Int {
  case t.shape {
    [_, c, ..] -> c
    [n] -> n
    [] -> 0
  }
}

// --- Element Access ---

/// Access element by linear index
pub fn get(t: Tensor, index: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(data, _) ->
      list_at_float(data, index)
      |> result.map_error(fn(_) {
        DimensionError("Index " <> int.to_string(index) <> " out of bounds")
      })
    NativeTensor(ref, _) ->
      case ffi.nt_to_list(ref) {
        Ok(data) ->
          list_at_float(data, index)
          |> result.map_error(fn(_) {
            DimensionError("Index " <> int.to_string(index) <> " out of bounds")
          })
        Error(reason) -> Error(DimensionError(reason))
      }
    StridedTensor(storage, shape, strides, offset) -> {
      let indices = flat_to_multi(index, shape)
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

/// Access 2D element
pub fn get2d(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  case t.shape {
    [_rows, num_cols] -> {
      let index = row * num_cols + col
      get(t, index)
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

/// Get matrix row as vector
pub fn get_row(t: Tensor, row_idx: Int) -> Result(Tensor, TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      case row_idx >= 0 && row_idx < num_rows {
        True -> {
          let data = get_data(t)
          let start = row_idx * num_cols
          let row_data =
            data
            |> list.drop(start)
            |> list.take(num_cols)
          Ok(from_list(row_data))
        }
        False -> Error(DimensionError("Row index out of bounds"))
      }
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

/// Get matrix column as vector
pub fn get_col(t: Tensor, col_idx: Int) -> Result(Tensor, TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      case col_idx >= 0 && col_idx < num_cols {
        True -> {
          let col_data =
            list.range(0, num_rows - 1)
            |> list.filter_map(fn(row) { get2d(t, row, col_idx) })
          Ok(from_list(col_data))
        }
        False -> Error(DimensionError("Column index out of bounds"))
      }
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

// --- Element-wise Operations ---

/// Apply function to each element, preserving materialization failures.
pub fn try_map(
  t: Tensor,
  f: fn(Float) -> Float,
) -> Result(Tensor, TensorError) {
  use data <- result.try(try_to_list(t))
  Ok(Tensor(data: list.map(data, f), shape: t.shape))
}

/// Apply function to each element.
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  try_map(t, f)
  |> result.unwrap(t)
}

/// Apply function with index, preserving materialization failures.
pub fn try_map_indexed(
  t: Tensor,
  f: fn(Float, Int) -> Float,
) -> Result(Tensor, TensorError) {
  use data <- result.try(try_to_list(t))
  Ok(Tensor(data: list.index_map(data, fn(x, i) { f(x, i) }), shape: t.shape))
}

/// Apply function with index.
pub fn map_indexed(t: Tensor, f: fn(Float, Int) -> Float) -> Tensor {
  try_map_indexed(t, f)
  |> result.unwrap(t)
}

/// Apply a binary function element-wise over tensors with the same shape.
pub fn map2(
  a: Tensor,
  b: Tensor,
  f: fn(Float, Float) -> Float,
) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> elementwise_fallback(a, b, f)
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise addition
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True ->
      case a, b {
        NativeTensor(a_ref, shape), NativeTensor(b_ref, _) -> {
          case ffi.nt_add(a_ref, b_ref) {
            Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
            Error(_) -> add_dense(a, b)
          }
        }
        _, _ -> add_dense(a, b)
      }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

fn add_dense(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  map2(a, b, fn(x, y) { x +. y })
}

/// Element-wise subtraction
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True ->
      case a, b {
        NativeTensor(a_ref, shape), NativeTensor(b_ref, _) -> {
          case ffi.nt_sub(a_ref, b_ref) {
            Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
            Error(_) -> sub_dense(a, b)
          }
        }
        _, _ -> sub_dense(a, b)
      }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

fn sub_dense(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  map2(a, b, fn(x, y) { x -. y })
}

/// Element-wise multiplication (Hadamard product)
/// Not to be confused with matmul. Named after Jacques Hadamard (1865-1963).
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True ->
      case a, b {
        NativeTensor(a_ref, shape), NativeTensor(b_ref, _) -> {
          case ffi.nt_mul(a_ref, b_ref) {
            Ok(ref) -> Ok(NativeTensor(ref: ref, shape: shape))
            Error(_) -> mul_dense(a, b)
          }
        }
        _, _ -> mul_dense(a, b)
      }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

fn mul_dense(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  map2(a, b, fn(x, y) { x *. y })
}

/// Element-wise division
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> div_dense(a, b)
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

fn div_dense(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  map2(a, b, fn(x, y) { x /. y })
}

/// Write out = a + b into a preallocated native tensor.
pub fn add_into(out: Tensor, a: Tensor, b: Tensor) -> Result(Nil, TensorError) {
  native_binary_into(out, a, b, ffi.nt_add_into)
}

/// Write out = a - b into a preallocated native tensor.
pub fn sub_into(out: Tensor, a: Tensor, b: Tensor) -> Result(Nil, TensorError) {
  native_binary_into(out, a, b, ffi.nt_sub_into)
}

/// Write out = a * b into a preallocated native tensor.
pub fn mul_into(out: Tensor, a: Tensor, b: Tensor) -> Result(Nil, TensorError) {
  native_binary_into(out, a, b, ffi.nt_mul_into)
}

/// Write out = a * scalar into a preallocated native tensor.
pub fn scale_into(
  out: Tensor,
  a: Tensor,
  scalar: Float,
) -> Result(Nil, TensorError) {
  case out.shape == a.shape {
    False -> Error(ShapeMismatch(expected: a.shape, got: out.shape))
    True -> {
      case out, a {
        NativeTensor(out_ref, _), NativeTensor(a_ref, _) ->
          ffi.nt_scale_into(out_ref, a_ref, scalar)
          |> result.map_error(fn(reason) { DimensionError(reason) })
        _, _ -> Error(DimensionError("scale_into requires native tensors"))
      }
    }
  }
}

/// Write out = a @ b into a preallocated native tensor.
pub fn matmul_into(
  out: Tensor,
  a: Tensor,
  b: Tensor,
) -> Result(Nil, TensorError) {
  case a.shape, b.shape, out.shape {
    [m, n], [n2, p], [out_m, out_p] if n == n2 && m == out_m && p == out_p -> {
      case out, a, b {
        NativeTensor(out_ref, _), NativeTensor(a_ref, _), NativeTensor(b_ref, _)
        ->
          ffi.nt_matmul_inplace(a_ref, b_ref, out_ref, m, p, n)
          |> result.map_error(fn(reason) { DimensionError(reason) })
        _, _, _ -> Error(DimensionError("matmul_into requires native tensors"))
      }
    }
    [m, _], [_, p], _ -> Error(ShapeMismatch(expected: [m, p], got: out.shape))
    _, _, _ -> Error(DimensionError("Expected two matrices and matrix output"))
  }
}

/// Fused linear layer with ReLU: max(0, a @ b + bias).
pub fn linear_relu(
  a: Tensor,
  b: Tensor,
  bias: Tensor,
) -> Result(Tensor, TensorError) {
  case a.shape, b.shape, bias.shape {
    [m, k], [k2, n], [bias_n] if k == k2 && n == bias_n -> {
      case a, b, bias {
        NativeTensor(a_ref, _),
          NativeTensor(b_ref, _),
          NativeTensor(bias_ref, _)
        -> {
          case ffi.nt_fused_linear_relu(a_ref, b_ref, bias_ref, m, n, k) {
            Ok(result_ref) -> Ok(NativeTensor(ref: result_ref, shape: [m, n]))
            Error(_) -> linear_relu_dense(a, b, bias, m, n)
          }
        }
        _, _, _ -> linear_relu_dense(a, b, bias, m, n)
      }
    }
    [_, _], [_, n], [bias_n] ->
      Error(ShapeMismatch(expected: [n], got: [bias_n]))
    _, _, _ -> Error(DimensionError("Expected [m,k], [k,n], and [n] bias"))
  }
}

/// Write out = max(0, a @ b + bias) into a preallocated native tensor.
pub fn linear_relu_into(
  out: Tensor,
  a: Tensor,
  b: Tensor,
  bias: Tensor,
) -> Result(Nil, TensorError) {
  case a.shape, b.shape, bias.shape, out.shape {
    [m, k], [k2, n], [bias_n], [out_m, out_n]
      if k == k2 && n == bias_n && m == out_m && n == out_n
    -> {
      case out, a, b, bias {
        NativeTensor(out_ref, _),
          NativeTensor(a_ref, _),
          NativeTensor(b_ref, _),
          NativeTensor(bias_ref, _)
        ->
          ffi.nt_fused_linear_relu_into(
            out_ref,
            a_ref,
            b_ref,
            bias_ref,
            m,
            n,
            k,
          )
          |> result.map_error(fn(reason) { DimensionError(reason) })
        _, _, _, _ ->
          Error(DimensionError("linear_relu_into requires native tensors"))
      }
    }
    [m, _], [_, n], _, _ ->
      Error(ShapeMismatch(expected: [m, n], got: out.shape))
    _, _, _, _ ->
      Error(DimensionError("Expected [m,k], [k,n], [n], and [m,n] output"))
  }
}

fn linear_relu_dense(
  a: Tensor,
  b: Tensor,
  bias: Tensor,
  m: Int,
  n: Int,
) -> Result(Tensor, TensorError) {
  use product <- result.try(matmul(a, b))
  let data =
    list.range(0, m * n - 1)
    |> list.map(fn(i) {
      let value =
        get_element_or_zero(product, i) +. get_element_or_zero(bias, i % n)
      case value >. 0.0 {
        True -> value
        False -> 0.0
      }
    })
  Ok(Tensor(data: data, shape: [m, n]))
}

fn native_binary_into(
  out: Tensor,
  a: Tensor,
  b: Tensor,
  op: fn(NativeTensorRef, NativeTensorRef, NativeTensorRef) ->
    Result(Nil, String),
) -> Result(Nil, TensorError) {
  case a.shape == b.shape && out.shape == a.shape {
    False -> Error(ShapeMismatch(expected: a.shape, got: out.shape))
    True -> {
      case out, a, b {
        NativeTensor(out_ref, _), NativeTensor(a_ref, _), NativeTensor(b_ref, _)
        ->
          op(out_ref, a_ref, b_ref)
          |> result.map_error(fn(reason) { DimensionError(reason) })
        _, _, _ ->
          Error(DimensionError("into operations require native tensors"))
      }
    }
  }
}

fn elementwise_fallback(
  a: Tensor,
  b: Tensor,
  f: fn(Float, Float) -> Float,
) -> Result(Tensor, TensorError) {
  case is_native(a) || is_native(b) {
    True -> materialized_elementwise(a, b, f)
    False -> {
      case a, b {
        Tensor(a_data, _), Tensor(b_data, _) -> {
          let data = list.map2(a_data, b_data, f)
          Ok(Tensor(data: data, shape: a.shape))
        }
        _, _ -> indexed_elementwise(a, b, f)
      }
    }
  }
}

fn materialized_elementwise(
  a: Tensor,
  b: Tensor,
  f: fn(Float, Float) -> Float,
) -> Result(Tensor, TensorError) {
  use a_data <- result.try(try_to_list(a))
  use b_data <- result.try(try_to_list(b))
  let data = list.map2(a_data, b_data, f)
  Ok(Tensor(data: data, shape: a.shape))
}

fn indexed_elementwise(
  a: Tensor,
  b: Tensor,
  f: fn(Float, Float) -> Float,
) -> Result(Tensor, TensorError) {
  let data_result =
    list.range(0, size(a) - 1)
    |> list.fold(Ok([]), fn(acc, i) {
      use values <- result.try(acc)
      use x <- result.try(get_fast(a, i))
      use y <- result.try(get_fast(b, i))
      Ok([f(x, y), ..values])
    })

  use data <- result.try(data_result)
  Ok(Tensor(data: list.reverse(data), shape: a.shape))
}

fn get_element_or_zero(t: Tensor, index: Int) -> Float {
  case get_fast(t, index) {
    Ok(value) -> value
    Error(_) -> 0.0
  }
}

/// Scale by constant
pub fn try_scale(t: Tensor, s: Float) -> Result(Tensor, TensorError) {
  case t {
    NativeTensor(ref, shape) -> {
      case ffi.nt_scale(ref, s) {
        Ok(result_ref) -> Ok(NativeTensor(ref: result_ref, shape: shape))
        Error(_) -> try_map(t, fn(x) { x *. s })
      }
    }
    _ -> try_map(t, fn(x) { x *. s })
  }
}

/// Scale by constant.
pub fn scale(t: Tensor, s: Float) -> Tensor {
  try_scale(t, s)
  |> result.unwrap(t)
}

/// Add constant, preserving materialization failures.
pub fn try_add_scalar(t: Tensor, s: Float) -> Result(Tensor, TensorError) {
  try_map(t, fn(x) { x +. s })
}

/// Add constant.
pub fn add_scalar(t: Tensor, s: Float) -> Tensor {
  try_add_scalar(t, s)
  |> result.unwrap(t)
}

/// Negation, preserving materialization failures.
pub fn try_negate(t: Tensor) -> Result(Tensor, TensorError) {
  try_scale(t, -1.0)
}

/// Negation.
pub fn negate(t: Tensor) -> Tensor {
  try_negate(t)
  |> result.unwrap(t)
}

// --- Reduction Operations ---

/// Sum all elements
pub fn try_sum(t: Tensor) -> Result(Float, TensorError) {
  case t {
    NativeTensor(ref, _) -> {
      case ffi.nt_sum(ref) {
        Ok(value) -> Ok(value)
        Error(_) -> sum_dense(t)
      }
    }
    _ -> sum_dense(t)
  }
}

/// Sum all elements.
pub fn sum(t: Tensor) -> Float {
  try_sum(t)
  |> result.unwrap(0.0)
}

fn sum_dense(t: Tensor) -> Result(Float, TensorError) {
  use data <- result.try(try_to_list(t))
  Ok(list.fold(data, 0.0, fn(acc, x) { acc +. x }))
}

/// Product of all elements, preserving materialization failures.
pub fn try_product(t: Tensor) -> Result(Float, TensorError) {
  use data <- result.try(try_to_list(t))
  Ok(list.fold(data, 1.0, fn(acc, x) { acc *. x }))
}

/// Product of all elements.
pub fn product(t: Tensor) -> Float {
  try_product(t)
  |> result.unwrap(1.0)
}

/// Mean: E[X] = (1/n) * sum(x_i), preserving empty-tensor and materialization errors.
pub fn try_mean(t: Tensor) -> Result(Float, TensorError) {
  use total <- result.try(try_sum(t))
  let n = int.to_float(size(t))
  case n >. 0.0 {
    True -> Ok(total /. n)
    False -> Error(DimensionError("Cannot compute mean of an empty tensor"))
  }
}

/// Mean: E[X] = (1/n) * sum(x_i)
pub fn mean(t: Tensor) -> Float {
  try_mean(t)
  |> result.unwrap(0.0)
}

fn variance_dense(t: Tensor) -> Result(Float, TensorError) {
  use data <- result.try(try_to_list(t))
  use m <- result.try(try_mean(t))
  let n = int.to_float(list.length(data))
  case n >. 0.0 {
    False -> Error(DimensionError("Cannot compute variance of an empty tensor"))
    True -> {
      let squared_diffs =
        list.map(data, fn(x) {
          let diff = x -. m
          diff *. diff
        })

      Ok(list.fold(squared_diffs, 0.0, fn(acc, x) { acc +. x }) /. n)
    }
  }
}

/// Variance, preserving empty-tensor and materialization errors.
pub fn try_variance(t: Tensor) -> Result(Float, TensorError) {
  variance_dense(t)
}

/// Variance: Var(X) = E[(X - mean)^2].
pub fn variance(t: Tensor) -> Float {
  try_variance(t)
  |> result.unwrap(0.0)
}

/// Standard deviation, preserving empty-tensor and materialization errors.
pub fn try_std(t: Tensor) -> Result(Float, TensorError) {
  use value <- result.try(try_variance(t))
  Ok(ffi.sqrt(value))
}

/// Standard deviation: sqrt(Var(X)).
pub fn std(t: Tensor) -> Float {
  try_std(t)
  |> result.unwrap(0.0)
}

/// Maximum value, preserving materialization failures.
pub fn try_max(t: Tensor) -> Result(Float, TensorError) {
  case t {
    NativeTensor(ref, _) -> {
      case ffi.nt_max(ref) {
        Ok(value) -> Ok(value)
        Error(_) -> max_dense(t)
      }
    }
    _ -> max_dense(t)
  }
}

/// Maximum value.
pub fn max(t: Tensor) -> Float {
  try_max(t)
  |> result.unwrap(0.0)
}

fn max_dense(t: Tensor) -> Result(Float, TensorError) {
  use data <- result.try(try_to_list(t))
  case data {
    [] -> Error(DimensionError("Cannot compute max of an empty tensor"))
    [first, ..rest] ->
      Ok(list.fold(rest, first, fn(acc, x) { float.max(acc, x) }))
  }
}

/// Minimum value, preserving materialization failures.
pub fn try_min(t: Tensor) -> Result(Float, TensorError) {
  case t {
    NativeTensor(ref, _) -> {
      case ffi.nt_min(ref) {
        Ok(value) -> Ok(value)
        Error(_) -> min_dense(t)
      }
    }
    _ -> min_dense(t)
  }
}

/// Minimum value.
pub fn min(t: Tensor) -> Float {
  try_min(t)
  |> result.unwrap(0.0)
}

fn min_dense(t: Tensor) -> Result(Float, TensorError) {
  use data <- result.try(try_to_list(t))
  case data {
    [] -> Error(DimensionError("Cannot compute min of an empty tensor"))
    [first, ..rest] ->
      Ok(list.fold(rest, first, fn(acc, x) { float.min(acc, x) }))
  }
}

/// Argmax - index of largest element, preserving materialization failures.
pub fn try_argmax(t: Tensor) -> Result(Int, TensorError) {
  use data <- result.try(try_to_list(t))
  case data {
    [] -> Error(DimensionError("Cannot compute argmax of an empty tensor"))
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x >. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      Ok(idx)
    }
  }
}

/// Argmax - index of largest element.
pub fn argmax(t: Tensor) -> Int {
  try_argmax(t)
  |> result.unwrap(0)
}

/// Argmin - index of smallest element, preserving materialization failures.
pub fn try_argmin(t: Tensor) -> Result(Int, TensorError) {
  use data <- result.try(try_to_list(t))
  case data {
    [] -> Error(DimensionError("Cannot compute argmin of an empty tensor"))
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x <. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      Ok(idx)
    }
  }
}

/// Argmin - index of smallest element.
pub fn argmin(t: Tensor) -> Int {
  try_argmin(t)
  |> result.unwrap(0)
}

/// Sum along a specific axis, preserving materialization failures.
/// For a [2, 3] tensor, sum_axis(_, 0) gives [3], sum_axis(_, 1) gives [2].
pub fn try_sum_axis(t: Tensor, axis_idx: Int) -> Result(Tensor, TensorError) {
  sum_axis_with_keepdims(t, axis_idx, False)
}

/// Sum along a specific axis.
pub fn sum_axis(t: Tensor, axis_idx: Int) -> Result(Tensor, TensorError) {
  try_sum_axis(t, axis_idx)
}

/// Sum along a specific axis while keeping the reduced dimension as size 1.
pub fn try_sum_axis_keepdims(
  t: Tensor,
  axis_idx: Int,
) -> Result(Tensor, TensorError) {
  sum_axis_with_keepdims(t, axis_idx, True)
}

/// Sum along a specific axis while keeping the reduced dimension as size 1.
pub fn sum_axis_keepdims(
  t: Tensor,
  axis_idx: Int,
) -> Result(Tensor, TensorError) {
  try_sum_axis_keepdims(t, axis_idx)
}

fn sum_axis_with_keepdims(
  t: Tensor,
  axis_idx: Int,
  keepdims: Bool,
) -> Result(Tensor, TensorError) {
  let r = rank(t)
  case axis_idx >= 0 && axis_idx < r {
    False -> Error(DimensionError("Invalid axis index"))
    True -> {
      case t.shape {
        [] -> Error(DimensionError("Cannot reduce scalar"))
        _ -> {
          use axis_size <- result.try(axis_size(t.shape, axis_idx))
          use data <- result.try(try_to_list(t))

          let new_shape = reduced_shape(t.shape, axis_idx, keepdims)
          let new_size = list.fold(new_shape, 1, fn(acc, d) { acc * d })

          case new_size <= 0 {
            True -> Ok(Tensor(data: [], shape: new_shape))
            False ->
              reduce_sum_axis_data(
                data,
                t.shape,
                new_shape,
                axis_idx,
                axis_size,
              )
          }
        }
      }
    }
  }
}

/// Mean along a specific axis, preserving materialization failures.
pub fn try_mean_axis(t: Tensor, axis_idx: Int) -> Result(Tensor, TensorError) {
  mean_axis_with_keepdims(t, axis_idx, False)
}

/// Mean along a specific axis.
pub fn mean_axis(t: Tensor, axis_idx: Int) -> Result(Tensor, TensorError) {
  try_mean_axis(t, axis_idx)
}

/// Mean along a specific axis while keeping the reduced dimension as size 1.
pub fn try_mean_axis_keepdims(
  t: Tensor,
  axis_idx: Int,
) -> Result(Tensor, TensorError) {
  mean_axis_with_keepdims(t, axis_idx, True)
}

/// Mean along a specific axis while keeping the reduced dimension as size 1.
pub fn mean_axis_keepdims(
  t: Tensor,
  axis_idx: Int,
) -> Result(Tensor, TensorError) {
  try_mean_axis_keepdims(t, axis_idx)
}

fn mean_axis_with_keepdims(
  t: Tensor,
  axis_idx: Int,
  keepdims: Bool,
) -> Result(Tensor, TensorError) {
  let r = rank(t)
  case axis_idx >= 0 && axis_idx < r {
    False -> Error(DimensionError("Invalid axis index"))
    True -> {
      use axis_size <- result.try(axis_size(t.shape, axis_idx))
      case axis_size <= 0 {
        True -> Error(DimensionError("Cannot compute mean along empty axis"))
        False ->
          case sum_axis_with_keepdims(t, axis_idx, keepdims) {
            Error(e) -> Error(e)
            Ok(summed) -> Ok(scale(summed, 1.0 /. int.to_float(axis_size)))
          }
      }
    }
  }
}

/// Softmax along one axis, preserving the input shape and materialization failures.
pub fn try_softmax_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  let shp = t.shape
  let rnk = list.length(shp)

  case axis >= 0 && axis < rnk {
    False -> Error(DimensionError("Invalid axis for softmax"))
    True -> {
      use axis_size <- result.try(axis_size(shp, axis))
      let inner_size = layout_math.size(list.drop(shp, axis + 1))

      case axis_size <= 0 {
        True -> Ok(Tensor(data: [], shape: shp))
        False -> {
          use input <- result.try(try_to_list(t))
          use data <- result.try(softmax_axis_data(
            input,
            size(t),
            axis_size,
            inner_size,
          ))
          Ok(Tensor(data: data, shape: shp))
        }
      }
    }
  }
}

/// Softmax along one axis, preserving the input shape.
pub fn softmax_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  try_softmax_axis(t, axis)
}

fn softmax_axis_data(
  data: List(Float),
  total_size: Int,
  axis_size: Int,
  inner_size: Int,
) -> Result(List(Float), TensorError) {
  let group_width = axis_size * inner_size

  case group_width <= 0 {
    True -> Ok([])
    False -> {
      let outer_size = total_size / group_width
      list.range(0, outer_size - 1)
      |> list.fold(Ok([]), fn(acc, outer) {
        use values <- result.try(acc)
        use chunk <- result.try(softmax_axis_outer(
          data,
          outer,
          axis_size,
          inner_size,
        ))
        Ok(list.append(values, chunk))
      })
    }
  }
}

fn softmax_axis_outer(
  data: List(Float),
  outer: Int,
  axis_size: Int,
  inner_size: Int,
) -> Result(List(Float), TensorError) {
  use groups <- result.try(
    list.range(0, inner_size - 1)
    |> list.fold(Ok([]), fn(acc, inner) {
      use values <- result.try(acc)
      use normalized <- result.try(softmax_axis_values(
        data,
        outer,
        inner,
        axis_size,
        inner_size,
      ))
      Ok([normalized, ..values])
    })
    |> result.map(list.reverse),
  )

  list.range(0, axis_size - 1)
  |> list.fold(Ok([]), fn(acc, axis_pos) {
    use values <- result.try(acc)
    use axis_values <- result.try(
      groups
      |> list.fold(Ok([]), fn(group_acc, group) {
        use group_values <- result.try(group_acc)
        use value <- result.try(
          list_at_float(group, axis_pos)
          |> result.map_error(fn(_) {
            IndexOutOfBounds(axis_pos, list.length(group))
          }),
        )
        Ok([value, ..group_values])
      })
      |> result.map(list.reverse),
    )
    Ok(list.append(values, axis_values))
  })
}

fn softmax_axis_values(
  data: List(Float),
  outer: Int,
  inner: Int,
  axis_size: Int,
  inner_size: Int,
) -> Result(List(Float), TensorError) {
  case
    list.range(0, axis_size - 1)
    |> list.fold(Ok([]), fn(acc, axis_pos) {
      use values <- result.try(acc)
      let index = outer * axis_size * inner_size + inner + axis_pos * inner_size
      use value <- result.try(
        list_at_float(data, index)
        |> result.map_error(fn(_) { IndexOutOfBounds(index, list.length(data)) }),
      )
      Ok([value, ..values])
    })
    |> result.map(list.reverse)
  {
    Ok(values) -> softmax_values(values)
    Error(error) -> Error(error)
  }
}

fn softmax_values(values: List(Float)) -> Result(List(Float), TensorError) {
  case values {
    [] -> Error(DimensionError("Cannot compute softmax over an empty slice"))
    [first, ..rest] -> {
      let max_value =
        list.fold(rest, first, fn(acc, value) { float.max(acc, value) })
      let shifted = list.map(values, fn(value) { ffi.exp(value -. max_value) })
      let total = list.fold(shifted, 0.0, fn(acc, value) { acc +. value })

      Ok(list.map(shifted, fn(value) { value /. total }))
    }
  }
}

fn remove_at_index(lst: List(a), idx: Int) -> List(a) {
  lst
  |> list.index_map(fn(item, i) { #(item, i) })
  |> list.filter(fn(pair) { pair.1 != idx })
  |> list.map(fn(pair) { pair.0 })
}

fn axis_size(shape: List(Int), axis_idx: Int) -> Result(Int, TensorError) {
  list_at_int(shape, axis_idx)
  |> result.map_error(fn(_) { DimensionError("Invalid axis index") })
}

fn reduced_shape(shape: List(Int), axis_idx: Int, keepdims: Bool) -> List(Int) {
  case keepdims {
    True ->
      shape
      |> list.index_map(fn(dim, idx) {
        case idx == axis_idx {
          True -> 1
          False -> dim
        }
      })
    False -> remove_at_index(shape, axis_idx)
  }
}

fn reduce_sum_axis_data(
  data: List(Float),
  input_shape: List(Int),
  output_shape: List(Int),
  axis_idx: Int,
  axis_size: Int,
) -> Result(Tensor, TensorError) {
  case axis_size <= 0 {
    True -> {
      let output_size = list.fold(output_shape, 1, fn(acc, dim) { acc * dim })
      Ok(Tensor(data: list.repeat(0.0, output_size), shape: output_shape))
    }
    False -> {
      let output_size = list.fold(output_shape, 1, fn(acc, dim) { acc * dim })
      list.range(0, output_size - 1)
      |> list.fold(Ok([]), fn(acc, out_idx) {
        use values <- result.try(acc)
        use value <- result.try(sum_axis_output(
          data,
          input_shape,
          output_shape,
          out_idx,
          axis_idx,
          axis_size,
        ))
        Ok([value, ..values])
      })
      |> result.map(fn(values) {
        Tensor(data: list.reverse(values), shape: output_shape)
      })
    }
  }
}

fn sum_axis_output(
  data: List(Float),
  input_shape: List(Int),
  output_shape: List(Int),
  out_idx: Int,
  axis_idx: Int,
  axis_size: Int,
) -> Result(Float, TensorError) {
  let output_coords = flat_to_multi(out_idx, output_shape)
  list.range(0, axis_size - 1)
  |> list.fold(Ok(0.0), fn(acc, axis_pos) {
    use total <- result.try(acc)
    let input_coords =
      insert_reduced_axis(
        output_coords,
        axis_idx,
        axis_pos,
        output_shape,
        input_shape,
      )

    let input_idx = multi_to_flat(input_coords, input_shape)
    use value <- result.try(
      list_at_float(data, input_idx)
      |> result.map_error(fn(_) {
        IndexOutOfBounds(input_idx, list.length(data))
      }),
    )
    Ok(total +. value)
  })
}

fn insert_reduced_axis(
  output_coords: List(Int),
  axis_idx: Int,
  axis_pos: Int,
  output_shape: List(Int),
  input_shape: List(Int),
) -> List(Int) {
  case list.length(output_shape) == list.length(input_shape) {
    True ->
      output_coords
      |> list.index_map(fn(coord, idx) {
        case idx == axis_idx {
          True -> axis_pos
          False -> coord
        }
      })
    False -> {
      let #(before, after) = list.split(output_coords, axis_idx)
      list.flatten([before, [axis_pos], after])
    }
  }
}

// --- Matrix Operations ---

/// Dot product of two vectors: a . b = sum(a_i * b_i)
pub fn dot(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case rank(a) == 1 && rank(b) == 1 && size(a) == size(b) {
    True ->
      case a, b {
        NativeTensor(a_ref, _), NativeTensor(b_ref, _) -> {
          case ffi.nt_dot(a_ref, b_ref) {
            Ok(value) -> Ok(value)
            Error(_) -> dot_dense(a, b)
          }
        }
        _, _ -> dot_dense(a, b)
      }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

fn dot_dense(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  use a_data <- result.try(try_to_list(a))
  use b_data <- result.try(try_to_list(b))
  let products = list.map2(a_data, b_data, fn(x, y) { x *. y })
  Ok(list.fold(products, 0.0, fn(acc, x) { acc +. x }))
}

/// Matrix-vector multiplication: [m, n] @ [n] -> [m]
/// C_i = sum_j(A_ij * x_j)
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  case mat.shape, vec.shape {
    [m, n], [vec_n] if n == vec_n -> {
      use mat_data <- result.try(try_to_list(mat))
      use vec_data <- result.try(try_to_list(vec))
      let result_data =
        list.range(0, m - 1)
        |> list.map(fn(row_idx) {
          let start = row_idx * n
          let row =
            mat_data
            |> list.drop(start)
            |> list.take(n)
          list.map2(row, vec_data, fn(a, b) { a *. b })
          |> list.fold(0.0, fn(acc, x) { acc +. x })
        })
      Ok(Tensor(data: result_data, shape: [m]))
    }
    [_m, n], [vec_n] -> Error(ShapeMismatch(expected: [n], got: [vec_n]))
    _, _ -> Error(DimensionError("Expected matrix and vector"))
  }
}

/// Matrix-matrix multiplication: [m, n] @ [n, p] -> [m, p]
/// C_ij = sum_k(A_ik * B_kj)
///
/// This is O(mnp) - for large matrices, use the NIF backend.
pub fn matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape, b.shape {
    [m, n], [n2, p] if n == n2 -> {
      case a, b {
        NativeTensor(a_ref, _), NativeTensor(b_ref, _) -> {
          case ffi.nt_matmul(a_ref, b_ref, m, p, n) {
            Ok(ref) -> Ok(NativeTensor(ref: ref, shape: [m, p]))
            Error(_) -> matmul_dense(a, b, m, n, p)
          }
        }
        _, _ -> matmul_dense(a, b, m, n, p)
      }
    }
    [_m, n], [n2, _p] -> Error(ShapeMismatch(expected: [n, -1], got: [n2, -1]))
    _, _ -> Error(DimensionError("Expected two matrices"))
  }
}

fn matmul_dense(
  a: Tensor,
  b: Tensor,
  m: Int,
  n: Int,
  p: Int,
) -> Result(Tensor, TensorError) {
  use a_data <- result.try(try_to_list(a))
  use b_data <- result.try(try_to_list(b))
  let a_array = ffi.list_to_array(a_data)
  let b_array = ffi.list_to_array(b_data)

  let result_data =
    list.range(0, m - 1)
    |> list.flat_map(fn(i) {
      list.range(0, p - 1)
      |> list.map(fn(j) {
        list.range(0, n - 1)
        |> list.fold(0.0, fn(acc, k) {
          let a_ik = ffi.array_get(a_array, i * n + k)
          let b_kj = ffi.array_get(b_array, k * p + j)
          acc +. a_ik *. b_kj
        })
      })
    })
  Ok(Tensor(data: result_data, shape: [m, p]))
}

/// Matrix transpose: A^T where (A^T)_ij = A_ji
pub fn transpose(t: Tensor) -> Result(Tensor, TensorError) {
  case t.shape {
    [m, n] -> {
      case t {
        NativeTensor(ref, _) -> {
          case ffi.nt_transpose(ref) {
            Ok(result_ref) -> Ok(NativeTensor(ref: result_ref, shape: [n, m]))
            Error(_) -> transpose_dense(t, m, n)
          }
        }
        _ -> transpose_dense(t, m, n)
      }
    }
    _ -> Error(DimensionError("Transpose requires 2D tensor"))
  }
}

fn transpose_dense(t: Tensor, m: Int, n: Int) -> Result(Tensor, TensorError) {
  let result_data =
    list.range(0, n - 1)
    |> list.flat_map(fn(j) {
      list.range(0, m - 1)
      |> list.filter_map(fn(i) { get2d(t, i, j) })
    })
  Ok(Tensor(data: result_data, shape: [n, m]))
}

/// Outer product: [m] x [n] -> [m, n]
/// C_ij = a_i * b_j
pub fn outer(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case rank(a) == 1 && rank(b) == 1 {
    True -> {
      let m = size(a)
      let n = size(b)
      use a_data <- result.try(try_to_list(a))
      use b_data <- result.try(try_to_list(b))
      let result_data =
        list.flat_map(a_data, fn(ai) { list.map(b_data, fn(bj) { ai *. bj }) })
      Ok(Tensor(data: result_data, shape: [m, n]))
    }
    False -> Error(DimensionError("Outer product requires two vectors"))
  }
}

// --- Utility ---

/// Convert to list
pub fn to_list(t: Tensor) -> List(Float) {
  get_data(t)
}

/// Convert matrix to list of lists
pub fn to_list2d(t: Tensor) -> Result(List(List(Float)), TensorError) {
  case t.shape {
    [num_rows, num_cols] -> {
      use data <- result.try(try_to_list(t))
      let rows_list =
        list.range(0, num_rows - 1)
        |> list.map(fn(i) {
          let start = i * num_cols
          data
          |> list.drop(start)
          |> list.take(num_cols)
        })
      Ok(rows_list)
    }
    _ -> Error(DimensionError("Tensor is not 2D"))
  }
}

/// Clone tensor, preserving native materialization failures.
pub fn try_clone(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    NativeTensor(ref, shape) -> {
      case ffi.nt_to_list(ref) {
        Ok(data) -> {
          case ffi.nt_from_list(data, shape) {
            Ok(cloned_ref) -> Ok(NativeTensor(ref: cloned_ref, shape: shape))
            Error(_) -> Ok(Tensor(data: data, shape: shape))
          }
        }
        Error(_) -> Error(DimensionError("Could not materialize native tensor"))
      }
    }
    _ -> {
      use data <- result.try(try_to_list(t))
      Ok(Tensor(data: data, shape: t.shape))
    }
  }
}

/// Clone tensor (creates a copy).
pub fn clone(t: Tensor) -> Tensor {
  try_clone(t)
  |> result.unwrap(Tensor(data: [], shape: t.shape))
}

/// Reshape tensor - same data, different shape
/// The total number of elements must match.
pub fn reshape(t: Tensor, new_shape: List(Int)) -> Result(Tensor, TensorError) {
  let old_size = size(t)
  let new_size = list.fold(new_shape, 1, fn(acc, dim) { acc * dim })

  case old_size == new_size {
    True ->
      case t {
        NativeTensor(ref, _) -> Ok(NativeTensor(ref: ref, shape: new_shape))
        Tensor(data, _) -> Ok(Tensor(data: data, shape: new_shape))
        StridedTensor(storage, shape, strides, offset) ->
          case strides == compute_strides(shape) {
            True ->
              Ok(StridedTensor(
                storage: storage,
                shape: new_shape,
                strides: compute_strides(new_shape),
                offset: offset,
              ))
            False -> {
              use data <- result.try(try_to_list(t))
              Ok(Tensor(data: data, shape: new_shape))
            }
          }
      }
    False ->
      Error(InvalidShape(
        "Cannot reshape: size mismatch ("
        <> int.to_string(old_size)
        <> " vs "
        <> int.to_string(new_size)
        <> ")",
      ))
  }
}

/// Flatten to 1D, preserving materialization failures.
pub fn try_flatten(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    NativeTensor(ref, _) -> Ok(NativeTensor(ref: ref, shape: [size(t)]))
    _ -> {
      use data <- result.try(try_to_list(t))
      Ok(Tensor(data: data, shape: [size(t)]))
    }
  }
}

/// Flatten to 1D.
pub fn flatten(t: Tensor) -> Tensor {
  try_flatten(t)
  |> result.unwrap(Tensor(data: [], shape: [0]))
}

/// Concatenate vectors (1D), preserving materialization failures.
pub fn try_concat(tensors: List(Tensor)) -> Result(Tensor, TensorError) {
  use data <- result.try(materialize_many(tensors))
  Ok(from_list(list.flatten(data)))
}

/// Concatenate vectors (1D).
pub fn concat(tensors: List(Tensor)) -> Tensor {
  try_concat(tensors)
  |> result.unwrap(from_list([]))
}

/// Concatenate tensors along a specific axis
/// For [2,3] and [2,3] tensors: concat_axis([a, b], 0) -> [4,3]
/// For [2,3] and [2,3] tensors: concat_axis([a, b], 1) -> [2,6]
pub fn concat_axis(
  tensors: List(Tensor),
  axis: Int,
) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(InvalidShape("Cannot concatenate empty list"))
    [single] -> Ok(single)
    [first, ..rest] -> {
      let base_shape = first.shape
      let r = list.length(base_shape)

      case axis >= 0 && axis < r {
        False -> Error(DimensionError("Invalid axis for concatenation"))
        True -> {
          // Verify all tensors have same shape except on concat axis
          let shapes_ok =
            list.all(rest, fn(t) {
              let t_shape = t.shape
              case list.length(t_shape) == r {
                False -> False
                True -> {
                  list.zip(base_shape, t_shape)
                  |> list.index_map(fn(pair, i) { #(pair, i) })
                  |> list.all(fn(x) {
                    let #(#(dim_a, dim_b), i) = x
                    i == axis || dim_a == dim_b
                  })
                }
              }
            })

          case shapes_ok {
            False ->
              Error(InvalidShape("Shapes must match except on concat axis"))
            True -> {
              // Build new shape
              let concat_dim =
                list.fold(tensors, 0, fn(acc, t) {
                  case axis_size(t.shape, axis) {
                    Ok(d) -> acc + d
                    Error(_) -> acc
                  }
                })

              let new_shape =
                base_shape
                |> list.index_map(fn(d, i) {
                  case i == axis {
                    True -> concat_dim
                    False -> d
                  }
                })

              // Concatenate data
              // For axis=0, we just append all data
              // For other axes, we need to interleave
              case axis == 0 {
                True -> {
                  use chunks <- result.try(materialize_many(tensors))
                  let data = list.flatten(chunks)
                  Ok(Tensor(data: data, shape: new_shape))
                }
                False -> {
                  use materialized <- result.try(materialize_many(tensors))
                  // General case: interleave based on axis
                  let total_size =
                    list.fold(new_shape, 1, fn(acc, d) { acc * d })
                  let _new_strides = compute_strides(new_shape)

                  let result =
                    list.range(0, total_size - 1)
                    |> list.fold(Ok([]), fn(acc, flat_idx) {
                      use values <- result.try(acc)
                      let indices = flat_to_multi(flat_idx, new_shape)
                      use axis_idx <- result.try(
                        list_at_int(indices, axis)
                        |> result.map_error(fn(_) {
                          DimensionError("Invalid axis for concatenation")
                        }),
                      )

                      // Find which tensor and local index
                      let #(tensor_idx, local_axis_idx, _) =
                        list.fold(tensors, #(-1, axis_idx, 0), fn(acc, t) {
                          let #(found_t, remaining, t_idx) = acc
                          case found_t >= 0 {
                            True -> acc
                            False -> {
                              let t_axis_size = case axis_size(t.shape, axis) {
                                Ok(d) -> d
                                Error(_) -> 0
                              }
                              case remaining < t_axis_size {
                                True -> #(t_idx, remaining, t_idx)
                                False -> #(
                                  -1,
                                  remaining - t_axis_size,
                                  t_idx + 1,
                                )
                              }
                            }
                          }
                        })

                      // Build local indices
                      let local_indices =
                        indices
                        |> list.index_map(fn(idx, i) {
                          case i == axis {
                            True -> local_axis_idx
                            False -> idx
                          }
                        })

                      // Get value from correct tensor
                      case
                        list_at_tensor(tensors, tensor_idx),
                        list_at_float_list(materialized, tensor_idx)
                      {
                        Ok(t), Ok(t_data) -> {
                          let t_strides = compute_strides(t.shape)
                          let local_flat =
                            list.zip(local_indices, t_strides)
                            |> list.fold(0, fn(a, p) { a + p.0 * p.1 })
                          use value <- result.try(
                            list_at_float(t_data, local_flat)
                            |> result.map_error(fn(_) {
                              IndexOutOfBounds(local_flat, list.length(t_data))
                            }),
                          )
                          Ok([value, ..values])
                        }
                        _, _ ->
                          Error(DimensionError(
                            "Invalid tensor index for concatenation",
                          ))
                      }
                    })
                    |> result.map(list.reverse)

                  case result {
                    Ok(data) -> Ok(Tensor(data: data, shape: new_shape))
                    Error(error) -> Error(error)
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/// Stack tensors along a new axis
/// For [3] and [3] tensors: stack([a, b], 0) -> [2, 3]
/// For [3] and [3] tensors: stack([a, b], 1) -> [3, 2]
pub fn stack(tensors: List(Tensor), axis: Int) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(InvalidShape("Cannot stack empty list"))
    [first, ..rest] -> {
      let base_shape = first.shape
      let shapes_ok = list.all(rest, fn(t) { t.shape == base_shape })

      case shapes_ok {
        False -> Error(ShapeMismatch(expected: base_shape, got: []))
        True -> {
          let n_tensors = list.length(tensors)
          let r = list.length(base_shape)
          let insert_axis = case axis < 0 {
            True -> r + axis + 1
            False -> axis
          }

          case insert_axis >= 0 && insert_axis <= r {
            False -> Error(DimensionError("Invalid axis for stacking"))
            True -> {
              // New shape: insert n_tensors at axis position
              let #(before, after) = list.split(base_shape, insert_axis)
              let _new_shape = list.flatten([before, [n_tensors], after])

              // Unsqueeze each tensor and concat
              let unsqueezed =
                tensors
                |> list.map(fn(t) { unsqueeze(t, insert_axis) })

              concat_axis(unsqueezed, insert_axis)
            }
          }
        }
      }
    }
  }
}

/// Take first N elements along first axis, preserving materialization failures.
pub fn try_take_first(t: Tensor, n: Int) -> Result(Tensor, TensorError) {
  use data <- result.try(try_to_list(t))
  case t.shape {
    [] -> Ok(t)
    [first_dim, ..rest_dims] -> {
      let take_n = int.min(n, first_dim)
      let stride = list.fold(rest_dims, 1, fn(acc, d) { acc * d })
      let new_data = list.take(data, take_n * stride)
      let new_shape = [take_n, ..rest_dims]
      Ok(Tensor(data: new_data, shape: new_shape))
    }
  }
}

/// Take first N elements along first axis.
pub fn take_first(t: Tensor, n: Int) -> Tensor {
  try_take_first(t, n)
  |> result.unwrap(Tensor(data: [], shape: [0]))
}

/// Take last N elements along first axis, preserving materialization failures.
pub fn try_take_last(t: Tensor, n: Int) -> Result(Tensor, TensorError) {
  use data <- result.try(try_to_list(t))
  case t.shape {
    [] -> Ok(t)
    [first_dim, ..rest_dims] -> {
      let take_n = int.min(n, first_dim)
      let stride = list.fold(rest_dims, 1, fn(acc, d) { acc * d })
      let skip = { first_dim - take_n } * stride
      let new_data = list.drop(data, skip)
      let new_shape = [take_n, ..rest_dims]
      Ok(Tensor(data: new_data, shape: new_shape))
    }
  }
}

/// Take last N elements along first axis.
pub fn take_last(t: Tensor, n: Int) -> Tensor {
  try_take_last(t, n)
  |> result.unwrap(Tensor(data: [], shape: [0]))
}

/// Slice tensor: extract sub-tensor from start to start+lengths
/// slice(t, [1], [3]) extracts elements at indices 1, 2, 3
pub fn slice(
  t: Tensor,
  start: List(Int),
  lengths: List(Int),
) -> Result(Tensor, TensorError) {
  use data <- result.try(try_to_list(t))
  let r = rank(t)

  case list.length(start) == r && list.length(lengths) == r {
    False -> Error(DimensionError("Slice dimensions must match tensor rank"))
    True -> {
      case r {
        1 -> {
          use s <- result.try(
            list_at_int(start, 0)
            |> result.map_error(fn(_) { DimensionError("Invalid slice start") }),
          )
          use len <- result.try(
            list_at_int(lengths, 0)
            |> result.map_error(fn(_) { DimensionError("Invalid slice length") }),
          )
          use dim <- result.try(axis_size(t.shape, 0))

          case s < 0 || len < 0 || s + len > dim {
            True -> Error(IndexOutOfBounds(s + len, dim))
            False -> {
              let sliced = data |> list.drop(s) |> list.take(len)
              Ok(Tensor(data: sliced, shape: [len]))
            }
          }
        }
        _ -> {
          case slice_bounds_valid(t.shape, start, lengths) {
            False -> Error(DimensionError("Slice bounds exceed tensor shape"))
            True -> {
              // Multi-dimensional slice - general case
              let new_size = list.fold(lengths, 1, fn(acc, d) { acc * d })

              let result =
                list.range(0, new_size - 1)
                |> list.fold(Ok([]), fn(acc, flat_idx) {
                  use values <- result.try(acc)
                  let local_indices = flat_to_multi(flat_idx, lengths)
                  let global_indices =
                    list.map2(local_indices, start, fn(l, s) { l + s })
                  let global_flat = multi_to_flat(global_indices, t.shape)
                  use value <- result.try(
                    list_at_float(data, global_flat)
                    |> result.map_error(fn(_) {
                      IndexOutOfBounds(global_flat, list.length(data))
                    }),
                  )
                  Ok([value, ..values])
                })
                |> result.map(list.reverse)

              case result {
                Ok(values) -> Ok(Tensor(data: values, shape: lengths))
                Error(error) -> Error(error)
              }
            }
          }
        }
      }
    }
  }
}

/// L2 norm: ||x||_2 = sqrt(sum(x_i^2)), preserving materialization failures.
pub fn try_norm(t: Tensor) -> Result(Float, TensorError) {
  use data <- result.try(try_to_list(t))
  let sum_sq = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  Ok(ffi.sqrt(sum_sq))
}

/// L2 norm: ||x||_2 = sqrt(sum(x_i^2)).
pub fn norm(t: Tensor) -> Float {
  try_norm(t)
  |> result.unwrap(0.0)
}

/// Normalize to unit length: x / ||x||_2, preserving materialization failures.
pub fn try_normalize(t: Tensor) -> Result(Tensor, TensorError) {
  use n <- result.try(try_norm(t))
  case n >. 0.0001 {
    True -> try_scale(t, 1.0 /. n)
    False -> Ok(t)
  }
}

/// Normalize to unit length: x / ||x||_2.
pub fn normalize(t: Tensor) -> Tensor {
  try_normalize(t)
  |> result.unwrap(t)
}

/// Compare two scalars with relative and absolute tolerances.
pub fn is_close(a: Float, b: Float, rtol: Float, atol: Float) -> Bool {
  maths.is_close(a, b, rtol, atol)
}

/// Compare two tensors element-wise and return whether all pairs are close.
pub fn all_close(
  a: Tensor,
  b: Tensor,
  rtol: Float,
  atol: Float,
) -> Result(Bool, TensorError) {
  case a.shape == b.shape {
    False -> Error(ShapeMismatch(a.shape, b.shape))
    True -> {
      use a_data <- result.try(try_to_list(a))
      use b_data <- result.try(try_to_list(b))

      maths.all_close(list.zip(a_data, b_data), rtol, atol)
      |> list.all(fn(close) { close })
      |> Ok
    }
  }
}

/// Clamp values to [min, max], preserving materialization failures.
pub fn try_clamp(
  t: Tensor,
  min_val: Float,
  max_val: Float,
) -> Result(Tensor, TensorError) {
  try_map(t, fn(x) { float.min(float.max(x, min_val), max_val) })
}

/// Clamp values to [min, max].
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  try_clamp(t, min_val, max_val)
  |> result.unwrap(t)
}

// --- Random ---

/// Tensor with uniform random values [0, 1)
pub fn random_uniform(shape: List(Int)) -> Tensor {
  let size_val = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size_val)
    |> list.map(fn(_) { ffi.random_uniform() })
  Tensor(data: data, shape: shape)
}

/// Tensor with normal random values via Box-Muller transform.
/// Box & Muller (1958) - A Note on the Generation of Random Normal Deviates
pub fn random_normal(
  shape: List(Int),
  mean_val: Float,
  std_val: Float,
) -> Tensor {
  let size_val = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size_val)
    |> list.map(fn(_) {
      let u1 = float.max(ffi.random_uniform(), 0.0001)
      let u2 = ffi.random_uniform()
      // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
      let z =
        ffi.sqrt(-2.0 *. ffi.log(u1)) *. ffi.cos(2.0 *. 3.14159265359 *. u2)
      mean_val +. z *. std_val
    })
  Tensor(data: data, shape: shape)
}

/// Xavier/Glorot initialization for weights.
/// Glorot & Bengio (2010) - Understanding the difficulty of training deep feedforward NNs
///
/// limit = sqrt(6 / (fan_in + fan_out))
/// W ~ Uniform(-limit, limit)
pub fn xavier_init(fan_in: Int, fan_out: Int) -> Tensor {
  let limit = ffi.sqrt(6.0 /. int.to_float(fan_in + fan_out))
  let data =
    list.range(1, fan_in * fan_out)
    |> list.map(fn(_) {
      let r = ffi.random_uniform()
      r *. 2.0 *. limit -. limit
    })
  // Shape [fan_out, fan_in] follows PyTorch convention: W @ x where x is [fan_in]
  Tensor(data: data, shape: [fan_out, fan_in])
}

/// He/Kaiming initialization (for ReLU networks).
/// He et al. (2015) - Delving Deep into Rectifiers
///
/// std = sqrt(2 / fan_in)
/// W ~ Normal(0, std)
pub fn he_init(fan_in: Int, fan_out: Int) -> Tensor {
  let std_val = ffi.sqrt(2.0 /. int.to_float(fan_in))
  // Shape [fan_out, fan_in] follows PyTorch convention
  random_normal([fan_out, fan_in], 0.0, std_val)
}

// --- Broadcasting ---
// NumPy broadcasting rules (van der Walt et al., 2011)
// 1. If ranks differ, prepend 1s to the smaller shape
// 2. Dimensions are compatible if equal or one of them is 1
// 3. Output dimension is the maximum of the two

/// Check if two shapes can be broadcast together
pub fn can_broadcast(a: List(Int), b: List(Int)) -> Bool {
  let #(longer, shorter) = case list.length(a) >= list.length(b) {
    True -> #(a, b)
    False -> #(b, a)
  }

  let diff = list.length(longer) - list.length(shorter)
  let padded = list.append(list.repeat(1, diff), shorter)

  list.zip(longer, padded)
  |> list.all(fn(pair) {
    let #(dim_a, dim_b) = pair
    dim_a == dim_b || dim_a == 1 || dim_b == 1
  })
}

/// Compute broadcast shape
pub fn broadcast_shape(
  a: List(Int),
  b: List(Int),
) -> Result(List(Int), TensorError) {
  case can_broadcast(a, b) {
    False -> Error(BroadcastError(shape_a: a, shape_b: b))
    True -> {
      let max_rank = int.max(list.length(a), list.length(b))
      let diff_a = max_rank - list.length(a)
      let diff_b = max_rank - list.length(b)
      let padded_a = list.append(list.repeat(1, diff_a), a)
      let padded_b = list.append(list.repeat(1, diff_b), b)

      let result_shape =
        list.zip(padded_a, padded_b)
        |> list.map(fn(pair) {
          let #(dim_a, dim_b) = pair
          int.max(dim_a, dim_b)
        })

      Ok(result_shape)
    }
  }
}

/// Compute the common shape for any number of broadcastable shapes.
pub fn broadcast_shapes(
  shapes: List(List(Int)),
) -> Result(List(Int), TensorError) {
  case shapes {
    [] -> Ok([])
    [first, ..rest] -> broadcast_shapes_loop(rest, first)
  }
}

fn broadcast_shapes_loop(
  shapes: List(List(Int)),
  current: List(Int),
) -> Result(List(Int), TensorError) {
  case shapes {
    [] -> Ok(current)
    [next, ..rest] -> {
      use result_shape <- result.try(broadcast_shape(current, next))
      broadcast_shapes_loop(rest, result_shape)
    }
  }
}

/// Broadcast tensor to target shape.
/// Dense and strided tensors return a zero-stride view; native tensors
/// materialize until the NIF layer exposes strided/broadcast views.
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  case can_broadcast(t.shape, target_shape) {
    False -> Error(BroadcastError(shape_a: t.shape, shape_b: target_shape))
    True -> {
      case t.shape == target_shape {
        True -> Ok(t)
        False -> {
          case t {
            Tensor(data, shape) -> {
              let storage = ffi.list_to_array(data)
              let strides =
                broadcast_strides(shape, compute_strides(shape), target_shape)
              Ok(StridedTensor(
                storage: storage,
                shape: target_shape,
                strides: strides,
                offset: 0,
              ))
            }
            StridedTensor(storage, shape, strides, offset) -> {
              let view_strides = broadcast_strides(shape, strides, target_shape)
              Ok(StridedTensor(
                storage: storage,
                shape: target_shape,
                strides: view_strides,
                offset: offset,
              ))
            }
            NativeTensor(ref, _) ->
              case ffi.nt_broadcast_to(ref, target_shape) {
                Ok(view_ref) ->
                  Ok(NativeTensor(ref: view_ref, shape: target_shape))
                Error(_) -> {
                  use data <- result.try(broadcast_data(t, target_shape))
                  Ok(Tensor(data: data, shape: target_shape))
                }
              }
          }
        }
      }
    }
  }
}

/// Broadcast two tensors to their common shape.
pub fn broadcast_pair(
  a: Tensor,
  b: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  use result_shape <- result.try(broadcast_shape(a.shape, b.shape))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  Ok(#(a_bc, b_bc))
}

/// Element-wise addition with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use pair <- result.try(broadcast_pair(a, b))
  let #(a_bc, b_bc) = pair
  add(a_bc, b_bc)
}

/// Element-wise subtraction with broadcasting
pub fn sub_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use pair <- result.try(broadcast_pair(a, b))
  let #(a_bc, b_bc) = pair
  sub(a_bc, b_bc)
}

/// Element-wise multiplication with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use pair <- result.try(broadcast_pair(a, b))
  let #(a_bc, b_bc) = pair
  mul(a_bc, b_bc)
}

/// Element-wise division with broadcasting
pub fn div_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use pair <- result.try(broadcast_pair(a, b))
  let #(a_bc, b_bc) = pair
  div(a_bc, b_bc)
}

// --- Shape Manipulation ---

/// Remove dimensions of size 1 (squeeze operation)
pub fn squeeze(t: Tensor) -> Tensor {
  case t {
    Tensor(data, shape) -> Tensor(data: data, shape: squeezed_shape(shape))
    StridedTensor(storage, shape, strides, offset) -> {
      let #(final_shape, final_strides) = squeeze_shape_strides(shape, strides)
      StridedTensor(
        storage: storage,
        shape: final_shape,
        strides: final_strides,
        offset: offset,
      )
    }
    NativeTensor(ref, shape) ->
      NativeTensor(ref: ref, shape: squeezed_shape(shape))
  }
}

fn squeezed_shape(shape: List(Int)) -> List(Int) {
  case list.filter(shape, fn(d) { d != 1 }) {
    [] -> [1]
    squeezed -> squeezed
  }
}

fn squeeze_shape_strides(
  shape: List(Int),
  strides: List(Int),
) -> #(List(Int), List(Int)) {
  let kept =
    list.zip(shape, strides)
    |> list.filter(fn(pair) {
      let #(dim, _) = pair
      dim != 1
    })

  case kept {
    [] -> #([1], [1])
    _ -> {
      let final_shape = list.map(kept, fn(pair) { pair.0 })
      let final_strides = list.map(kept, fn(pair) { pair.1 })
      #(final_shape, final_strides)
    }
  }
}

fn insert_stride(strides: List(Int), axis: Int, stride: Int) -> List(Int) {
  let #(before, after) = list.split(strides, axis)
  list.flatten([before, [stride], after])
}

/// Remove dimension at specific axis if it's 1
pub fn squeeze_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  case list_at_int(t.shape, axis) {
    Error(_) -> Error(DimensionError("Axis out of bounds"))
    Ok(d) -> {
      case d == 1 {
        False -> Error(InvalidShape("Dimension at axis is not 1"))
        True -> {
          let new_shape = remove_at_index(t.shape, axis)
          case t {
            Tensor(data, _) -> Ok(Tensor(data: data, shape: new_shape))
            NativeTensor(ref, _) -> Ok(NativeTensor(ref: ref, shape: new_shape))
            StridedTensor(storage, _, strides, offset) ->
              Ok(StridedTensor(
                storage: storage,
                shape: new_shape,
                strides: remove_at_index(strides, axis),
                offset: offset,
              ))
          }
        }
      }
    }
  }
}

/// Add dimension of size 1 at specified axis (unsqueeze operation).
pub fn try_unsqueeze(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  let rnk = list.length(t.shape)
  let insert_at = case axis < 0 {
    True -> rnk + axis + 1
    False -> axis
  }

  case insert_at < 0 || insert_at > rnk {
    True -> Error(DimensionError("Axis out of bounds"))
    False -> {
      let #(before, after) = list.split(t.shape, insert_at)
      let new_shape = list.flatten([before, [1], after])
      case t {
        Tensor(data, _) -> Ok(Tensor(data: data, shape: new_shape))
        NativeTensor(ref, _) -> Ok(NativeTensor(ref: ref, shape: new_shape))
        StridedTensor(storage, shape, strides, offset) -> {
          let new_strides = case strides == compute_strides(shape) {
            True -> compute_strides(new_shape)
            False -> insert_stride(strides, insert_at, 0)
          }
          Ok(StridedTensor(
            storage: storage,
            shape: new_shape,
            strides: new_strides,
            offset: offset,
          ))
        }
      }
    }
  }
}

/// Add dimension of size 1 at specified axis (unsqueeze operation).
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  try_unsqueeze(t, axis)
  |> result.unwrap(t)
}

/// Expand tensor to add batch dimension (alias for unsqueeze)
pub fn expand_dims(t: Tensor, axis: Int) -> Tensor {
  unsqueeze(t, axis)
}

/// Expand tensor to add batch dimension (fallible alias for try_unsqueeze).
pub fn try_expand_dims(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  try_unsqueeze(t, axis)
}

// --- Strided Tensor (Zero-copy operations) ---

/// Convert regular tensor to strided (O(n) once, then O(1) access)
pub fn try_to_strided(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    StridedTensor(_, _, _, _) -> Ok(t)
    Tensor(data, shape) -> {
      let storage = ffi.list_to_array(data)
      let strides = compute_strides(shape)
      Ok(StridedTensor(
        storage: storage,
        shape: shape,
        strides: strides,
        offset: 0,
      ))
    }
    NativeTensor(_, _) -> {
      use data <- result.try(try_to_list(t))
      let storage = ffi.list_to_array(data)
      let strides = compute_strides(t.shape)
      Ok(StridedTensor(
        storage: storage,
        shape: t.shape,
        strides: strides,
        offset: 0,
      ))
    }
  }
}

/// Convert regular tensor to strided (O(n) once, then O(1) access)
pub fn to_strided(t: Tensor) -> Tensor {
  try_to_strided(t)
  |> result.unwrap(t)
}

/// Convert strided tensor back to regular (materializes the view)
pub fn try_to_contiguous(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    Tensor(_, _) -> Ok(t)
    NativeTensor(_, _) -> Ok(t)
    StridedTensor(_, _, _, _) -> {
      use data <- result.try(try_to_list(t))
      Ok(Tensor(data: data, shape: t.shape))
    }
  }
}

/// Convert strided tensor back to regular (materializes the view)
pub fn to_contiguous(t: Tensor) -> Tensor {
  try_to_contiguous(t)
  |> result.unwrap(t)
}

/// ZERO-COPY TRANSPOSE - just swap strides and shape!
/// This is the magic of strided tensors: transpose is O(1).
pub fn transpose_strided(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    Tensor(_, shape) -> {
      case shape {
        [_m, _n] -> {
          let strided = to_strided(t)
          transpose_strided(strided)
        }
        _ -> Error(DimensionError("Transpose requires 2D tensor"))
      }
    }
    NativeTensor(_, _) -> transpose(t)
    StridedTensor(storage, shape, strides, offset) -> {
      case shape, strides {
        [m, n], [s0, s1] -> {
          Ok(StridedTensor(
            storage: storage,
            shape: [n, m],
            strides: [s1, s0],
            offset: offset,
          ))
        }
        _, _ -> Error(DimensionError("Transpose requires 2D tensor"))
      }
    }
  }
}

/// Check if tensor is contiguous in memory
pub fn is_contiguous(t: Tensor) -> Bool {
  case t {
    Tensor(_, _) -> True
    NativeTensor(_, _) -> True
    StridedTensor(_, shape, strides, _) -> {
      let expected_strides = compute_strides(shape)
      strides == expected_strides
    }
  }
}

/// Get element with O(1) access for StridedTensor
pub fn get_fast(t: Tensor, index: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(data, _) ->
      list_at_float(data, index)
      |> result.map_error(fn(_) {
        DimensionError("Index " <> int.to_string(index) <> " out of bounds")
      })
    NativeTensor(_, _) -> get(t, index)
    StridedTensor(storage, shape, strides, offset) -> {
      let indices = flat_to_multi(index, shape)
      let flat_idx =
        list.zip(indices, strides)
        |> list.fold(offset, fn(acc, pair) {
          let #(idx, stride) = pair
          acc + idx * stride
        })
      Ok(ffi.array_get(storage, flat_idx))
    }
  }
}

/// Get 2D element with O(1) access
pub fn get2d_fast(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(_, _) -> get2d(t, row, col)
    NativeTensor(_, _) -> get2d(t, row, col)
    StridedTensor(storage, shape, strides, offset) -> {
      case shape, strides {
        [_rows, _cols], [s0, s1] -> {
          let flat_idx = offset + row * s0 + col * s1
          Ok(ffi.array_get(storage, flat_idx))
        }
        _, _ -> Error(DimensionError("Tensor is not 2D"))
      }
    }
  }
}

// --- Internal Helpers ---

fn list_at_int(lst: List(Int), index: Int) -> Result(Int, Nil) {
  layout_math.at(lst, index)
}

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  layout_math.at(lst, index)
}

fn list_at_tensor(lst: List(Tensor), index: Int) -> Result(Tensor, Nil) {
  layout_math.at(lst, index)
}

fn list_at_float_list(
  lst: List(List(Float)),
  index: Int,
) -> Result(List(Float), Nil) {
  layout_math.at(lst, index)
}

fn materialize_many(
  tensors: List(Tensor),
) -> Result(List(List(Float)), TensorError) {
  tensors
  |> list.fold(Ok([]), fn(acc, tensor) {
    use values <- result.try(acc)
    use data <- result.try(try_to_list(tensor))
    Ok([data, ..values])
  })
  |> result.map(list.reverse)
}

fn slice_bounds_valid(
  shape: List(Int),
  start: List(Int),
  lengths: List(Int),
) -> Bool {
  list.zip(shape, list.zip(start, lengths))
  |> list.all(fn(item) {
    let #(dim, bounds) = item
    let #(offset, len) = bounds
    offset >= 0 && len >= 0 && offset + len <= dim
  })
}

fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  layout_math.flat_to_multi(flat, shape)
}

fn compute_strides(shape: List(Int)) -> List(Int) {
  layout_math.compute_strides(shape)
}

fn broadcast_strides(
  src_shape: List(Int),
  src_strides: List(Int),
  target_shape: List(Int),
) -> List(Int) {
  layout_math.broadcast_strides(src_shape, src_strides, target_shape)
}

fn broadcast_data(
  t: Tensor,
  target_shape: List(Int),
) -> Result(List(Float), TensorError) {
  let target_size = list.fold(target_shape, 1, fn(acc, dim) { acc * dim })
  let src_shape = t.shape
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  use data <- result.try(try_to_list(t))

  let diff = target_rank - src_rank
  let padded_shape = list.append(list.repeat(1, diff), src_shape)

  list.range(0, target_size - 1)
  |> list.fold(Ok([]), fn(acc, flat_idx) {
    use values <- result.try(acc)
    let target_indices = flat_to_multi(flat_idx, target_shape)

    let src_indices =
      list.zip(target_indices, padded_shape)
      |> list.map(fn(pair) {
        let #(idx, dim) = pair
        case dim == 1 {
          True -> 0
          False -> idx
        }
      })
      |> list.drop(diff)

    let src_flat = multi_to_flat(src_indices, src_shape)
    use value <- result.try(
      list_at_float(data, src_flat)
      |> result.map_error(fn(_) {
        IndexOutOfBounds(src_flat, list.length(data))
      }),
    )
    Ok([value, ..values])
  })
  |> result.map(list.reverse)
}

fn multi_to_flat(indices: List(Int), shape: List(Int)) -> Int {
  layout_math.multi_to_flat(indices, shape)
}

// --- Convolution & Pooling ---
// LeCun et al. (1989) - Backpropagation Applied to Handwritten Zip Code Recognition
// The paper that started it all. ConvNets are now 35+ years old!
//
// Why no im2col? Because direct convolution is clearer and im2col wastes memory.
// For production, use the NIF with BLAS/cuDNN anyway.

/// Conv2D configuration
pub type Conv2dConfig {
  Conv2dConfig(
    kernel_h: Int,
    kernel_w: Int,
    stride_h: Int,
    stride_w: Int,
    padding_h: Int,
    padding_w: Int,
  )
}

/// Default conv2d config (3x3 kernel, stride 1, no padding)
pub fn conv2d_config() -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: 3,
    kernel_w: 3,
    stride_h: 1,
    stride_w: 1,
    padding_h: 0,
    padding_w: 0,
  )
}

/// Conv2d config with "same" padding (output same size as input)
pub fn conv2d_same(kernel_h: Int, kernel_w: Int) -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: 1,
    stride_w: 1,
    padding_h: kernel_h / 2,
    padding_w: kernel_w / 2,
  )
}

/// Pad a 2D tensor with zeros
/// Input: [H, W], Output: [H + 2*pad_h, W + 2*pad_w]
pub fn pad2d(t: Tensor, pad_h: Int, pad_w: Int) -> Result(Tensor, TensorError) {
  let shp = shape(t)
  case shp {
    [h, w] -> {
      let new_h = h + 2 * pad_h
      let new_w = w + 2 * pad_w
      let data = get_data(t)

      // Build padded data row by row
      let padded =
        list.range(0, new_h - 1)
        |> list.flat_map(fn(row) {
          list.range(0, new_w - 1)
          |> list.map(fn(col) {
            let src_row = row - pad_h
            let src_col = col - pad_w
            case src_row >= 0 && src_row < h && src_col >= 0 && src_col < w {
              True -> {
                let idx = src_row * w + src_col
                case list_at_float(data, idx) {
                  Ok(v) -> v
                  Error(_) -> 0.0
                }
              }
              False -> 0.0
            }
          })
        })

      Ok(Tensor(data: padded, shape: [new_h, new_w]))
    }
    _ -> Error(InvalidShape(reason: "pad2d requires 2D tensor [H, W]"))
  }
}

/// Pad a 4D tensor (batch) with zeros
/// Input: [N, C, H, W], Output: [N, C, H + 2*pad_h, W + 2*pad_w]
pub fn pad4d(t: Tensor, pad_h: Int, pad_w: Int) -> Result(Tensor, TensorError) {
  let shp = shape(t)
  case shp {
    [n, c, h, w] -> {
      let new_h = h + 2 * pad_h
      let new_w = w + 2 * pad_w
      let data = get_data(t)
      let spatial_size = h * w
      let _new_spatial_size = new_h * new_w

      // Process each batch and channel
      let padded =
        list.range(0, n - 1)
        |> list.flat_map(fn(batch) {
          list.range(0, c - 1)
          |> list.flat_map(fn(channel) {
            let base_idx = batch * c * spatial_size + channel * spatial_size

            list.range(0, new_h - 1)
            |> list.flat_map(fn(row) {
              list.range(0, new_w - 1)
              |> list.map(fn(col) {
                let src_row = row - pad_h
                let src_col = col - pad_w
                case
                  src_row >= 0 && src_row < h && src_col >= 0 && src_col < w
                {
                  True -> {
                    let idx = base_idx + src_row * w + src_col
                    case list_at_float(data, idx) {
                      Ok(v) -> v
                      Error(_) -> 0.0
                    }
                  }
                  False -> 0.0
                }
              })
            })
          })
        })

      Ok(Tensor(data: padded, shape: [n, c, new_h, new_w]))
    }
    _ -> Error(InvalidShape(reason: "pad4d requires 4D tensor [N, C, H, W]"))
  }
}

/// 2D Convolution with optimized O(1) array access.
///
/// Output size formula: O = floor((I - K + 2P) / S) + 1
/// where I = input size, K = kernel size, P = padding, S = stride
///
/// Input shapes supported:
/// - [H, W] with kernel [KH, KW] -> [H_out, W_out]
/// - [C, H, W] with kernel [C, KH, KW] -> [H_out, W_out]
/// - [N, C_in, H, W] with kernel [C_out, C_in, KH, KW] -> [N, C_out, H_out, W_out]
pub fn conv2d(
  input: Tensor,
  kernel: Tensor,
  config: Conv2dConfig,
) -> Result(Tensor, TensorError) {
  let in_shape = shape(input)
  let k_shape = shape(kernel)

  case in_shape, k_shape {
    // Simple 2D conv: [H, W] * [KH, KW] -> [H_out, W_out]
    [h, w], [kh, kw] -> {
      conv2d_simple(input, kernel, h, w, kh, kw, config)
    }

    // Multi-channel: [C, H, W] * [C, KH, KW] -> [H_out, W_out]
    [c_in, h, w], [c_k, kh, kw] if c_in == c_k -> {
      conv2d_multichannel(input, kernel, c_in, h, w, kh, kw, config)
    }

    // Full conv: [N, C_in, H, W] * [C_out, C_in, KH, KW] -> [N, C_out, H_out, W_out]
    [n, c_in, h, w], [c_out, c_k, kh, kw] if c_in == c_k -> {
      conv2d_full(input, kernel, n, c_in, c_out, h, w, kh, kw, config)
    }

    _, _ ->
      Error(InvalidShape(
        reason: "conv2d shape mismatch: input="
        <> shape_to_string(in_shape)
        <> " kernel="
        <> shape_to_string(k_shape),
      ))
  }
}

/// Simple 2D convolution (single channel) with O(1) array access
fn conv2d_simple(
  input: Tensor,
  kernel: Tensor,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  config: Conv2dConfig,
) -> Result(Tensor, TensorError) {
  // Apply padding if needed
  use padded <- result.try(case config.padding_h > 0 || config.padding_w > 0 {
    True -> pad2d(input, config.padding_h, config.padding_w)
    False -> Ok(input)
  })

  let padded_shape = shape(padded)
  let #(ph, pw) = case padded_shape {
    [ph, pw] -> #(ph, pw)
    _ -> #(h, w)
  }

  // Output size: O = floor((I - K + 2P) / S) + 1 (padding already applied)
  let out_h = { ph - kh } / config.stride_h + 1
  let out_w = { pw - kw } / config.stride_w + 1

  // Convert to arrays for O(1) access
  let in_arr = ffi.list_to_array(get_data(padded))
  let k_arr = ffi.list_to_array(get_data(kernel))

  // Compute output using direct array access
  let output =
    conv2d_simple_loop(
      in_arr,
      k_arr,
      pw,
      kh,
      kw,
      config.stride_h,
      config.stride_w,
      out_h,
      out_w,
      0,
      0,
      [],
    )

  Ok(Tensor(data: list.reverse(output), shape: [out_h, out_w]))
}

/// Tail-recursive conv2d loop with O(1) array access
fn conv2d_simple_loop(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  in_w: Int,
  kh: Int,
  kw: Int,
  stride_h: Int,
  stride_w: Int,
  out_h: Int,
  out_w: Int,
  oh: Int,
  ow: Int,
  acc: List(Float),
) -> List(Float) {
  case oh >= out_h {
    True -> acc
    False -> {
      case ow >= out_w {
        True ->
          conv2d_simple_loop(
            in_arr,
            k_arr,
            in_w,
            kh,
            kw,
            stride_h,
            stride_w,
            out_h,
            out_w,
            oh + 1,
            0,
            acc,
          )
        False -> {
          let row = oh * stride_h
          let col = ow * stride_w

          // Compute dot product inline
          let val =
            conv2d_dot_product(in_arr, k_arr, in_w, row, col, kh, kw, 0, 0, 0.0)

          conv2d_simple_loop(
            in_arr,
            k_arr,
            in_w,
            kh,
            kw,
            stride_h,
            stride_w,
            out_h,
            out_w,
            oh,
            ow + 1,
            [val, ..acc],
          )
        }
      }
    }
  }
}

/// Compute dot product of kernel with input patch - O(1) access per element
fn conv2d_dot_product(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  in_w: Int,
  row: Int,
  col: Int,
  kh: Int,
  kw: Int,
  kr: Int,
  kc: Int,
  acc: Float,
) -> Float {
  case kr >= kh {
    True -> acc
    False -> {
      case kc >= kw {
        True ->
          conv2d_dot_product(
            in_arr,
            k_arr,
            in_w,
            row,
            col,
            kh,
            kw,
            kr + 1,
            0,
            acc,
          )
        False -> {
          let in_idx = { row + kr } * in_w + { col + kc }
          let k_idx = kr * kw + kc
          let in_val = ffi.array_get(in_arr, in_idx)
          let k_val = ffi.array_get(k_arr, k_idx)

          conv2d_dot_product(
            in_arr,
            k_arr,
            in_w,
            row,
            col,
            kh,
            kw,
            kr,
            kc + 1,
            acc +. in_val *. k_val,
          )
        }
      }
    }
  }
}

/// Multi-channel convolution (sum over channels)
fn conv2d_multichannel(
  input: Tensor,
  kernel: Tensor,
  c_in: Int,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  config: Conv2dConfig,
) -> Result(Tensor, TensorError) {
  let out_h = { h + 2 * config.padding_h - kh } / config.stride_h + 1
  let out_w = { w + 2 * config.padding_w - kw } / config.stride_w + 1
  let spatial_size = h * w
  let k_spatial = kh * kw

  // Convert to arrays for O(1) access
  let in_arr = ffi.list_to_array(get_data(input))
  let k_arr = ffi.list_to_array(get_data(kernel))

  let output =
    conv2d_mc_loop(
      in_arr,
      k_arr,
      c_in,
      h,
      w,
      kh,
      kw,
      spatial_size,
      k_spatial,
      config.stride_h,
      config.stride_w,
      config.padding_h,
      config.padding_w,
      out_h,
      out_w,
      0,
      0,
      [],
    )

  Ok(Tensor(data: list.reverse(output), shape: [out_h, out_w]))
}

/// Multi-channel conv loop (tail recursive)
fn conv2d_mc_loop(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  c_in: Int,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  spatial_size: Int,
  k_spatial: Int,
  stride_h: Int,
  stride_w: Int,
  pad_h: Int,
  pad_w: Int,
  out_h: Int,
  out_w: Int,
  oh: Int,
  ow: Int,
  acc: List(Float),
) -> List(Float) {
  case oh >= out_h {
    True -> acc
    False -> {
      case ow >= out_w {
        True ->
          conv2d_mc_loop(
            in_arr,
            k_arr,
            c_in,
            h,
            w,
            kh,
            kw,
            spatial_size,
            k_spatial,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            oh + 1,
            0,
            acc,
          )
        False -> {
          let row = oh * stride_h - pad_h
          let col = ow * stride_w - pad_w

          // Sum over all channels
          let val =
            conv2d_mc_channels(
              in_arr,
              k_arr,
              c_in,
              h,
              w,
              kh,
              kw,
              spatial_size,
              k_spatial,
              row,
              col,
              0,
              0.0,
            )

          conv2d_mc_loop(
            in_arr,
            k_arr,
            c_in,
            h,
            w,
            kh,
            kw,
            spatial_size,
            k_spatial,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            oh,
            ow + 1,
            [val, ..acc],
          )
        }
      }
    }
  }
}

/// Sum over input channels
fn conv2d_mc_channels(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  c_in: Int,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  spatial_size: Int,
  k_spatial: Int,
  row: Int,
  col: Int,
  c: Int,
  acc: Float,
) -> Float {
  case c >= c_in {
    True -> acc
    False -> {
      let ch_offset = c * spatial_size
      let k_offset = c * k_spatial

      let channel_sum =
        conv2d_kernel_sum(
          in_arr,
          k_arr,
          h,
          w,
          kh,
          kw,
          ch_offset,
          k_offset,
          row,
          col,
          0,
          0,
          0.0,
        )

      conv2d_mc_channels(
        in_arr,
        k_arr,
        c_in,
        h,
        w,
        kh,
        kw,
        spatial_size,
        k_spatial,
        row,
        col,
        c + 1,
        acc +. channel_sum,
      )
    }
  }
}

/// Sum over kernel window with bounds checking
fn conv2d_kernel_sum(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  ch_offset: Int,
  k_offset: Int,
  row: Int,
  col: Int,
  kr: Int,
  kc: Int,
  acc: Float,
) -> Float {
  case kr >= kh {
    True -> acc
    False -> {
      case kc >= kw {
        True ->
          conv2d_kernel_sum(
            in_arr,
            k_arr,
            h,
            w,
            kh,
            kw,
            ch_offset,
            k_offset,
            row,
            col,
            kr + 1,
            0,
            acc,
          )
        False -> {
          let r = row + kr
          let c_pos = col + kc

          let in_val = case r >= 0 && r < h && c_pos >= 0 && c_pos < w {
            True -> ffi.array_get(in_arr, ch_offset + r * w + c_pos)
            False -> 0.0
          }

          let k_val = ffi.array_get(k_arr, k_offset + kr * kw + kc)

          conv2d_kernel_sum(
            in_arr,
            k_arr,
            h,
            w,
            kh,
            kw,
            ch_offset,
            k_offset,
            row,
            col,
            kr,
            kc + 1,
            acc +. in_val *. k_val,
          )
        }
      }
    }
  }
}

/// Full batched convolution with O(1) array access
fn conv2d_full(
  input: Tensor,
  kernel: Tensor,
  n: Int,
  c_in: Int,
  c_out: Int,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  config: Conv2dConfig,
) -> Result(Tensor, TensorError) {
  let out_h = { h + 2 * config.padding_h - kh } / config.stride_h + 1
  let out_w = { w + 2 * config.padding_w - kw } / config.stride_w + 1
  let in_spatial = h * w
  let in_batch_size = c_in * in_spatial
  let k_spatial = kh * kw
  let k_filter_size = c_in * k_spatial

  // Convert to arrays for O(1) access
  let in_arr = ffi.list_to_array(get_data(input))
  let k_arr = ffi.list_to_array(get_data(kernel))

  let output =
    conv2d_full_loop(
      in_arr,
      k_arr,
      n,
      c_in,
      c_out,
      h,
      w,
      kh,
      kw,
      in_spatial,
      in_batch_size,
      k_spatial,
      k_filter_size,
      config.stride_h,
      config.stride_w,
      config.padding_h,
      config.padding_w,
      out_h,
      out_w,
      0,
      0,
      0,
      0,
      [],
    )

  Ok(Tensor(data: list.reverse(output), shape: [n, c_out, out_h, out_w]))
}

/// Full conv loop: batch -> output_channel -> oh -> ow
fn conv2d_full_loop(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  n: Int,
  c_in: Int,
  c_out: Int,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  in_spatial: Int,
  in_batch_size: Int,
  k_spatial: Int,
  k_filter_size: Int,
  stride_h: Int,
  stride_w: Int,
  pad_h: Int,
  pad_w: Int,
  out_h: Int,
  out_w: Int,
  batch: Int,
  oc: Int,
  oh: Int,
  ow: Int,
  acc: List(Float),
) -> List(Float) {
  case batch >= n {
    True -> acc
    False -> {
      case oc >= c_out {
        True ->
          conv2d_full_loop(
            in_arr,
            k_arr,
            n,
            c_in,
            c_out,
            h,
            w,
            kh,
            kw,
            in_spatial,
            in_batch_size,
            k_spatial,
            k_filter_size,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            out_h,
            out_w,
            batch + 1,
            0,
            0,
            0,
            acc,
          )
        False -> {
          case oh >= out_h {
            True ->
              conv2d_full_loop(
                in_arr,
                k_arr,
                n,
                c_in,
                c_out,
                h,
                w,
                kh,
                kw,
                in_spatial,
                in_batch_size,
                k_spatial,
                k_filter_size,
                stride_h,
                stride_w,
                pad_h,
                pad_w,
                out_h,
                out_w,
                batch,
                oc + 1,
                0,
                0,
                acc,
              )
            False -> {
              case ow >= out_w {
                True ->
                  conv2d_full_loop(
                    in_arr,
                    k_arr,
                    n,
                    c_in,
                    c_out,
                    h,
                    w,
                    kh,
                    kw,
                    in_spatial,
                    in_batch_size,
                    k_spatial,
                    k_filter_size,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    out_h,
                    out_w,
                    batch,
                    oc,
                    oh + 1,
                    0,
                    acc,
                  )
                False -> {
                  let batch_offset = batch * in_batch_size
                  let filter_offset = oc * k_filter_size
                  let row = oh * stride_h - pad_h
                  let col = ow * stride_w - pad_w

                  // Sum over all input channels
                  let val =
                    conv2d_full_channels(
                      in_arr,
                      k_arr,
                      c_in,
                      h,
                      w,
                      kh,
                      kw,
                      in_spatial,
                      k_spatial,
                      batch_offset,
                      filter_offset,
                      row,
                      col,
                      0,
                      0.0,
                    )

                  conv2d_full_loop(
                    in_arr,
                    k_arr,
                    n,
                    c_in,
                    c_out,
                    h,
                    w,
                    kh,
                    kw,
                    in_spatial,
                    in_batch_size,
                    k_spatial,
                    k_filter_size,
                    stride_h,
                    stride_w,
                    pad_h,
                    pad_w,
                    out_h,
                    out_w,
                    batch,
                    oc,
                    oh,
                    ow + 1,
                    [val, ..acc],
                  )
                }
              }
            }
          }
        }
      }
    }
  }
}

/// Sum over input channels for full conv
fn conv2d_full_channels(
  in_arr: ErlangArray,
  k_arr: ErlangArray,
  c_in: Int,
  h: Int,
  w: Int,
  kh: Int,
  kw: Int,
  in_spatial: Int,
  k_spatial: Int,
  batch_offset: Int,
  filter_offset: Int,
  row: Int,
  col: Int,
  ic: Int,
  acc: Float,
) -> Float {
  case ic >= c_in {
    True -> acc
    False -> {
      let ch_offset = batch_offset + ic * in_spatial
      let k_ch_offset = filter_offset + ic * k_spatial

      let sum =
        conv2d_kernel_sum(
          in_arr,
          k_arr,
          h,
          w,
          kh,
          kw,
          ch_offset,
          k_ch_offset,
          row,
          col,
          0,
          0,
          0.0,
        )

      conv2d_full_channels(
        in_arr,
        k_arr,
        c_in,
        h,
        w,
        kh,
        kw,
        in_spatial,
        k_spatial,
        batch_offset,
        filter_offset,
        row,
        col,
        ic + 1,
        acc +. sum,
      )
    }
  }
}

// --- Pooling Operations ---
// Scherer et al. (2010) - Evaluation of Pooling Operations in Convolutional Architectures
// Spoiler: max pooling usually wins, but average pooling has its uses.

/// Max pooling 2D with O(1) array access
/// Input: [H, W] or [N, C, H, W]
/// Output: [H_out, W_out] or [N, C, H_out, W_out]
pub fn max_pool2d(
  input: Tensor,
  pool_h: Int,
  pool_w: Int,
  stride_h: Int,
  stride_w: Int,
) -> Result(Tensor, TensorError) {
  let shp = shape(input)
  let arr = ffi.list_to_array(get_data(input))

  case shp {
    [h, w] -> {
      let out_h = { h - pool_h } / stride_h + 1
      let out_w = { w - pool_w } / stride_w + 1

      let output =
        pool2d_loop(
          arr,
          w,
          pool_h,
          pool_w,
          stride_h,
          stride_w,
          out_h,
          out_w,
          0,
          0,
          0,
          True,
          [],
        )

      Ok(Tensor(data: list.reverse(output), shape: [out_h, out_w]))
    }

    [n, c, h, w] -> {
      let out_h = { h - pool_h } / stride_h + 1
      let out_w = { w - pool_w } / stride_w + 1
      let spatial_size = h * w
      let batch_size = c * spatial_size

      let output =
        pool4d_loop(
          arr,
          n,
          c,
          w,
          pool_h,
          pool_w,
          stride_h,
          stride_w,
          spatial_size,
          batch_size,
          out_h,
          out_w,
          0,
          0,
          0,
          0,
          True,
          [],
        )

      Ok(Tensor(data: list.reverse(output), shape: [n, c, out_h, out_w]))
    }

    _ -> Error(InvalidShape(reason: "max_pool2d requires 2D or 4D tensor"))
  }
}

/// 2D pooling loop (tail recursive)
fn pool2d_loop(
  arr: ErlangArray,
  w: Int,
  pool_h: Int,
  pool_w: Int,
  stride_h: Int,
  stride_w: Int,
  out_h: Int,
  out_w: Int,
  oh: Int,
  ow: Int,
  base: Int,
  is_max: Bool,
  acc: List(Float),
) -> List(Float) {
  case oh >= out_h {
    True -> acc
    False -> {
      case ow >= out_w {
        True ->
          pool2d_loop(
            arr,
            w,
            pool_h,
            pool_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
            oh + 1,
            0,
            base,
            is_max,
            acc,
          )
        False -> {
          let row = oh * stride_h
          let col = ow * stride_w

          let val =
            pool_window(
              arr,
              w,
              row,
              col,
              pool_h,
              pool_w,
              base,
              0,
              0,
              is_max,
              case is_max {
                True -> -1.0e308
                False -> 0.0
              },
            )

          let final_val = case is_max {
            True -> val
            False -> val /. int.to_float(pool_h * pool_w)
          }

          pool2d_loop(
            arr,
            w,
            pool_h,
            pool_w,
            stride_h,
            stride_w,
            out_h,
            out_w,
            oh,
            ow + 1,
            base,
            is_max,
            [final_val, ..acc],
          )
        }
      }
    }
  }
}

/// 4D pooling loop: batch -> channel -> oh -> ow
fn pool4d_loop(
  arr: ErlangArray,
  n: Int,
  c: Int,
  w: Int,
  pool_h: Int,
  pool_w: Int,
  stride_h: Int,
  stride_w: Int,
  spatial_size: Int,
  batch_size: Int,
  out_h: Int,
  out_w: Int,
  batch: Int,
  channel: Int,
  oh: Int,
  ow: Int,
  is_max: Bool,
  acc: List(Float),
) -> List(Float) {
  case batch >= n {
    True -> acc
    False -> {
      case channel >= c {
        True ->
          pool4d_loop(
            arr,
            n,
            c,
            w,
            pool_h,
            pool_w,
            stride_h,
            stride_w,
            spatial_size,
            batch_size,
            out_h,
            out_w,
            batch + 1,
            0,
            0,
            0,
            is_max,
            acc,
          )
        False -> {
          case oh >= out_h {
            True ->
              pool4d_loop(
                arr,
                n,
                c,
                w,
                pool_h,
                pool_w,
                stride_h,
                stride_w,
                spatial_size,
                batch_size,
                out_h,
                out_w,
                batch,
                channel + 1,
                0,
                0,
                is_max,
                acc,
              )
            False -> {
              case ow >= out_w {
                True ->
                  pool4d_loop(
                    arr,
                    n,
                    c,
                    w,
                    pool_h,
                    pool_w,
                    stride_h,
                    stride_w,
                    spatial_size,
                    batch_size,
                    out_h,
                    out_w,
                    batch,
                    channel,
                    oh + 1,
                    0,
                    is_max,
                    acc,
                  )
                False -> {
                  let base = batch * batch_size + channel * spatial_size
                  let row = oh * stride_h
                  let col = ow * stride_w

                  let val =
                    pool_window(
                      arr,
                      w,
                      row,
                      col,
                      pool_h,
                      pool_w,
                      base,
                      0,
                      0,
                      is_max,
                      case is_max {
                        True -> -1.0e308
                        False -> 0.0
                      },
                    )

                  let final_val = case is_max {
                    True -> val
                    False -> val /. int.to_float(pool_h * pool_w)
                  }

                  pool4d_loop(
                    arr,
                    n,
                    c,
                    w,
                    pool_h,
                    pool_w,
                    stride_h,
                    stride_w,
                    spatial_size,
                    batch_size,
                    out_h,
                    out_w,
                    batch,
                    channel,
                    oh,
                    ow + 1,
                    is_max,
                    [final_val, ..acc],
                  )
                }
              }
            }
          }
        }
      }
    }
  }
}

/// Pool over a window - returns max or sum depending on is_max
fn pool_window(
  arr: ErlangArray,
  w: Int,
  row: Int,
  col: Int,
  pool_h: Int,
  pool_w: Int,
  base: Int,
  pr: Int,
  pc: Int,
  is_max: Bool,
  acc: Float,
) -> Float {
  case pr >= pool_h {
    True -> acc
    False -> {
      case pc >= pool_w {
        True ->
          pool_window(
            arr,
            w,
            row,
            col,
            pool_h,
            pool_w,
            base,
            pr + 1,
            0,
            is_max,
            acc,
          )
        False -> {
          let idx = base + { row + pr } * w + { col + pc }
          let val = ffi.array_get(arr, idx)

          let new_acc = case is_max {
            True ->
              case val >. acc {
                True -> val
                False -> acc
              }
            False -> acc +. val
          }

          pool_window(
            arr,
            w,
            row,
            col,
            pool_h,
            pool_w,
            base,
            pr,
            pc + 1,
            is_max,
            new_acc,
          )
        }
      }
    }
  }
}

/// Average pooling 2D with O(1) array access
pub fn avg_pool2d(
  input: Tensor,
  pool_h: Int,
  pool_w: Int,
  stride_h: Int,
  stride_w: Int,
) -> Result(Tensor, TensorError) {
  let shp = shape(input)
  let arr = ffi.list_to_array(get_data(input))

  case shp {
    [h, w] -> {
      let out_h = { h - pool_h } / stride_h + 1
      let out_w = { w - pool_w } / stride_w + 1

      let output =
        pool2d_loop(
          arr,
          w,
          pool_h,
          pool_w,
          stride_h,
          stride_w,
          out_h,
          out_w,
          0,
          0,
          0,
          False,
          [],
        )

      Ok(Tensor(data: list.reverse(output), shape: [out_h, out_w]))
    }

    [n, c, h, w] -> {
      let out_h = { h - pool_h } / stride_h + 1
      let out_w = { w - pool_w } / stride_w + 1
      let spatial_size = h * w
      let batch_size = c * spatial_size

      let output =
        pool4d_loop(
          arr,
          n,
          c,
          w,
          pool_h,
          pool_w,
          stride_h,
          stride_w,
          spatial_size,
          batch_size,
          out_h,
          out_w,
          0,
          0,
          0,
          0,
          False,
          [],
        )

      Ok(Tensor(data: list.reverse(output), shape: [n, c, out_h, out_w]))
    }

    _ -> Error(InvalidShape(reason: "avg_pool2d requires 2D or 4D tensor"))
  }
}

/// Global average pooling - reduces spatial dimensions to 1x1.
/// The modern replacement for flatten+dense in classification heads.
/// Lin et al. (2013) - Network In Network
///
/// Input: [N, C, H, W] -> Output: [N, C, 1, 1]
pub fn global_avg_pool2d(input: Tensor) -> Result(Tensor, TensorError) {
  let shp = shape(input)

  case shp {
    [n, c, h, w] -> {
      let spatial_size = h * w
      let pool_size = int.to_float(spatial_size)
      let batch_size = c * spatial_size
      let data = get_data(input)

      let output =
        list.range(0, n - 1)
        |> list.flat_map(fn(batch) {
          list.range(0, c - 1)
          |> list.map(fn(channel) {
            let base = batch * batch_size + channel * spatial_size

            // Average over entire spatial dimension
            list.range(0, spatial_size - 1)
            |> list.fold(0.0, fn(sum, i) {
              case list_at_float(data, base + i) {
                Ok(v) -> sum +. v
                Error(_) -> sum
              }
            })
            |> fn(s) { s /. pool_size }
          })
        })

      Ok(Tensor(data: output, shape: [n, c, 1, 1]))
    }

    _ ->
      Error(InvalidShape(
        reason: "global_avg_pool2d requires 4D tensor [N, C, H, W]",
      ))
  }
}

fn shape_to_string(shp: List(Int)) -> String {
  "[" <> list.map(shp, int.to_string) |> string_join(", ") <> "]"
}

fn string_join(strings: List(String), sep: String) -> String {
  case strings {
    [] -> ""
    [s] -> s
    [s, ..rest] -> s <> sep <> string_join(rest, sep)
  }
}
