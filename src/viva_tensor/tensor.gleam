//// Tensor - N-dimensional arrays for numerical computing
////
//// "All you need is shapes" - a tensor library author, probably
////
//// This is the PUBLIC FACADE over core/tensor.gleam. Why two modules?
//// Because opaque types are great until you need pattern matching in tests.
//// This module gives you the nice API; core/ gives you the guarantees.
////
//// Design: NumPy-inspired with strides for zero-copy views.
//// Uses Erlang :array for O(1) access + strides for efficient transpose/reshape.
//// For matrices > 100x100, enable the NIF. The difference is 100-1000x.

import gleam/float
import gleam/int
import gleam/list
import gleam/result

// --- Types ---

/// Opaque type for Erlang :array
pub type ErlangArray

/// Tensor with NumPy-style strides for zero-copy views
/// - storage: contiguous data buffer (Erlang array for O(1) access)
/// - shape: dimensions [d0, d1, ..., dn]
/// - strides: bytes to skip for each dimension [s0, s1, ..., sn]
/// - offset: starting position in storage (for views/slices)
///
/// NCHW vs NHWC: we use NCHW because that's what the cool kids (PyTorch) do.
/// TensorFlow uses NHWC by default, but they also use Python so...
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
  StridedTensor(
    storage: ErlangArray,
    shape: List(Int),
    strides: List(Int),
    offset: Int,
  )
}

/// Tensor operation errors
pub type TensorError {
  ShapeMismatch(expected: List(Int), got: List(Int))
  InvalidShape(reason: String)
  DimensionError(reason: String)
  BroadcastError(a: List(Int), b: List(Int))
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

// --- Properties ---

/// Get tensor shape
pub fn shape(t: Tensor) -> List(Int) {
  case t {
    Tensor(_, s) -> s
    StridedTensor(_, s, _, _) -> s
  }
}

/// Extract data as list from any tensor variant
pub fn get_data(t: Tensor) -> List(Float) {
  case t {
    Tensor(data, _) -> data
    StridedTensor(storage, shape, strides, offset) -> {
      let total_size = list.fold(shape, 1, fn(acc, dim) { acc * dim })
      list.range(0, total_size - 1)
      |> list.map(fn(flat_idx) {
        let indices = flat_to_multi(flat_idx, shape)
        let idx =
          list.zip(indices, strides)
          |> list.fold(offset, fn(acc, pair) {
            let #(i, s) = pair
            acc + i * s
          })
        array_get(storage, idx)
      })
    }
  }
}

/// Total number of elements
pub fn size(t: Tensor) -> Int {
  case t {
    Tensor(data, _) -> list.length(data)
    StridedTensor(_, shape, _, _) ->
      list.fold(shape, 1, fn(acc, dim) { acc * dim })
  }
}

/// Number of dimensions (rank)
pub fn rank(t: Tensor) -> Int {
  list.length(t.shape)
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
    StridedTensor(storage, shape, strides, offset) -> {
      let indices = flat_to_multi(index, shape)
      let flat_idx =
        list.zip(indices, strides)
        |> list.fold(offset, fn(acc, pair) {
          let #(i, s) = pair
          acc + i * s
        })
      Ok(array_get(storage, flat_idx))
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

/// Apply function to each element
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  let data = get_data(t)
  Tensor(data: list.map(data, f), shape: t.shape)
}

/// Apply function with index
pub fn map_indexed(t: Tensor, f: fn(Float, Int) -> Float) -> Tensor {
  let data = get_data(t)
  Tensor(data: list.index_map(data, fn(x, i) { f(x, i) }), shape: t.shape)
}

/// Element-wise addition
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x +. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise subtraction
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x -. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise multiplication (Hadamard product)
/// Not to be confused with matmul. Named after Jacques Hadamard (1865-1963).
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x *. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Element-wise division
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case a.shape == b.shape {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x /. y })
      Ok(Tensor(data: data, shape: a.shape))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Scale by constant
pub fn scale(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x *. s })
}

/// Add constant
pub fn add_scalar(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x +. s })
}

/// Negation
pub fn negate(t: Tensor) -> Tensor {
  scale(t, -1.0)
}

// --- Reduction Operations ---

/// Sum all elements
pub fn sum(t: Tensor) -> Float {
  let data = get_data(t)
  list.fold(data, 0.0, fn(acc, x) { acc +. x })
}

/// Product of all elements
pub fn product(t: Tensor) -> Float {
  let data = get_data(t)
  list.fold(data, 1.0, fn(acc, x) { acc *. x })
}

/// Mean: E[X] = (1/n) * sum(x_i)
pub fn mean(t: Tensor) -> Float {
  let s = sum(t)
  let n = int.to_float(size(t))
  case n >. 0.0 {
    True -> s /. n
    False -> 0.0
  }
}

/// Maximum value
pub fn max(t: Tensor) -> Float {
  let data = get_data(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.max(acc, x) })
  }
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  let data = get_data(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.min(acc, x) })
  }
}

/// Argmax - index of largest element
pub fn argmax(t: Tensor) -> Int {
  let data = get_data(t)
  case data {
    [] -> 0
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x >. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      idx
    }
  }
}

/// Argmin - index of smallest element
pub fn argmin(t: Tensor) -> Int {
  let data = get_data(t)
  case data {
    [] -> 0
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, x) {
          let #(best_idx, best_val, curr_idx) = acc
          case x <. best_val {
            True -> #(curr_idx, x, curr_idx + 1)
            False -> #(best_idx, best_val, curr_idx + 1)
          }
        })
      idx
    }
  }
}

/// Variance: Var(X) = E[X^2] - E[X]^2  (computational form)
/// We use the two-pass algorithm for numerical stability.
pub fn variance(t: Tensor) -> Float {
  let data = get_data(t)
  let m = mean(t)
  let squared_diffs =
    list.map(data, fn(x) {
      let diff = x -. m
      diff *. diff
    })
  let n = int.to_float(size(t))
  case n >. 0.0 {
    True -> list.fold(squared_diffs, 0.0, fn(acc, x) { acc +. x }) /. n
    False -> 0.0
  }
}

/// Standard deviation: sqrt(Var(X))
pub fn std(t: Tensor) -> Float {
  float_sqrt(variance(t))
}

/// Sum along a specific axis
/// For a [2, 3] tensor, sum_axis(_, 0) gives [3], sum_axis(_, 1) gives [2]
pub fn sum_axis(t: Tensor, axis_idx: Int) -> Result(Tensor, TensorError) {
  let r = rank(t)
  case axis_idx >= 0 && axis_idx < r {
    False -> Error(DimensionError("Invalid axis index"))
    True -> {
      case t.shape {
        [] -> Error(DimensionError("Cannot reduce scalar"))
        [_] ->
          // 1D: reduce to scalar wrapped in [1]
          Ok(Tensor(data: [sum(t)], shape: [1]))
        _ -> {
          // General case: reduce along axis
          let axis_size = case list.drop(t.shape, axis_idx) |> list.first {
            Ok(s) -> s
            Error(_) -> 1
          }
          let new_shape = remove_at_index(t.shape, axis_idx)
          let new_size = list.fold(new_shape, 1, fn(acc, d) { acc * d })
          let data = get_data(t)

          let result =
            list.range(0, new_size - 1)
            |> list.map(fn(out_idx) {
              // Sum over all values at this position along the axis
              list.range(0, axis_size - 1)
              |> list.fold(0.0, fn(acc, axis_pos) {
                let in_idx =
                  compute_index_with_axis(t.shape, out_idx, axis_idx, axis_pos)
                let val = case list.drop(data, in_idx) |> list.first {
                  Ok(v) -> v
                  Error(_) -> 0.0
                }
                acc +. val
              })
            })

          Ok(Tensor(data: result, shape: new_shape))
        }
      }
    }
  }
}

/// Mean along a specific axis
pub fn mean_axis(t: Tensor, axis_idx: Int) -> Result(Tensor, TensorError) {
  let r = rank(t)
  case axis_idx >= 0 && axis_idx < r {
    False -> Error(DimensionError("Invalid axis index"))
    True -> {
      let axis_size = case list.drop(t.shape, axis_idx) |> list.first {
        Ok(s) -> s
        Error(_) -> 1
      }
      case sum_axis(t, axis_idx) {
        Error(e) -> Error(e)
        Ok(summed) -> Ok(scale(summed, 1.0 /. int.to_float(axis_size)))
      }
    }
  }
}

fn remove_at_index(lst: List(a), idx: Int) -> List(a) {
  lst
  |> list.index_map(fn(item, i) { #(item, i) })
  |> list.filter(fn(pair) { pair.1 != idx })
  |> list.map(fn(pair) { pair.0 })
}

fn compute_index_with_axis(
  shape: List(Int),
  out_idx: Int,
  axis_idx: Int,
  axis_pos: Int,
) -> Int {
  let strides = compute_strides(shape)
  let _r = list.length(shape)

  // Convert out_idx to coordinates (without axis dimension)
  let shape_without_axis = remove_at_index(shape, axis_idx)
  let strides_without_axis = compute_strides(shape_without_axis)

  let out_coords =
    list.range(0, list.length(shape_without_axis) - 1)
    |> list.map(fn(i) {
      let stride = case list.drop(strides_without_axis, i) |> list.first {
        Ok(s) -> s
        Error(_) -> 1
      }
      { out_idx / stride }
      % {
        case list.drop(shape_without_axis, i) |> list.first {
          Ok(d) -> d
          Error(_) -> 1
        }
      }
    })

  // Insert axis_pos at axis_idx position
  let #(before, after) = list.split(out_coords, axis_idx)
  let full_coords = list.flatten([before, [axis_pos], after])

  // Convert to flat index
  list.zip(full_coords, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(coord, stride) = pair
    acc + coord * stride
  })
}

// --- Matrix Operations ---

/// Dot product of two vectors: a . b = sum(a_i * b_i)
pub fn dot(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case rank(a) == 1 && rank(b) == 1 && size(a) == size(b) {
    True -> {
      let a_data = get_data(a)
      let b_data = get_data(b)
      let products = list.map2(a_data, b_data, fn(x, y) { x *. y })
      Ok(list.fold(products, 0.0, fn(acc, x) { acc +. x }))
    }
    False -> Error(ShapeMismatch(expected: a.shape, got: b.shape))
  }
}

/// Matrix-vector multiplication: [m, n] @ [n] -> [m]
/// C_i = sum_j(A_ij * x_j)
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  case mat.shape, vec.shape {
    [m, n], [vec_n] if n == vec_n -> {
      let mat_data = get_data(mat)
      let vec_data = get_data(vec)
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
      let result_data =
        list.range(0, m - 1)
        |> list.flat_map(fn(i) {
          list.range(0, p - 1)
          |> list.map(fn(j) {
            list.range(0, n - 1)
            |> list.fold(0.0, fn(acc, k) {
              let a_ik = case get2d(a, i, k) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
              let b_kj = case get2d(b, k, j) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
              acc +. a_ik *. b_kj
            })
          })
        })
      Ok(Tensor(data: result_data, shape: [m, p]))
    }
    [_m, n], [n2, _p] -> Error(ShapeMismatch(expected: [n, -1], got: [n2, -1]))
    _, _ -> Error(DimensionError("Expected two matrices"))
  }
}

/// Matrix transpose: A^T where (A^T)_ij = A_ji
pub fn transpose(t: Tensor) -> Result(Tensor, TensorError) {
  case t.shape {
    [m, n] -> {
      let result_data =
        list.range(0, n - 1)
        |> list.flat_map(fn(j) {
          list.range(0, m - 1)
          |> list.filter_map(fn(i) { get2d(t, i, j) })
        })
      Ok(Tensor(data: result_data, shape: [n, m]))
    }
    _ -> Error(DimensionError("Transpose requires 2D tensor"))
  }
}

/// Outer product: [m] x [n] -> [m, n]
/// C_ij = a_i * b_j
pub fn outer(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case rank(a) == 1 && rank(b) == 1 {
    True -> {
      let m = size(a)
      let n = size(b)
      let a_data = get_data(a)
      let b_data = get_data(b)
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
      let data = get_data(t)
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

/// Clone tensor (creates a copy)
pub fn clone(t: Tensor) -> Tensor {
  let data = get_data(t)
  Tensor(data: data, shape: t.shape)
}

/// Reshape tensor - same data, different shape
/// The total number of elements must match.
pub fn reshape(t: Tensor, new_shape: List(Int)) -> Result(Tensor, TensorError) {
  let old_size = size(t)
  let new_size = list.fold(new_shape, 1, fn(acc, dim) { acc * dim })

  case old_size == new_size {
    True -> {
      let data = get_data(t)
      Ok(Tensor(data: data, shape: new_shape))
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

/// Flatten to 1D
pub fn flatten(t: Tensor) -> Tensor {
  let data = get_data(t)
  Tensor(data: data, shape: [size(t)])
}

/// Concatenate vectors (1D)
pub fn concat(tensors: List(Tensor)) -> Tensor {
  let data = list.flat_map(tensors, fn(t) { get_data(t) })
  from_list(data)
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
                  case list.drop(t.shape, axis) |> list.first {
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
                  let data = list.flat_map(tensors, fn(t) { get_data(t) })
                  Ok(Tensor(data: data, shape: new_shape))
                }
                False -> {
                  // General case: interleave based on axis
                  let total_size =
                    list.fold(new_shape, 1, fn(acc, d) { acc * d })
                  let _new_strides = compute_strides(new_shape)

                  let result =
                    list.range(0, total_size - 1)
                    |> list.map(fn(flat_idx) {
                      let indices = flat_to_multi(flat_idx, new_shape)
                      let axis_idx = case
                        list.drop(indices, axis) |> list.first
                      {
                        Ok(i) -> i
                        Error(_) -> 0
                      }

                      // Find which tensor and local index
                      let #(tensor_idx, local_axis_idx, _) =
                        list.fold(tensors, #(-1, axis_idx, 0), fn(acc, t) {
                          let #(found_t, remaining, t_idx) = acc
                          case found_t >= 0 {
                            True -> acc
                            False -> {
                              let t_axis_size = case
                                list.drop(t.shape, axis) |> list.first
                              {
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
                      case list.drop(tensors, tensor_idx) |> list.first {
                        Ok(t) -> {
                          let t_strides = compute_strides(t.shape)
                          let local_flat =
                            list.zip(local_indices, t_strides)
                            |> list.fold(0, fn(a, p) { a + p.0 * p.1 })
                          let t_data = get_data(t)
                          case list.drop(t_data, local_flat) |> list.first {
                            Ok(v) -> v
                            Error(_) -> 0.0
                          }
                        }
                        Error(_) -> 0.0
                      }
                    })

                  Ok(Tensor(data: result, shape: new_shape))
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

/// Take first N elements along first axis
pub fn take_first(t: Tensor, n: Int) -> Tensor {
  let data = get_data(t)
  case t.shape {
    [] -> t
    [first_dim, ..rest_dims] -> {
      let take_n = int.min(n, first_dim)
      let stride = list.fold(rest_dims, 1, fn(acc, d) { acc * d })
      let new_data = list.take(data, take_n * stride)
      let new_shape = [take_n, ..rest_dims]
      Tensor(data: new_data, shape: new_shape)
    }
  }
}

/// Take last N elements along first axis
pub fn take_last(t: Tensor, n: Int) -> Tensor {
  let data = get_data(t)
  case t.shape {
    [] -> t
    [first_dim, ..rest_dims] -> {
      let take_n = int.min(n, first_dim)
      let stride = list.fold(rest_dims, 1, fn(acc, d) { acc * d })
      let skip = { first_dim - take_n } * stride
      let new_data = list.drop(data, skip)
      let new_shape = [take_n, ..rest_dims]
      Tensor(data: new_data, shape: new_shape)
    }
  }
}

/// Slice tensor: extract sub-tensor from start to start+lengths
/// slice(t, [1], [3]) extracts elements at indices 1, 2, 3
pub fn slice(
  t: Tensor,
  start: List(Int),
  lengths: List(Int),
) -> Result(Tensor, TensorError) {
  let data = get_data(t)
  let r = rank(t)

  case list.length(start) == r && list.length(lengths) == r {
    False -> Error(DimensionError("Slice dimensions must match tensor rank"))
    True -> {
      case r {
        1 -> {
          // 1D slice
          let s = case list.first(start) {
            Ok(v) -> v
            Error(_) -> 0
          }
          let len = case list.first(lengths) {
            Ok(v) -> v
            Error(_) -> 0
          }
          let sliced = data |> list.drop(s) |> list.take(len)
          Ok(Tensor(data: sliced, shape: [len]))
        }
        _ -> {
          // Multi-dimensional slice - general case
          let new_size = list.fold(lengths, 1, fn(acc, d) { acc * d })

          let result =
            list.range(0, new_size - 1)
            |> list.map(fn(flat_idx) {
              let local_indices = flat_to_multi(flat_idx, lengths)
              let global_indices =
                list.map2(local_indices, start, fn(l, s) { l + s })
              let global_flat = multi_to_flat(global_indices, t.shape)
              case list.drop(data, global_flat) |> list.first {
                Ok(v) -> v
                Error(_) -> 0.0
              }
            })

          Ok(Tensor(data: result, shape: lengths))
        }
      }
    }
  }
}

/// L2 norm: ||x||_2 = sqrt(sum(x_i^2))
pub fn norm(t: Tensor) -> Float {
  let data = get_data(t)
  let sum_sq = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  float_sqrt(sum_sq)
}

/// Normalize to unit length: x / ||x||_2
pub fn normalize(t: Tensor) -> Tensor {
  let n = norm(t)
  case n >. 0.0001 {
    True -> scale(t, 1.0 /. n)
    False -> t
  }
}

/// Clamp values to [min, max]
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  map(t, fn(x) { float.min(float.max(x, min_val), max_val) })
}

// --- Random ---

/// Tensor with uniform random values [0, 1)
pub fn random_uniform(shape: List(Int)) -> Tensor {
  let size_val = list.fold(shape, 1, fn(acc, dim) { acc * dim })
  let data =
    list.range(1, size_val)
    |> list.map(fn(_) { random_float() })
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
      let u1 = float.max(random_float(), 0.0001)
      let u2 = random_float()
      // Box-Muller: z = sqrt(-2*ln(u1)) * cos(2*pi*u2)
      let z =
        float_sqrt(-2.0 *. float_log(u1))
        *. float_cos(2.0 *. 3.14159265359 *. u2)
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
  let limit = float_sqrt(6.0 /. int.to_float(fan_in + fan_out))
  let data =
    list.range(1, fan_in * fan_out)
    |> list.map(fn(_) {
      let r = random_float()
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
  let std_val = float_sqrt(2.0 /. int.to_float(fan_in))
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
    False -> Error(BroadcastError(a, b))
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

/// Broadcast tensor to target shape.
/// Zero-copy when possible, materialized when necessary.
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  case can_broadcast(t.shape, target_shape) {
    False -> Error(BroadcastError(t.shape, target_shape))
    True -> {
      case t.shape == target_shape {
        True -> Ok(t)
        False -> {
          let data = broadcast_data(t, target_shape)
          Ok(Tensor(data: data, shape: target_shape))
        }
      }
    }
  }
}

/// Element-wise addition with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(a.shape, b.shape))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  add(a_bc, b_bc)
}

/// Element-wise multiplication with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(a.shape, b.shape))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  mul(a_bc, b_bc)
}

// --- Shape Manipulation ---

/// Remove dimensions of size 1 (squeeze operation)
pub fn squeeze(t: Tensor) -> Tensor {
  let data = get_data(t)
  let new_shape = list.filter(t.shape, fn(d) { d != 1 })
  let final_shape = case new_shape {
    [] -> [1]
    _ -> new_shape
  }
  Tensor(data: data, shape: final_shape)
}

/// Remove dimension at specific axis if it's 1
pub fn squeeze_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  case list_at_int(t.shape, axis) {
    Error(_) -> Error(DimensionError("Axis out of bounds"))
    Ok(d) -> {
      case d == 1 {
        False -> Error(InvalidShape("Dimension at axis is not 1"))
        True -> {
          let data = get_data(t)
          let new_shape =
            t.shape
            |> list.index_map(fn(dim, i) { #(dim, i) })
            |> list.filter(fn(pair) { pair.1 != axis })
            |> list.map(fn(pair) { pair.0 })
          Ok(Tensor(data: data, shape: new_shape))
        }
      }
    }
  }
}

/// Add dimension of size 1 at specified axis (unsqueeze operation)
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  let data = get_data(t)
  let rnk = list.length(t.shape)
  let insert_at = case axis < 0 {
    True -> rnk + axis + 1
    False -> axis
  }

  let #(before, after) = list.split(t.shape, insert_at)
  let new_shape = list.flatten([before, [1], after])
  Tensor(data: data, shape: new_shape)
}

/// Expand tensor to add batch dimension (alias for unsqueeze)
pub fn expand_dims(t: Tensor, axis: Int) -> Tensor {
  unsqueeze(t, axis)
}

// --- Strided Tensor (Zero-copy operations) ---

/// Convert regular tensor to strided (O(n) once, then O(1) access)
pub fn to_strided(t: Tensor) -> Tensor {
  case t {
    StridedTensor(_, _, _, _) -> t
    Tensor(data, shape) -> {
      let storage = list_to_array(data)
      let strides = compute_strides(shape)
      StridedTensor(storage: storage, shape: shape, strides: strides, offset: 0)
    }
  }
}

/// Convert strided tensor back to regular (materializes the view)
pub fn to_contiguous(t: Tensor) -> Tensor {
  case t {
    Tensor(_, _) -> t
    StridedTensor(_, _, _, _) -> {
      let data = get_data(t)
      Tensor(data: data, shape: t.shape)
    }
  }
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
    StridedTensor(storage, shape, strides, offset) -> {
      let indices = flat_to_multi(index, shape)
      let flat_idx =
        list.zip(indices, strides)
        |> list.fold(offset, fn(acc, pair) {
          let #(idx, stride) = pair
          acc + idx * stride
        })
      Ok(array_get(storage, flat_idx))
    }
  }
}

/// Get 2D element with O(1) access
pub fn get2d_fast(t: Tensor, row: Int, col: Int) -> Result(Float, TensorError) {
  case t {
    Tensor(_, _) -> get2d(t, row, col)
    StridedTensor(storage, shape, strides, offset) -> {
      case shape, strides {
        [_rows, _cols], [s0, s1] -> {
          let flat_idx = offset + row * s0 + col * s1
          Ok(array_get(storage, flat_idx))
        }
        _, _ -> Error(DimensionError("Tensor is not 2D"))
      }
    }
  }
}

// --- Internal Helpers ---

fn list_at_int(lst: List(Int), index: Int) -> Result(Int, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}

fn array_get(arr: ErlangArray, index: Int) -> Float {
  array_get_ffi(arr, index)
}

fn list_to_array(lst: List(Float)) -> ErlangArray {
  list_to_array_ffi(lst)
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

fn compute_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(strides, _) =
    list.fold(reversed, #([], 1), fn(acc, dim) {
      let #(s, running) = acc
      #([running, ..s], running * dim)
    })
  strides
}

fn broadcast_data(t: Tensor, target_shape: List(Int)) -> List(Float) {
  let target_size = list.fold(target_shape, 1, fn(acc, dim) { acc * dim })
  let src_shape = t.shape
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  let data = get_data(t)

  let diff = target_rank - src_rank
  let padded_shape = list.append(list.repeat(1, diff), src_shape)

  list.range(0, target_size - 1)
  |> list.map(fn(flat_idx) {
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
    case list_at_float(data, src_flat) {
      Ok(v) -> v
      Error(_) -> 0.0
    }
  })
}

fn multi_to_flat(indices: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)

  list.zip(indices, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(idx, stride) = pair
    acc + idx * stride
  })
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
  let in_arr = list_to_array_ffi(get_data(padded))
  let k_arr = list_to_array_ffi(get_data(kernel))

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
          let in_val = array_get_ffi(in_arr, in_idx)
          let k_val = array_get_ffi(k_arr, k_idx)

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
  let in_arr = list_to_array_ffi(get_data(input))
  let k_arr = list_to_array_ffi(get_data(kernel))

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
            True -> array_get_ffi(in_arr, ch_offset + r * w + c_pos)
            False -> 0.0
          }

          let k_val = array_get_ffi(k_arr, k_offset + kr * kw + kc)

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
  let in_arr = list_to_array_ffi(get_data(input))
  let k_arr = list_to_array_ffi(get_data(kernel))

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
  let arr = list_to_array_ffi(get_data(input))

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
          let val = array_get_ffi(arr, idx)

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
  let arr = list_to_array_ffi(get_data(input))

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

// --- FFI (Erlang externals) ---

@external(erlang, "math", "sqrt")
fn float_sqrt(x: Float) -> Float

@external(erlang, "math", "log")
fn float_log(x: Float) -> Float

@external(erlang, "math", "cos")
fn float_cos(x: Float) -> Float

@external(erlang, "rand", "uniform")
fn random_float() -> Float

@external(erlang, "viva_tensor_ffi", "list_to_array")
fn list_to_array_ffi(lst: List(Float)) -> ErlangArray

@external(erlang, "viva_tensor_ffi", "array_get")
fn array_get_ffi(arr: ErlangArray, index: Int) -> Float
