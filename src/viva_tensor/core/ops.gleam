//// Tensor Operations - Mathematical operations on tensors
////
//// This module provides element-wise, reduction, matrix, and broadcasting
//// operations. All operations preserve type safety and validate inputs.
////
//// ## Element-wise Operations
////
//// ```gleam
//// let a = tensor.from_list([1.0, 2.0, 3.0])
//// let b = tensor.from_list([4.0, 5.0, 6.0])
////
//// // Addition, subtraction, multiplication
//// let sum = ops.add(a, b)      // Ok([5.0, 7.0, 9.0])
//// let diff = ops.sub(a, b)     // Ok([-3.0, -3.0, -3.0])
//// let prod = ops.mul(a, b)     // Ok([4.0, 10.0, 18.0])
//// ```
////
//// ## Reductions
////
//// ```gleam
//// let t = tensor.from_list([1.0, 2.0, 3.0, 4.0])
////
//// ops.sum(t)      // 10.0
//// ops.mean(t)     // 2.5
//// ops.max(t)      // 4.0
//// ops.min(t)      // 1.0
//// ```
////
//// ## Matrix Operations
////
//// ```gleam
//// let a = tensor.matrix(rows: 2, cols: 3, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
//// let b = tensor.matrix(rows: 3, cols: 2, data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
////
//// ops.matmul(a, b)  // [2, 2] matrix
//// ```

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/ffi
import viva_tensor/core/tensor.{type Tensor}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

/// Apply a function to each element of a tensor.
///
/// ## Examples
///
/// ```gleam
/// let t = tensor.from_list([1.0, 2.0, 3.0])
/// ops.map(t, fn(x) { x *. 2.0 })
/// // -> [2.0, 4.0, 6.0]
/// ```
pub fn map(t: Tensor, f: fn(Float) -> Float) -> Tensor {
  let data = tensor.to_list(t)
  let new_data = list.map(data, f)
  case tensor.new(new_data, tensor.shape(t)) {
    Ok(result) -> result
    Error(_) -> t
  }
}

/// Apply a function to each element with its index.
///
/// ## Examples
///
/// ```gleam
/// let t = tensor.from_list([10.0, 20.0, 30.0])
/// ops.map_indexed(t, fn(x, i) { x +. int.to_float(i) })
/// // -> [10.0, 21.0, 32.0]
/// ```
pub fn map_indexed(t: Tensor, f: fn(Float, Int) -> Float) -> Tensor {
  let data = tensor.to_list(t)
  let new_data = list.index_map(data, fn(x, i) { f(x, i) })
  case tensor.new(new_data, tensor.shape(t)) {
    Ok(result) -> result
    Error(_) -> t
  }
}

/// Element-wise addition of two tensors.
///
/// Returns `Error(ShapeMismatch)` if shapes don't match.
///
/// ## Examples
///
/// ```gleam
/// let a = tensor.from_list([1.0, 2.0])
/// let b = tensor.from_list([3.0, 4.0])
/// ops.add(a, b)
/// // -> Ok([4.0, 6.0])
/// ```
pub fn add(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x +. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Element-wise subtraction of two tensors.
///
/// ## Examples
///
/// ```gleam
/// let a = tensor.from_list([5.0, 4.0])
/// let b = tensor.from_list([3.0, 1.0])
/// ops.sub(a, b)
/// // -> Ok([2.0, 3.0])
/// ```
pub fn sub(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x -. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Element-wise multiplication (Hadamard product).
///
/// ## Examples
///
/// ```gleam
/// let a = tensor.from_list([2.0, 3.0])
/// let b = tensor.from_list([4.0, 5.0])
/// ops.mul(a, b)
/// // -> Ok([8.0, 15.0])
/// ```
pub fn mul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x *. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Element-wise division of two tensors.
///
/// ## Examples
///
/// ```gleam
/// let a = tensor.from_list([10.0, 20.0])
/// let b = tensor.from_list([2.0, 4.0])
/// ops.div(a, b)
/// // -> Ok([5.0, 5.0])
/// ```
pub fn div(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a) == tensor.shape(b) {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let data = list.map2(a_data, b_data, fn(x, y) { x /. y })
      tensor.new(data, tensor.shape(a))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Multiply all elements by a scalar.
///
/// ## Examples
///
/// ```gleam
/// let t = tensor.from_list([1.0, 2.0, 3.0])
/// ops.scale(t, 2.0)
/// // -> [2.0, 4.0, 6.0]
/// ```
pub fn scale(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x *. s })
}

/// Add a scalar to all elements.
///
/// ## Examples
///
/// ```gleam
/// let t = tensor.from_list([1.0, 2.0, 3.0])
/// ops.add_scalar(t, 10.0)
/// // -> [11.0, 12.0, 13.0]
/// ```
pub fn add_scalar(t: Tensor, s: Float) -> Tensor {
  map(t, fn(x) { x +. s })
}

/// Negate all elements (multiply by -1).
pub fn negate(t: Tensor) -> Tensor {
  scale(t, -1.0)
}

/// Element-wise absolute value
pub fn abs(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.abs(x) })
}

/// Element-wise square
pub fn square(t: Tensor) -> Tensor {
  map(t, fn(x) { x *. x })
}

/// Element-wise square root
pub fn sqrt(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.sqrt(x) })
}

/// Element-wise exponential
pub fn exp(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.exp(x) })
}

/// Element-wise natural log
pub fn log(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.log(x) })
}

/// Element-wise power
pub fn pow(t: Tensor, exponent: Float) -> Tensor {
  map(t, fn(x) { ffi.pow(x, exponent) })
}

// =============================================================================
// REDUCTION OPERATIONS
// =============================================================================

/// Sum all elements
pub fn sum(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  list.fold(data, 0.0, fn(acc, x) { acc +. x })
}

/// Product of all elements
pub fn product(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  list.fold(data, 1.0, fn(acc, x) { acc *. x })
}

/// Mean of all elements
pub fn mean(t: Tensor) -> Float {
  let s = sum(t)
  let n = int.to_float(tensor.size(t))
  case n >. 0.0 {
    True -> s /. n
    False -> 0.0
  }
}

/// Maximum value
pub fn max(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.max(acc, x) })
  }
}

/// Minimum value
pub fn min(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  case data {
    [] -> 0.0
    [first, ..rest] -> list.fold(rest, first, fn(acc, x) { float.min(acc, x) })
  }
}

/// Index of maximum element
pub fn argmax(t: Tensor) -> Int {
  let data = tensor.to_list(t)
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

/// Index of minimum element
pub fn argmin(t: Tensor) -> Int {
  let data = tensor.to_list(t)
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

/// Variance of all elements
pub fn variance(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  let m = mean(t)
  let squared_diffs =
    list.map(data, fn(x) {
      let diff = x -. m
      diff *. diff
    })
  let n = int.to_float(tensor.size(t))
  case n >. 0.0 {
    True -> list.fold(squared_diffs, 0.0, fn(acc, x) { acc +. x }) /. n
    False -> 0.0
  }
}

/// Standard deviation
pub fn std(t: Tensor) -> Float {
  ffi.sqrt(variance(t))
}

/// L2 norm
pub fn norm(t: Tensor) -> Float {
  let data = tensor.to_list(t)
  let sum_sq = list.fold(data, 0.0, fn(acc, x) { acc +. x *. x })
  ffi.sqrt(sum_sq)
}

/// Normalize to unit length
pub fn normalize(t: Tensor) -> Tensor {
  let n = norm(t)
  case n >. 0.0001 {
    True -> scale(t, 1.0 /. n)
    False -> t
  }
}

// =============================================================================
// MATRIX OPERATIONS
// =============================================================================

/// Dot product of two vectors
pub fn dot(a: Tensor, b: Tensor) -> Result(Float, TensorError) {
  case
    tensor.rank(a) == 1
    && tensor.rank(b) == 1
    && tensor.size(a) == tensor.size(b)
  {
    True -> {
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let products = list.map2(a_data, b_data, fn(x, y) { x *. y })
      Ok(list.fold(products, 0.0, fn(acc, x) { acc +. x }))
    }
    False -> Error(error.ShapeMismatch(tensor.shape(a), tensor.shape(b)))
  }
}

/// Matrix-vector multiplication: [m, n] @ [n] -> [m]
pub fn matmul_vec(mat: Tensor, vec: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(mat), tensor.shape(vec) {
    [m, n], [vec_n] if n == vec_n -> {
      let mat_data = tensor.to_list(mat)
      let vec_data = tensor.to_list(vec)
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
      tensor.new(result_data, [m])
    }
    [_m, n], [vec_n] -> Error(error.ShapeMismatch([n], [vec_n]))
    _, _ -> Error(error.DimensionError("Expected matrix and vector"))
  }
}

/// Matrix-matrix multiplication: [m, n] @ [n, p] -> [m, p]
pub fn matmul(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(a), tensor.shape(b) {
    [m, n], [n2, p] if n == n2 -> {
      let result_data =
        list.range(0, m - 1)
        |> list.flat_map(fn(i) {
          list.range(0, p - 1)
          |> list.map(fn(j) {
            list.range(0, n - 1)
            |> list.fold(0.0, fn(acc, k) {
              let a_ik = case tensor.get2d(a, i, k) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
              let b_kj = case tensor.get2d(b, k, j) {
                Ok(v) -> v
                Error(_) -> 0.0
              }
              acc +. a_ik *. b_kj
            })
          })
        })
      tensor.new(result_data, [m, p])
    }
    [_m, n], [n2, _p] -> Error(error.ShapeMismatch([n, -1], [n2, -1]))
    _, _ -> Error(error.DimensionError("Expected two matrices"))
  }
}

/// Matrix transpose
pub fn transpose(t: Tensor) -> Result(Tensor, TensorError) {
  case tensor.shape(t) {
    [m, n] -> {
      let result_data =
        list.range(0, n - 1)
        |> list.flat_map(fn(j) {
          list.range(0, m - 1)
          |> list.filter_map(fn(i) { tensor.get2d(t, i, j) })
        })
      tensor.new(result_data, [n, m])
    }
    _ -> Error(error.DimensionError("Transpose requires 2D tensor"))
  }
}

/// Outer product: [m] @ [n] -> [m, n]
pub fn outer(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  case tensor.rank(a) == 1 && tensor.rank(b) == 1 {
    True -> {
      let m = tensor.size(a)
      let n = tensor.size(b)
      let a_data = tensor.to_list(a)
      let b_data = tensor.to_list(b)
      let result_data =
        list.flat_map(a_data, fn(ai) { list.map(b_data, fn(bj) { ai *. bj }) })
      tensor.new(result_data, [m, n])
    }
    False -> Error(error.DimensionError("Outer product requires two vectors"))
  }
}

// =============================================================================
// UTILITY OPERATIONS
// =============================================================================

/// Clamp values to range
pub fn clamp(t: Tensor, min_val: Float, max_val: Float) -> Tensor {
  map(t, fn(x) { float.min(float.max(x, min_val), max_val) })
}

/// ReLU activation
pub fn relu(t: Tensor) -> Tensor {
  map(t, fn(x) { float.max(x, 0.0) })
}

/// Sigmoid activation
pub fn sigmoid(t: Tensor) -> Tensor {
  map(t, fn(x) { 1.0 /. { 1.0 +. ffi.exp(0.0 -. x) } })
}

/// Tanh activation
pub fn tanh(t: Tensor) -> Tensor {
  map(t, fn(x) { ffi.tanh(x) })
}

/// Softmax (along last axis for 1D)
pub fn softmax(t: Tensor) -> Tensor {
  let data = tensor.to_list(t)
  let max_val = max(t)
  let shifted = list.map(data, fn(x) { ffi.exp(x -. max_val) })
  let sum_exp = list.fold(shifted, 0.0, fn(acc, x) { acc +. x })
  let result = list.map(shifted, fn(x) { x /. sum_exp })
  case tensor.new(result, tensor.shape(t)) {
    Ok(r) -> r
    Error(_) -> t
  }
}

// =============================================================================
// BROADCASTING OPERATIONS
// =============================================================================

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
    False -> Error(error.BroadcastError(a, b))
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

/// Element-wise addition with broadcasting
pub fn add_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(
    tensor.shape(a),
    tensor.shape(b),
  ))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  add(a_bc, b_bc)
}

/// Element-wise multiplication with broadcasting
pub fn mul_broadcast(a: Tensor, b: Tensor) -> Result(Tensor, TensorError) {
  use result_shape <- result.try(broadcast_shape(
    tensor.shape(a),
    tensor.shape(b),
  ))
  use a_bc <- result.try(broadcast_to(a, result_shape))
  use b_bc <- result.try(broadcast_to(b, result_shape))
  mul(a_bc, b_bc)
}

/// Broadcast tensor to target shape
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  let src_shape = tensor.shape(t)
  case can_broadcast(src_shape, target_shape) {
    False -> Error(error.BroadcastError(src_shape, target_shape))
    True -> {
      case src_shape == target_shape {
        True -> Ok(t)
        False -> {
          let data = broadcast_data(t, target_shape)
          tensor.new(data, target_shape)
        }
      }
    }
  }
}

fn broadcast_data(t: Tensor, target_shape: List(Int)) -> List(Float) {
  let target_size = list.fold(target_shape, 1, fn(acc, dim) { acc * dim })
  let src_shape = tensor.shape(t)
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  let data = tensor.to_list(t)

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

fn multi_to_flat(indices: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)
  list.zip(indices, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(idx, stride) = pair
    acc + idx * stride
  })
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

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      lst
      |> list.drop(index)
      |> list.first
  }
}
