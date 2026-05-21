//// Core Tensor module.
////
//// The tensor type is opaque so callers cannot construct invalid values where
//// the shape and data length disagree.
////
//// The strided representation follows the same storage model used by ndarray
//// libraries: views can share storage and reinterpret shape, strides, and
//// offsets without copying element data.
////
//// Erlang's `array` module is tree-backed rather than contiguous memory, so
//// view access has different performance characteristics than native buffers.
////
//// ```gleam
//// let a = tensor.zeros([2, 3])
//// let b = tensor.ones([2, 3])
//// use c <- result.try(tensor.add(a, b))
//// ```

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam_community/maths
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/ffi.{type ErlangArray, type NativeTensorRef}
import viva_tensor/core/layout_math

// --- Tensor Type -----------------------------------------------------------

/// The tensor itself. Opaque so nobody can break invariants.
///
/// Two flavors:
/// - Dense: backed by List, simple but O(n) access
/// - Strided: backed by Erlang :array, O(1) access + zero-copy views
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
  /// Native tensor - data lives in contiguous C memory via NIF resource.
  /// Zero-copy ops: no list conversion between operations.
  /// GC calls destructor to free native memory automatically.
  Native(ref: NativeTensorRef, shape: List(Int))
}

// --- Constructors -----------------------------------------------------------

/// Create tensor with validation. This is the "safe" constructor.
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

/// Identity matrix. The multiplicative identity of matrix algebra.
/// I*A = A*I = A. One of the few things in linear algebra that's intuitive.
pub fn eye(n: Int) -> Tensor {
  // Sparse would be O(n) space vs O(n²), but adds complexity. YAGNI for now.
  let data =
    range_int(0, n - 1)
    |> list.flat_map(fn(i) {
      range_int(0, n - 1)
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
      maths.linear_space(start, end, num, True)
      |> result.unwrap([])
      |> from_list()
    }
  }
}

// --- Random Constructors ----------------------------------------------------
// Uses Erlang's :rand (xorshift116+ algorithm). Fast, decent statistical
// properties, definitely NOT cryptographically secure. Perfect for ML.
// If you need crypto, you're in the wrong library.

/// Uniform random in [0, 1). Seeds from system entropy on first call.
pub fn random_uniform(shape: List(Int)) -> Tensor {
  let size = compute_size(shape)
  let data =
    range_int(1, size)
    |> list.map(fn(_) { ffi.random_uniform() })
  Dense(data: data, shape: shape)
}

/// Normal distribution via Box-Muller transform (1958).
/// Could use Ziggurat for ~3x speedup but Box-Muller is elegant and
/// "premature optimization is the root of all evil" - Knuth
pub fn random_normal(
  shape shape: List(Int),
  mean mean: Float,
  std std: Float,
) -> Tensor {
  let size = compute_size(shape)
  let data =
    range_int(1, size)
    |> list.map(fn(_) {
      let u1 = float.max(ffi.random_uniform(), 0.0001)
      let u2 = ffi.random_uniform()
      let z = ffi.sqrt(-2.0 *. ffi.log(u1)) *. ffi.cos(2.0 *. ffi.pi *. u2)
      mean +. z *. std
    })
  Dense(data: data, shape: shape)
}

/// Xavier/Glorot init (2010 paper: "Understanding the difficulty of training deep FFNs")
/// The limit = sqrt(6 / (fan_in + fan_out)) keeps variance stable across layers.
/// Use this for tanh/sigmoid. For ReLU, use he_init instead.
pub fn xavier_init(fan_in fan_in: Int, fan_out fan_out: Int) -> Tensor {
  // Derived from Var(W) = 2/(fan_in + fan_out), uniform bounds = sqrt(3 * Var)
  let limit = ffi.sqrt(6.0 /. int.to_float(fan_in + fan_out))
  let data =
    range_int(1, fan_in * fan_out)
    |> list.map(fn(_) {
      let r = ffi.random_uniform()
      r *. 2.0 *. limit -. limit
    })
  Dense(data: data, shape: [fan_out, fan_in])
}

/// He init (2015 paper: "Delving Deep into Rectifiers")
/// std = sqrt(2/fan_in) accounts for ReLU killing half the activations.
/// The "2" is not arbitrary - it comes from E[ReLU(x)²] = Var(x)/2 for x~N(0,σ²)
pub fn he_init(fan_in fan_in: Int, fan_out fan_out: Int) -> Tensor {
  let std = ffi.sqrt(2.0 /. int.to_float(fan_in))
  random_normal(shape: [fan_out, fan_in], mean: 0.0, std: std)
}

// --- Accessors --------------------------------------------------------------

/// Get tensor shape
pub fn shape(t: Tensor) -> List(Int) {
  case t {
    Dense(_, s) -> s
    Strided(_, s, _, _) -> s
    Native(_, s) -> s
  }
}

/// Materialize tensor data as a list, preserving native download failures.
pub fn try_to_list(t: Tensor) -> Result(List(Float), TensorError) {
  case t {
    Dense(data, _) -> Ok(data)
    Native(ref, _) ->
      case ffi.nt_to_list(ref) {
        Ok(data) -> Ok(data)
        Error(reason) ->
          Error(error.DimensionError(
            "Native tensor materialization failed: " <> reason,
          ))
      }
    Strided(storage, shp, strides, offset) -> {
      let total_size = compute_size(shp)
      let data =
        range_int(0, total_size - 1)
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
      Ok(data)
    }
  }
}

/// Get tensor data as list.
///
/// Prefer `try_to_list` in fallible code paths so native materialization errors
/// are not hidden.
pub fn to_list(t: Tensor) -> List(Float) {
  try_to_list(t)
  |> result.unwrap([])
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

// --- Broadcasting ------------------------------------------------------------

/// Check if two shapes can broadcast together using NumPy-style rules.
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

/// Compute the output shape for broadcasting two shapes.
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

/// Broadcast a tensor to a target shape.
///
/// Dense and strided tensors return zero-stride views. Native tensors use the
/// NIF broadcast view when available and fall back to dense materialization.
pub fn broadcast_to(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  let src_shape = shape(t)

  case can_broadcast(src_shape, target_shape) {
    False -> Error(error.BroadcastError(src_shape, target_shape))
    True -> {
      case src_shape == target_shape {
        True -> Ok(t)
        False ->
          case t {
            Dense(data, _) -> {
              let storage = ffi.list_to_array(data)
              let strides =
                layout_math.broadcast_strides(
                  src_shape,
                  compute_strides(src_shape),
                  target_shape,
                )

              Ok(Strided(
                storage: storage,
                shape: target_shape,
                strides: strides,
                offset: 0,
              ))
            }

            Strided(storage, _, strides, offset) -> {
              let view_strides =
                layout_math.broadcast_strides(src_shape, strides, target_shape)

              Ok(Strided(
                storage: storage,
                shape: target_shape,
                strides: view_strides,
                offset: offset,
              ))
            }

            Native(ref, _) ->
              case ffi.nt_broadcast_to(ref, target_shape) {
                Ok(view_ref) -> Ok(Native(ref: view_ref, shape: target_shape))
                Error(_) -> {
                  use data <- result.try(broadcast_data(t, target_shape))
                  new(data, target_shape)
                }
              }
          }
      }
    }
  }
}

// --- Element Access ---------------------------------------------------------

/// Get element by flat index. For strided tensors, computes the real offset.
pub fn get(t: Tensor, index: Int) -> Result(Float, TensorError) {
  case t {
    Dense(data, _) ->
      list_at_float(data, index)
      |> result.map_error(fn(_) { error.IndexOutOfBounds(index, size(t)) })

    Native(ref, _) ->
      case try_to_list(Native(ref: ref, shape: shape(t))) {
        Ok(data) ->
          list_at_float(data, index)
          |> result.map_error(fn(_) { error.IndexOutOfBounds(index, size(t)) })
        Error(e) -> Error(e)
      }

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
    other -> Error(error.RankMismatch("get2d", 2, other))
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
    other -> Error(error.RankMismatch("get_row", 2, other))
  }
}

/// Get matrix column as vector
pub fn get_col(t: Tensor, col_idx: Int) -> Result(Tensor, TensorError) {
  case shape(t) {
    [num_rows, num_cols] -> {
      case col_idx >= 0 && col_idx < num_cols {
        True -> {
          let col_data =
            range_int(0, num_rows - 1)
            |> list.filter_map(fn(row) { get2d(t, row, col_idx) })
          Ok(from_list(col_data))
        }
        False -> Error(error.IndexOutOfBounds(col_idx, num_cols))
      }
    }
    other -> Error(error.RankMismatch("get_col", 2, other))
  }
}

// --- Strided Operations (Zero-copy magic) -----------------------------------
// The key insight: a 2D array in row-major order has strides [cols, 1].
// Transpose it? Just swap to [1, cols]. Same memory, different interpretation.
// PyTorch calls this a "view". NumPy calls it "non-contiguous". I call it genius.
//
// Warning: non-contiguous tensors are slower for sequential access (cache misses).
// Call to_contiguous() before heavy computation if you need speed.

/// Convert to strided (backed by Erlang :array for O(1) random access)
pub fn to_strided(t: Tensor) -> Tensor {
  case t {
    Strided(_, _, _, _) -> t
    Native(_, _) -> t
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
    Native(_, _) -> {
      let data = to_list(t)
      Dense(data: data, shape: shape(t))
    }
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
    Native(_, _) -> True
    Strided(_, shp, strides, _) -> {
      let expected_strides = compute_strides(shp)
      strides == expected_strides
    }
  }
}

/// Zero-copy transpose (just swap strides and shape)
pub fn transpose_strided(t: Tensor) -> Result(Tensor, TensorError) {
  case t {
    Native(ref, shp) ->
      case shp {
        [_m, _n] ->
          case ffi.nt_transpose(ref) {
            Ok(ref_t) -> Ok(Native(ref: ref_t, shape: list.reverse(shp)))
            Error(_) -> {
              let dense = to_dense(t)
              transpose_strided(dense)
            }
          }
        _ -> Error(error.DimensionError("Transpose requires 2D tensor"))
      }
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

// --- Internal Helpers -------------------------------------------------------
// These don't need to be pretty, just correct.

fn compute_size(shape: List(Int)) -> Int {
  layout_math.size(shape)
}

fn compute_strides(shape: List(Int)) -> List(Int) {
  layout_math.compute_strides(shape)
}

fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  layout_math.flat_to_multi(flat, shape)
}

fn list_at(lst: List(a), index: Int) -> Result(a, Nil) {
  layout_math.at(lst, index)
}

fn list_at_float(lst: List(Float), index: Int) -> Result(Float, Nil) {
  list_at(lst, index)
}

fn broadcast_data(
  t: Tensor,
  target_shape: List(Int),
) -> Result(List(Float), TensorError) {
  let target_size = compute_size(target_shape)
  let src_shape = shape(t)
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  use data <- result.try(try_to_list(t))

  let diff = target_rank - src_rank
  let padded_shape = list.append(list.repeat(1, diff), src_shape)

  layout_math.indices(target_size)
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

    let src_flat = layout_math.multi_to_flat(src_indices, src_shape)
    use value <- result.try(
      list_at_float(data, src_flat)
      |> result.map_error(fn(_) {
        error.IndexOutOfBounds(src_flat, list.length(data))
      }),
    )
    Ok([value, ..values])
  })
  |> result.map(list.reverse)
}

// --- Native Tensor API -------------------------------------------------------
// Direct access to NIF resource tensors. Data stays in contiguous C memory.
// Zero list conversion between ops. This is the fast path.

/// Check if tensor is backed by native C memory
pub fn is_native(t: Tensor) -> Bool {
  case t {
    Native(_, _) -> True
    _ -> False
  }
}

/// Extract native ref (for passing to NIF resource ops)
pub fn native_ref(t: Tensor) -> Result(NativeTensorRef, Nil) {
  case t {
    Native(ref, _) -> Ok(ref)
    _ -> Error(Nil)
  }
}

/// Wrap a NIF resource ref as a Tensor
pub fn from_native_ref(ref: NativeTensorRef, shape: List(Int)) -> Tensor {
  Native(ref: ref, shape: shape)
}

/// Create native tensor of zeros (data in C memory)
pub fn native_zeros(shape: List(Int)) -> Result(Tensor, TensorError) {
  case ffi.nt_zeros(shape) {
    Ok(ref) -> Ok(Native(ref: ref, shape: shape))
    Error("nif_not_loaded") -> Error(error.NifNotLoaded("native_zeros"))
    Error(_) -> Error(error.InvalidShape("NIF resource allocation failed"))
  }
}

/// Create native tensor of ones
pub fn native_ones(shape: List(Int)) -> Result(Tensor, TensorError) {
  case ffi.nt_ones(shape) {
    Ok(ref) -> Ok(Native(ref: ref, shape: shape))
    Error("nif_not_loaded") -> Error(error.NifNotLoaded("native_ones"))
    Error(_) -> Error(error.InvalidShape("NIF resource allocation failed"))
  }
}

/// Create native tensor filled with value
pub fn native_fill(
  shape: List(Int),
  value: Float,
) -> Result(Tensor, TensorError) {
  case ffi.nt_fill(shape, value) {
    Ok(ref) -> Ok(Native(ref: ref, shape: shape))
    Error("nif_not_loaded") -> Error(error.NifNotLoaded("native_fill"))
    Error(_) -> Error(error.InvalidShape("NIF resource allocation failed"))
  }
}

/// Create native tensor from list data
pub fn native_from_list(
  data: List(Float),
  shape: List(Int),
) -> Result(Tensor, TensorError) {
  case ffi.nt_from_list(data, shape) {
    Ok(ref) -> Ok(Native(ref: ref, shape: shape))
    Error("nif_not_loaded") -> Error(error.NifNotLoaded("native_from_list"))
    Error(_) -> {
      let expected = compute_size(shape)
      let actual = list.length(data)
      case expected == actual {
        True -> Error(error.InvalidShape("NIF resource allocation failed"))
        False ->
          Error(error.InvalidShape(
            "Data size "
            <> int.to_string(actual)
            <> " doesn't match shape "
            <> error.shape_to_string(shape),
          ))
      }
    }
  }
}

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}
