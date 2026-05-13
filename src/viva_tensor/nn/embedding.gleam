//// Embedding and positional encoding layers.
////
//// Pure Gleam, no NIF, no autograd integration. These layers are values:
//// they hold a weight `Tensor` (Dense `Tensor(data, shape)`) and expose a
//// `forward` function that returns a new tensor. Training-time gradients
//// are handled by a separate autograd pass when wired in later.
////
//// References:
//// - Bengio et al. (2003). "A Neural Probabilistic Language Model." The
////   original word embedding paper.
//// - Vaswani et al. (2017). "Attention Is All You Need." Sinusoidal
////   positional encoding.
//// - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
////   Embedding." Rotary positional embeddings (RoPE).

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam_community/maths
import viva_tensor/core/error.{
  type TensorError, DimensionError, IndexOutOfBounds, InvalidShape,
} as tensor_error
import viva_tensor/core/ffi
import viva_tensor/tensor.{type Tensor, Tensor}

// -------------------------------------------------------------------------
// Embedding - learnable lookup table
// -------------------------------------------------------------------------
//
// An embedding maps a finite set of integer ids (e.g. tokens) to a dense
// vector in R^d. The weight matrix is `[num_embeddings, embedding_dim]`.
// Lookup is a gather along axis 0: row i becomes the embedding of id i.

/// Learnable embedding table.
///
/// Stores a weight matrix of shape `[num_embeddings, embedding_dim]`.
/// Each row is the dense vector associated with one integer id.
pub type Embedding {
  Embedding(num_embeddings: Int, embedding_dim: Int, weight: Tensor)
}

/// Initialize an embedding table with zero weights.
///
/// ## Example
///
/// ```gleam
/// let layer = embedding_init(num_embeddings: 10, embedding_dim: 4)
/// // layer.weight has shape [10, 4], all zeros
/// ```
pub fn embedding_init(
  num_embeddings num_embeddings: Int,
  embedding_dim embedding_dim: Int,
) -> Embedding {
  let weight = tensor.zeros([num_embeddings, embedding_dim])
  Embedding(num_embeddings, embedding_dim, weight)
}

/// Initialize an embedding table with uniform random weights in
/// `[-1/sqrt(dim), 1/sqrt(dim)]`.
///
/// This matches PyTorch's default `nn.Embedding` initializer (uniform
/// scaled by the inverse square root of the feature dimension). It keeps
/// the variance of the lookup output stable for any reasonable `dim`.
///
/// ## Example
///
/// ```gleam
/// let layer = embedding_init_uniform(num_embeddings: 100, embedding_dim: 16)
/// // values fall in [-0.25, 0.25] since 1 / sqrt(16) = 0.25
/// ```
pub fn embedding_init_uniform(
  num_embeddings num_embeddings: Int,
  embedding_dim embedding_dim: Int,
) -> Embedding {
  let dim_f = int.to_float(embedding_dim)
  let limit = case dim_f >. 0.0 {
    True -> 1.0 /. ffi.sqrt(dim_f)
    False -> 0.0
  }
  let size = num_embeddings * embedding_dim
  let data = case size <= 0 {
    True -> []
    False ->
      list.range(1, size)
      |> list.map(fn(_) {
        let r = ffi.random_uniform()
        r *. 2.0 *. limit -. limit
      })
  }
  let weight = Tensor(data: data, shape: [num_embeddings, embedding_dim])
  Embedding(num_embeddings, embedding_dim, weight)
}

/// Forward pass: for each integer index in `indices`, look up the
/// corresponding row of `weight`. Negative indices wrap (NumPy convention).
///
/// `indices` is a 1D tensor of integer-valued floats; output shape is
/// `[len(indices), embedding_dim]`.
///
/// Errors:
/// - `DimensionError` if `indices` is not 1D.
/// - `IndexOutOfBounds` if any index >= num_embeddings (after wrap).
///
/// ## Example
///
/// ```gleam
/// let layer = embedding_init(num_embeddings: 3, embedding_dim: 2)
/// let assert Ok(indices) = tensor.matrix(1, 2, [0.0, 2.0])
/// // ...use the 1D form instead:
/// let indices = tensor.from_list([0.0, 2.0])
/// let assert Ok(out) = embedding_forward(layer, indices)
/// // out.shape == [2, 2]
/// ```
pub fn embedding_forward(
  layer: Embedding,
  indices: Tensor,
) -> Result(Tensor, TensorError) {
  case tensor.rank(indices) == 1 {
    False ->
      Error(DimensionError(
        "embedding_forward: indices must be 1D, got rank "
        <> int.to_string(tensor.rank(indices)),
      ))
    True -> {
      use idx_floats <- result.try(tensor.try_to_list(indices))
      use idx_ints <- result.try(normalize_indices(
        idx_floats,
        layer.num_embeddings,
      ))
      use weight_data <- result.try(tensor.try_to_list(layer.weight))
      use rows <- result.try(gather_rows(
        weight_data,
        idx_ints,
        layer.embedding_dim,
      ))
      Ok(
        Tensor(data: rows, shape: [list.length(idx_ints), layer.embedding_dim]),
      )
    }
  }
}

// -------------------------------------------------------------------------
// Sinusoidal positional encoding (no parameters)
// -------------------------------------------------------------------------

/// Generate a positional encoding tensor of shape `[max_len, embedding_dim]`
/// using the canonical sinusoidal formula from "Attention Is All You Need":
///
///     PE[pos, 2i]   = sin(pos / 10000^(2i/dim))
///     PE[pos, 2i+1] = cos(pos / 10000^(2i/dim))
///
/// `embedding_dim` must be even, otherwise returns
/// `InvalidShape("sinusoidal_encoding: embedding_dim must be even")`.
///
/// ## Example
///
/// ```gleam
/// let assert Ok(pe) = sinusoidal_encoding(max_len: 2, embedding_dim: 4)
/// // pe[0, 0] = sin(0)   = 0.0
/// // pe[0, 1] = cos(0)   = 1.0
/// // pe[1, 0] = sin(1.0) ≈ 0.8415
/// ```
pub fn sinusoidal_encoding(
  max_len max_len: Int,
  embedding_dim embedding_dim: Int,
) -> Result(Tensor, TensorError) {
  case max_len < 0 {
    True ->
      Error(InvalidShape("sinusoidal_encoding: max_len must be non-negative"))
    False ->
      case embedding_dim <= 0 {
        True ->
          Error(InvalidShape("sinusoidal_encoding: embedding_dim must be > 0"))
        False ->
          case embedding_dim % 2 == 0 {
            False ->
              Error(InvalidShape(
                "sinusoidal_encoding: embedding_dim must be even",
              ))
            True -> {
              let dim_f = int.to_float(embedding_dim)
              let positions = case max_len <= 0 {
                True -> []
                False -> list.range(0, max_len - 1)
              }
              let pair_indices = case embedding_dim / 2 <= 0 {
                True -> []
                False -> list.range(0, embedding_dim / 2 - 1)
              }
              let data =
                positions
                |> list.flat_map(fn(pos) {
                  let pos_f = int.to_float(pos)
                  pair_indices
                  |> list.flat_map(fn(i) {
                    let two_i = int.to_float(2 * i)
                    let exponent = two_i /. dim_f
                    let denom = pow_safe(10_000.0, exponent)
                    let angle = pos_f /. denom
                    [maths.sin(angle), maths.cos(angle)]
                  })
                })
              Ok(Tensor(data: data, shape: [max_len, embedding_dim]))
            }
          }
      }
  }
}

// -------------------------------------------------------------------------
// Learned positional encoding
// -------------------------------------------------------------------------

/// Learnable positional encoding table.
///
/// A thin wrapper around an `Embedding` of shape `[max_len, embedding_dim]`.
/// Lookups use positions `0..len-1` instead of arbitrary token ids.
pub type LearnedPositionalEncoding {
  LearnedPositionalEncoding(max_len: Int, embedding_dim: Int, weight: Tensor)
}

/// Initialize a learned positional encoding table with uniform random
/// weights in `[-1/sqrt(embedding_dim), 1/sqrt(embedding_dim)]`.
///
/// ## Example
///
/// ```gleam
/// let pe = learned_positional_init(max_len: 512, embedding_dim: 64)
/// // pe.weight has shape [512, 64]
/// ```
pub fn learned_positional_init(
  max_len max_len: Int,
  embedding_dim embedding_dim: Int,
) -> LearnedPositionalEncoding {
  let base = embedding_init_uniform(max_len, embedding_dim)
  LearnedPositionalEncoding(max_len, embedding_dim, base.weight)
}

/// Look up positions `0..len-1` inclusive. Returns shape `[len, embedding_dim]`.
///
/// Errors with `IndexOutOfBounds` if `len > max_len` or `len < 0`.
///
/// ## Example
///
/// ```gleam
/// let pe = learned_positional_init(max_len: 32, embedding_dim: 8)
/// let assert Ok(out) = learned_positional_forward(pe, len: 5)
/// // out.shape == [5, 8]
/// ```
pub fn learned_positional_forward(
  layer: LearnedPositionalEncoding,
  len: Int,
) -> Result(Tensor, TensorError) {
  case len < 0 {
    True -> Error(IndexOutOfBounds(len, layer.max_len))
    False ->
      case len > layer.max_len {
        True -> Error(IndexOutOfBounds(len, layer.max_len))
        False -> {
          use weight_data <- result.try(tensor.try_to_list(layer.weight))
          let indices = case len {
            0 -> []
            _ -> list.range(0, len - 1)
          }
          use rows <- result.try(gather_rows(
            weight_data,
            indices,
            layer.embedding_dim,
          ))
          Ok(Tensor(data: rows, shape: [len, layer.embedding_dim]))
        }
      }
  }
}

// -------------------------------------------------------------------------
// RoPE - Rotary Positional Embedding
// -------------------------------------------------------------------------
//
// Layout: interleaved pairs. Feature dim is partitioned into adjacent
// `(x_even, x_odd)` pairs. Each pair is rotated by an angle that depends
// on the position and the pair index. This matches the original RoFormer
// description ("rotate the embedding two coordinates at a time").

/// Apply Rotary Positional Embedding to an input of shape `[seq_len, dim]`.
///
/// `dim` must be even. Each adjacent pair `(x_even, x_odd)` at position
/// `pos` and pair index `i` is rotated by `theta = pos * 1 / base^(2i/dim)`:
///
///     (x', y') = (x*cos(theta) - y*sin(theta),
///                 x*sin(theta) + y*cos(theta))
///
/// `base` is the rotation base; pass `10000.0` to match Llama / RoFormer.
///
/// Errors:
/// - `DimensionError` if `input` is not 2D.
/// - `InvalidShape("rope: dim must be even")` if `dim` is odd.
///
/// ## Example
///
/// ```gleam
/// let assert Ok(x) = tensor.matrix(2, 2, [1.0, 0.0, 1.0, 0.0])
/// let assert Ok(out) = rope(x, 10_000.0)
/// // out.shape == [2, 2]
/// // pos 0: angle 0, rotation is identity -> (1.0, 0.0)
/// // pos 1: angle 1, -> (cos 1, sin 1)
/// ```
pub fn rope(input: Tensor, base: Float) -> Result(Tensor, TensorError) {
  case tensor.shape(input) {
    [seq_len, dim] ->
      case dim % 2 == 0 {
        False -> Error(InvalidShape("rope: dim must be even"))
        True ->
          case base <=. 0.0 {
            True -> Error(InvalidShape("rope: base must be > 0"))
            False -> {
              use data <- result.try(tensor.try_to_list(input))
              let dim_f = int.to_float(dim)
              let positions = case seq_len <= 0 {
                True -> []
                False -> list.range(0, seq_len - 1)
              }
              let rotated =
                positions
                |> list.flat_map(fn(pos) {
                  let pos_f = int.to_float(pos)
                  // Slice this position's row out of the flat buffer.
                  let row = take_row(data, pos, dim)
                  rotate_row(row, pos_f, dim_f, dim / 2, base)
                })
              Ok(Tensor(data: rotated, shape: [seq_len, dim]))
            }
          }
      }
    other ->
      Error(DimensionError(
        "rope: expected 2D tensor, got shape "
        <> tensor_error.shape_to_string(other),
      ))
  }
}

// -------------------------------------------------------------------------
// Internals
// -------------------------------------------------------------------------

/// Wrap negative indices NumPy-style and bounds-check.
fn normalize_indices(
  raw: List(Float),
  num_embeddings: Int,
) -> Result(List(Int), TensorError) {
  list.try_fold(raw, [], fn(acc, value) {
    let idx = float.truncate(value)
    let wrapped = case idx < 0 {
      True -> idx + num_embeddings
      False -> idx
    }
    case wrapped < 0 || wrapped >= num_embeddings {
      True -> Error(IndexOutOfBounds(idx, num_embeddings))
      False -> Ok([wrapped, ..acc])
    }
  })
  |> result.map(list.reverse)
}

/// Pull contiguous `dim`-sized rows from a flat `[N*dim]` buffer.
fn gather_rows(
  data: List(Float),
  indices: List(Int),
  dim: Int,
) -> Result(List(Float), TensorError) {
  list.try_fold(indices, [], fn(acc, idx) {
    let row = take_row(data, idx, dim)
    case list.length(row) == dim {
      True -> Ok([row, ..acc])
      False -> Error(IndexOutOfBounds(idx, list.length(data) / int.max(dim, 1)))
    }
  })
  |> result.map(fn(rows) {
    rows
    |> list.reverse
    |> list.flatten
  })
}

/// Drop `row * dim` elements then take `dim` elements.
fn take_row(data: List(Float), row: Int, dim: Int) -> List(Float) {
  data
  |> list.drop(row * dim)
  |> list.take(dim)
}

/// Rotate one row by interleaved pairs.
fn rotate_row(
  row: List(Float),
  pos_f: Float,
  dim_f: Float,
  num_pairs: Int,
  base: Float,
) -> List(Float) {
  rotate_row_loop(row, pos_f, dim_f, 0, num_pairs, base, [])
  |> list.reverse
}

fn rotate_row_loop(
  remaining: List(Float),
  pos_f: Float,
  dim_f: Float,
  pair_idx: Int,
  num_pairs: Int,
  base: Float,
  acc: List(Float),
) -> List(Float) {
  case pair_idx >= num_pairs, remaining {
    True, _ -> acc
    _, [] -> acc
    _, [_] -> acc
    _, [x, y, ..rest] -> {
      let two_i = int.to_float(2 * pair_idx)
      let theta = 1.0 /. pow_safe(base, two_i /. dim_f)
      let angle = pos_f *. theta
      let c = maths.cos(angle)
      let s = maths.sin(angle)
      let x_new = x *. c -. y *. s
      let y_new = x *. s +. y *. c
      rotate_row_loop(rest, pos_f, dim_f, pair_idx + 1, num_pairs, base, [
        y_new,
        x_new,
        ..acc
      ])
    }
  }
}

/// `float.power` returns `Result`; for our exponents (real base > 0,
/// real exponent) it never fails, but we still unwrap defensively.
fn pow_safe(base: Float, exponent: Float) -> Float {
  case float.power(base, exponent) {
    Ok(v) -> v
    Error(_) -> 1.0
  }
}
