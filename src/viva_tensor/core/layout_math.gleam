//// Shared tensor layout and indexing math.
////
//// This module is internal to the package. It centralizes row-major stride,
//// flat-index, and list-index helpers so tensor backends use the same layout
//// contract.

import gleam/list

/// Compute the number of elements represented by a shape.
pub fn size(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, dim) { acc * dim })
}

/// Return `[0, ..., size - 1]`, or an empty list for non-positive sizes.
pub fn indices(size: Int) -> List(Int) {
  case size <= 0 {
    True -> []
    False -> list.range(0, size - 1)
  }
}

/// Compute row-major contiguous strides for a shape.
pub fn compute_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(strides, _) =
    list.fold(reversed, #([], 1), fn(acc, dim) {
      let #(s, running) = acc
      #([running, ..s], running * dim)
    })
  strides
}

/// Convert a flat row-major index into coordinates for a shape.
pub fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(coordinates, _) =
    list.fold(reversed, #([], flat), fn(acc, dim) {
      let #(coords, remaining) = acc
      let coord = remaining % dim
      let next = remaining / dim
      #([coord, ..coords], next)
    })
  coordinates
}

/// Convert coordinates into a flat row-major index for a shape.
pub fn multi_to_flat(coordinates: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)
  list.zip(coordinates, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(coordinate, stride) = pair
    acc + coordinate * stride
  })
}

/// Get a list item by index.
pub fn at(values: List(a), index: Int) -> Result(a, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False ->
      values
      |> list.drop(index)
      |> list.first
  }
}

/// Get an integer by index, defaulting to `0` when absent.
pub fn dim_at(values: List(Int), index: Int) -> Int {
  case at(values, index) {
    Ok(value) -> value
    Error(_) -> 0
  }
}

/// Get a float by index, defaulting to `0.0` when absent.
pub fn value_at(values: List(Float), index: Int) -> Float {
  case at(values, index) {
    Ok(value) -> value
    Error(_) -> 0.0
  }
}

/// Replace a list item by index.
pub fn replace_at(values: List(Int), index: Int, value: Int) -> List(Int) {
  values
  |> list.index_map(fn(item, i) {
    case i == index {
      True -> value
      False -> item
    }
  })
}

/// Compute strides for a broadcast view.
///
/// Expanded dimensions receive stride `0` so repeated values share storage.
pub fn broadcast_strides(
  src_shape: List(Int),
  src_strides: List(Int),
  target_shape: List(Int),
) -> List(Int) {
  let diff = list.length(target_shape) - list.length(src_shape)
  let padded_shape = list.append(list.repeat(1, diff), src_shape)
  let padded_strides = list.append(list.repeat(0, diff), src_strides)

  list.zip(list.zip(padded_shape, target_shape), padded_strides)
  |> list.map(fn(item) {
    let #(#(src_dim, target_dim), stride) = item
    case src_dim == target_dim {
      True -> stride
      False -> 0
    }
  })
}
