//// Broadcasting helpers for tensor shape and dense fallback paths.
////
//// The public tensor facade owns storage-specific decisions. This module owns
//// pure shape compatibility, zero-stride planning, and dense list expansion.

import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{
  type TensorError, BroadcastError, IndexOutOfBounds,
}
import viva_tensor/core/layout_math

/// Check if two shapes can be broadcast together.
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

/// Compute the shape produced by broadcasting two shapes.
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

/// Compute zero-stride view strides for broadcasting one shape into another.
pub fn broadcast_strides(
  src_shape: List(Int),
  src_strides: List(Int),
  target_shape: List(Int),
) -> List(Int) {
  layout_math.broadcast_strides(src_shape, src_strides, target_shape)
}

/// Materialize broadcasted dense data from a source shape into a target shape.
pub fn broadcast_data_values(
  data: List(Float),
  src_shape: List(Int),
  target_shape: List(Int),
) -> Result(List(Float), TensorError) {
  let target_size = list.fold(target_shape, 1, fn(acc, dim) { acc * dim })
  let src_rank = list.length(src_shape)
  let target_rank = list.length(target_shape)
  let diff = target_rank - src_rank
  let padded_shape = list.append(list.repeat(1, diff), src_shape)

  list.range(0, target_size - 1)
  |> list.fold(Ok([]), fn(acc, flat_idx) {
    use values <- result.try(acc)
    let target_indices = layout_math.flat_to_multi(flat_idx, target_shape)

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
      layout_math.at(data, src_flat)
      |> result.map_error(fn(_) {
        IndexOutOfBounds(src_flat, list.length(data))
      }),
    )
    Ok([value, ..values])
  })
  |> result.map(list.reverse)
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
