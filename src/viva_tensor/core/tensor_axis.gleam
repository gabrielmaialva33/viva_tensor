//// Axis-oriented tensor helpers.
////
//// This module keeps pure list/shape machinery out of the compatibility
//// tensor facade. It does not construct public tensor storage variants.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import gleam_community/maths
import viva_tensor/core/error.{
  type TensorError, DimensionError, IndexOutOfBounds,
}
import viva_tensor/core/ffi
import viva_tensor/core/layout_math

pub fn axis_size(shape: List(Int), axis_idx: Int) -> Result(Int, TensorError) {
  layout_math.at(shape, axis_idx)
  |> result.map_error(fn(_) { DimensionError("Invalid axis index") })
}

pub fn reduced_shape(
  shape: List(Int),
  axis_idx: Int,
  keepdims: Bool,
) -> List(Int) {
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

pub fn reduce_sum_axis_data(
  data: List(Float),
  input_shape: List(Int),
  output_shape: List(Int),
  axis_idx: Int,
  axis_size: Int,
) -> Result(List(Float), TensorError) {
  case axis_size <= 0 {
    True -> {
      let output_size = list.fold(output_shape, 1, fn(acc, dim) { acc * dim })
      Ok(list.repeat(0.0, output_size))
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
      |> result.map(list.reverse)
    }
  }
}

pub fn softmax_axis_data(
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

pub fn axis_transform_data(
  data: List(Float),
  total_size: Int,
  axis_size: Int,
  inner_size: Int,
  transform: fn(List(Float)) -> List(Float),
) -> Result(List(Float), TensorError) {
  let group_width = axis_size * inner_size

  case group_width <= 0 {
    True -> Ok([])
    False -> {
      let outer_size = total_size / group_width
      list.range(0, outer_size - 1)
      |> list.fold(Ok([]), fn(acc, outer) {
        use values <- result.try(acc)
        use chunk <- result.try(axis_transform_outer(
          data,
          outer,
          axis_size,
          inner_size,
          transform,
        ))
        Ok(list.append(values, chunk))
      })
    }
  }
}

pub fn reduce_axis_data(
  data: List(Float),
  input_shape: List(Int),
  output_shape: List(Int),
  axis_idx: Int,
  axis_size: Int,
  reducer: fn(List(Float)) -> Result(Float, TensorError),
) -> Result(List(Float), TensorError) {
  let output_size = list.fold(output_shape, 1, fn(acc, d) { acc * d })

  list.range(0, output_size - 1)
  |> list.fold(Ok([]), fn(acc, out_idx) {
    use values <- result.try(acc)
    use value <- result.try(reduce_axis_output(
      data,
      input_shape,
      output_shape,
      out_idx,
      axis_idx,
      axis_size,
      reducer,
    ))
    Ok([value, ..values])
  })
  |> result.map(list.reverse)
}

pub fn max_list(values: List(Float)) -> Result(Float, TensorError) {
  case values {
    [] -> Error(DimensionError("Cannot compute max of an empty tensor"))
    [first, ..rest] ->
      Ok(list.fold(rest, first, fn(acc, x) { float.max(acc, x) }))
  }
}

pub fn min_list(values: List(Float)) -> Result(Float, TensorError) {
  case values {
    [] -> Error(DimensionError("Cannot compute min of an empty tensor"))
    [first, ..rest] ->
      Ok(list.fold(rest, first, fn(acc, x) { float.min(acc, x) }))
  }
}

pub fn variance_list(values: List(Float)) -> Result(Float, TensorError) {
  case maths.variance(values, 0) {
    Ok(value) -> Ok(value)
    Error(_) ->
      Error(DimensionError("Cannot compute variance of an empty tensor"))
  }
}

pub fn std_list(values: List(Float)) -> Result(Float, TensorError) {
  case maths.standard_deviation(values, 0) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(DimensionError("Cannot compute std of an empty tensor"))
  }
}

pub fn argmax_list(values: List(Float)) -> Result(Float, TensorError) {
  case values {
    [] -> Error(DimensionError("Cannot compute argmax of an empty tensor"))
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, value) {
          let #(best_idx, best_value, current_idx) = acc
          case value >. best_value {
            True -> #(current_idx, value, current_idx + 1)
            False -> #(best_idx, best_value, current_idx + 1)
          }
        })

      Ok(int.to_float(idx))
    }
  }
}

pub fn argmin_list(values: List(Float)) -> Result(Float, TensorError) {
  case values {
    [] -> Error(DimensionError("Cannot compute argmin of an empty tensor"))
    [first, ..rest] -> {
      let #(idx, _, _) =
        list.fold(rest, #(0, first, 1), fn(acc, value) {
          let #(best_idx, best_value, current_idx) = acc
          case value <. best_value {
            True -> #(current_idx, value, current_idx + 1)
            False -> #(best_idx, best_value, current_idx + 1)
          }
        })

      Ok(int.to_float(idx))
    }
  }
}

pub fn any_list(values: List(Float)) -> Result(Float, TensorError) {
  Ok(case list.any(values, fn(x) { x != 0.0 }) {
    True -> 1.0
    False -> 0.0
  })
}

pub fn all_list(values: List(Float)) -> Result(Float, TensorError) {
  Ok(case list.all(values, fn(x) { x != 0.0 }) {
    True -> 1.0
    False -> 0.0
  })
}

pub fn count_nonzero_list(values: List(Float)) -> Result(Float, TensorError) {
  Ok(
    values
    |> list.fold(0, fn(count, x) {
      case x != 0.0 {
        True -> count + 1
        False -> count
      }
    })
    |> int.to_float,
  )
}

pub fn remove_at_index(lst: List(a), idx: Int) -> List(a) {
  lst
  |> list.index_map(fn(item, i) { #(item, i) })
  |> list.filter(fn(pair) { pair.1 != idx })
  |> list.map(fn(pair) { pair.0 })
}

fn sum_axis_output(
  data: List(Float),
  input_shape: List(Int),
  output_shape: List(Int),
  out_idx: Int,
  axis_idx: Int,
  axis_size: Int,
) -> Result(Float, TensorError) {
  let output_coords = layout_math.flat_to_multi(out_idx, output_shape)
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

    let input_idx = layout_math.multi_to_flat(input_coords, input_shape)
    use value <- result.try(
      layout_math.at(data, input_idx)
      |> result.map_error(fn(_) {
        IndexOutOfBounds(input_idx, list.length(data))
      }),
    )
    Ok(total +. value)
  })
}

fn axis_transform_outer(
  data: List(Float),
  outer: Int,
  axis_size: Int,
  inner_size: Int,
  transform: fn(List(Float)) -> List(Float),
) -> Result(List(Float), TensorError) {
  use groups <- result.try(
    list.range(0, inner_size - 1)
    |> list.fold(Ok([]), fn(acc, inner) {
      use values <- result.try(acc)
      use axis_values <- result.try(axis_values(
        data,
        outer,
        inner,
        axis_size,
        inner_size,
      ))
      Ok([transform(axis_values), ..values])
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
          layout_math.at(group, axis_pos)
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

fn axis_values(
  data: List(Float),
  outer: Int,
  inner: Int,
  axis_size: Int,
  inner_size: Int,
) -> Result(List(Float), TensorError) {
  list.range(0, axis_size - 1)
  |> list.fold(Ok([]), fn(acc, axis_pos) {
    use values <- result.try(acc)
    let index = outer * axis_size * inner_size + inner + axis_pos * inner_size
    use value <- result.try(
      layout_math.at(data, index)
      |> result.map_error(fn(_) { IndexOutOfBounds(index, list.length(data)) }),
    )
    Ok([value, ..values])
  })
  |> result.map(list.reverse)
}

fn reduce_axis_output(
  data: List(Float),
  input_shape: List(Int),
  output_shape: List(Int),
  out_idx: Int,
  axis_idx: Int,
  axis_size: Int,
  reducer: fn(List(Float)) -> Result(Float, TensorError),
) -> Result(Float, TensorError) {
  let output_coords = layout_math.flat_to_multi(out_idx, output_shape)

  let values_result =
    list.range(0, axis_size - 1)
    |> list.fold(Ok([]), fn(acc, axis_pos) {
      use values <- result.try(acc)
      let input_coords =
        insert_reduced_axis(
          output_coords,
          axis_idx,
          axis_pos,
          output_shape,
          input_shape,
        )

      let input_idx = layout_math.multi_to_flat(input_coords, input_shape)
      use value <- result.try(
        layout_math.at(data, input_idx)
        |> result.map_error(fn(_) {
          IndexOutOfBounds(input_idx, list.length(data))
        }),
      )
      Ok([value, ..values])
    })

  case values_result {
    Ok(values) -> reducer(list.reverse(values))
    Error(error) -> Error(error)
  }
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
          layout_math.at(group, axis_pos)
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
        layout_math.at(data, index)
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
