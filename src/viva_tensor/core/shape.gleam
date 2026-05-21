//// Shape manipulation - reshape, slice, concat, stack.
////
//// The "plumbing" of tensor operations. Not glamorous, but essential.
////
//// Philosophy: these ops should be zero-copy when possible (via strides),
//// but we're not there yet. Current implementation copies data.
//// TODO: implement as views where safe (reshape of contiguous tensor, slice, etc.)
////
//// Reshape is particularly tricky because it changes how we interpret memory.
//// A [2,3] tensor reshaped to [3,2] has the same data but different semantics.
//// This only works if the tensor is contiguous (strides match row-major order).

import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/tensor.{type Tensor}

// --- Reshape ----------------------------------------------------------------

/// Reshape to new dimensions. Total size must match (obviously).
pub fn reshape(t: Tensor, new_shape: List(Int)) -> Result(Tensor, TensorError) {
  let old_size = tensor.size(t)
  let new_size = list.fold(new_shape, 1, fn(acc, dim) { acc * dim })

  case old_size == new_size {
    True -> {
      use data <- result.try(tensor.try_to_list(t))
      tensor.new(data, new_shape)
    }
    False ->
      Error(error.InvalidShape(
        "Cannot reshape: size mismatch ("
        <> int.to_string(old_size)
        <> " vs "
        <> int.to_string(new_size)
        <> ")",
      ))
  }
}

/// Flatten to 1D tensor
pub fn flatten(t: Tensor) -> Tensor {
  let data = tensor.to_list(t)
  tensor.from_list(data)
}

/// Remove all dimensions of size 1
pub fn squeeze(t: Tensor) -> Tensor {
  let data = tensor.to_list(t)
  let new_shape = list.filter(tensor.shape(t), fn(d) { d != 1 })
  let final_shape = case new_shape {
    [] -> [1]
    _ -> new_shape
  }
  case tensor.new(data, final_shape) {
    Ok(result) -> result
    Error(_) -> t
  }
}

/// Remove dimension at specific axis if it's 1
pub fn squeeze_axis(t: Tensor, axis: Int) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(t)
  case list_at(shp, axis) {
    Error(_) -> Error(error.DimensionError("Axis out of bounds"))
    Ok(d) -> {
      case d == 1 {
        False -> Error(error.InvalidShape("Dimension at axis is not 1"))
        True -> {
          let data = tensor.to_list(t)
          let new_shape =
            shp
            |> list.index_map(fn(dim, i) { #(dim, i) })
            |> list.filter(fn(pair) { pair.1 != axis })
            |> list.map(fn(pair) { pair.0 })
          tensor.new(data, new_shape)
        }
      }
    }
  }
}

/// Add dimension of size 1 at specified axis
pub fn unsqueeze(t: Tensor, axis: Int) -> Tensor {
  let data = tensor.to_list(t)
  let shp = tensor.shape(t)
  let rnk = list.length(shp)
  let insert_at = case axis < 0 {
    True -> rnk + axis + 1
    False -> axis
  }

  let #(before, after) = list.split(shp, insert_at)
  let new_shape = list.flatten([before, [1], after])
  case tensor.new(data, new_shape) {
    Ok(result) -> result
    Error(_) -> t
  }
}

/// Alias for unsqueeze - expand dimensions
pub fn expand_dims(t: Tensor, axis: Int) -> Tensor {
  unsqueeze(t, axis)
}

// --- Slicing ----------------------------------------------------------------

/// Take first n elements (along axis 0).
pub fn take_first(t: Tensor, n: Int) -> Tensor {
  let data = tensor.to_list(t)
  case tensor.shape(t) {
    [] -> t
    [first_dim, ..rest_dims] -> {
      let take_n = int.min(n, first_dim)
      let stride = list.fold(rest_dims, 1, fn(acc, d) { acc * d })
      let new_data = list.take(data, take_n * stride)
      let new_shape = [take_n, ..rest_dims]
      case tensor.new(new_data, new_shape) {
        Ok(result) -> result
        Error(_) -> t
      }
    }
  }
}

/// Take last N elements along first axis
pub fn take_last(t: Tensor, n: Int) -> Tensor {
  let data = tensor.to_list(t)
  case tensor.shape(t) {
    [] -> t
    [first_dim, ..rest_dims] -> {
      let take_n = int.min(n, first_dim)
      let stride = list.fold(rest_dims, 1, fn(acc, d) { acc * d })
      let skip = { first_dim - take_n } * stride
      let new_data = list.drop(data, skip)
      let new_shape = [take_n, ..rest_dims]
      case tensor.new(new_data, new_shape) {
        Ok(result) -> result
        Error(_) -> t
      }
    }
  }
}

/// General slice - specify start indices and lengths for each dimension.
/// This one's tricky for n-dimensional tensors.
pub fn slice(
  t: Tensor,
  start: List(Int),
  lengths: List(Int),
) -> Result(Tensor, TensorError) {
  use data <- result.try(tensor.try_to_list(t))
  let shp = tensor.shape(t)
  let r = tensor.rank(t)

  case list.length(start) == r && list.length(lengths) == r {
    False ->
      Error(error.DimensionError("Slice dimensions must match tensor rank"))
    True -> {
      case r {
        1 -> {
          use s <- result.try(
            list_at(start, 0)
            |> result.map_error(fn(_) {
              error.DimensionError("Slice start is missing")
            }),
          )
          use len <- result.try(
            list_at(lengths, 0)
            |> result.map_error(fn(_) {
              error.DimensionError("Slice length is missing")
            }),
          )
          let sliced = data |> list.drop(s) |> list.take(len)
          tensor.new(sliced, [len])
        }
        _ -> {
          // Multi-dimensional slice
          let new_size = list.fold(lengths, 1, fn(acc, d) { acc * d })

          let result_data =
            range_int(0, new_size - 1)
            |> list.fold(Ok([]), fn(acc, flat_idx) {
              use values <- result.try(acc)
              let local_indices = flat_to_multi(flat_idx, lengths)
              let global_indices =
                list.map2(local_indices, start, fn(l, s) { l + s })
              let global_flat = multi_to_flat(global_indices, shp)
              use value <- result.try(
                list_at_float(data, global_flat)
                |> result.map_error(fn(_) {
                  error.IndexOutOfBounds(global_flat, list.length(data))
                }),
              )
              Ok([value, ..values])
            })

          use result_data <- result.try(result_data)
          let result = list.reverse(result_data)
          tensor.new(result, lengths)
        }
      }
    }
  }
}

// --- Concat & Stack ---------------------------------------------------------

/// Concat 1D tensors. Just appends the data, nothing fancy.
pub fn concat(tensors: List(Tensor)) -> Tensor {
  let data = list.flat_map(tensors, fn(t) { tensor.to_list(t) })
  tensor.from_list(data)
}

/// Concat along arbitrary axis.
///
/// This function is gnarly. The general case requires computing which source
/// tensor each output index maps to, then translating coordinates. O(n) where
/// n is total output size, but the constant factor is high due to all the
/// index arithmetic. For axis=0, we fast-path to simple concatenation.
pub fn concat_axis(
  tensors: List(Tensor),
  axis: Int,
) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(error.InvalidShape("Cannot concatenate empty list"))
    [single] -> Ok(single)
    [first, ..rest] -> {
      let base_shape = tensor.shape(first)
      let r = list.length(base_shape)

      case axis >= 0 && axis < r {
        False -> Error(error.DimensionError("Invalid axis for concatenation"))
        True -> {
          // Verify all tensors have same shape except on concat axis
          let shapes_ok =
            list.all(rest, fn(t) {
              let t_shape = tensor.shape(t)
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
              Error(error.InvalidShape(
                "Shapes must match except on concat axis",
              ))
            True -> {
              // Build new shape
              let concat_dim =
                list.fold(tensors, 0, fn(acc, t) {
                  case list.drop(tensor.shape(t), axis) |> list.first {
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

              // For axis=0, just append all data
              case axis == 0 {
                True -> {
                  use data <- result.try(concat_data(tensors))
                  tensor.new(data, new_shape)
                }
                False -> {
                  // General case: interleave based on axis
                  let total_size =
                    list.fold(new_shape, 1, fn(acc, d) { acc * d })

                  let result_data =
                    range_int(0, total_size - 1)
                    |> list.fold(Ok([]), fn(acc, flat_idx) {
                      use values <- result.try(acc)
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
                                list.drop(tensor.shape(t), axis) |> list.first
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

                      use value <- result.try(concat_axis_value(
                        tensors,
                        tensor_idx,
                        local_indices,
                      ))
                      Ok([value, ..values])
                    })

                  use result_data <- result.try(result_data)
                  let result = list.reverse(result_data)
                  tensor.new(result, new_shape)
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
pub fn stack(tensors: List(Tensor), axis: Int) -> Result(Tensor, TensorError) {
  case tensors {
    [] -> Error(error.InvalidShape("Cannot stack empty list"))
    [first, ..rest] -> {
      let base_shape = tensor.shape(first)
      let shapes_ok = list.all(rest, fn(t) { tensor.shape(t) == base_shape })

      case shapes_ok {
        False -> Error(error.ShapeMismatch(base_shape, []))
        True -> {
          let _n_tensors = list.length(tensors)
          let r = list.length(base_shape)
          let insert_axis = case axis < 0 {
            True -> r + axis + 1
            False -> axis
          }

          case insert_axis >= 0 && insert_axis <= r {
            False -> Error(error.DimensionError("Invalid axis for stacking"))
            True -> {
              // Unsqueeze each tensor and concat
              let unsqueezed =
                list.map(tensors, fn(t) { unsqueeze(t, insert_axis) })
              concat_axis(unsqueezed, insert_axis)
            }
          }
        }
      }
    }
  }
}

// --- Helpers ----------------------------------------------------------------
// Index conversion between flat and multi-dimensional representations.
// flat_to_multi: linear index → [i, j, k, ...]
// multi_to_flat: [i, j, k, ...] → linear index
//
// The formula: flat = Σ(index[i] * stride[i])
// where stride[i] = Π(shape[j]) for j > i (row-major order)
//
// Example: shape [2,3,4], index [1,2,3]
//   strides = [12, 4, 1]
//   flat = 1*12 + 2*4 + 3*1 = 23

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

fn concat_data(tensors: List(Tensor)) -> Result(List(Float), TensorError) {
  list.fold(tensors, Ok([]), fn(acc, t) {
    use values <- result.try(acc)
    use data <- result.try(tensor.try_to_list(t))
    Ok(list.append(values, data))
  })
}

fn concat_axis_value(
  tensors: List(Tensor),
  tensor_idx: Int,
  local_indices: List(Int),
) -> Result(Float, TensorError) {
  use t <- result.try(
    list_at(tensors, tensor_idx)
    |> result.map_error(fn(_) {
      error.DimensionError("Concat index does not map to a source tensor")
    }),
  )
  use data <- result.try(tensor.try_to_list(t))
  let strides = compute_strides(tensor.shape(t))
  let local_flat =
    list.zip(local_indices, strides)
    |> list.fold(0, fn(acc, pair) {
      let #(idx, stride) = pair
      acc + idx * stride
    })

  list_at_float(data, local_flat)
  |> result.map_error(fn(_) {
    error.IndexOutOfBounds(local_flat, list.length(data))
  })
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
