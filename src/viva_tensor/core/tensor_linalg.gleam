//// Dense linear-algebra kernels used by the tensor facade.
////
//// Native dispatch and tensor storage construction stay in `viva_tensor/tensor`;
//// this module only works with materialized row-major data.

import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, IndexOutOfBounds}
import viva_tensor/core/ffi
import viva_tensor/core/layout_math

pub fn dot_values(a: List(Float), b: List(Float)) -> Float {
  list.map2(a, b, fn(x, y) { x *. y })
  |> list.fold(0.0, fn(acc, x) { acc +. x })
}

pub fn matmul_vec_values(
  mat_data: List(Float),
  vec_data: List(Float),
  rows: Int,
  cols: Int,
) -> List(Float) {
  list.range(0, rows - 1)
  |> list.map(fn(row_idx) {
    let start = row_idx * cols
    let row =
      mat_data
      |> list.drop(start)
      |> list.take(cols)
    dot_values(row, vec_data)
  })
}

pub fn matmul_values(
  a_data: List(Float),
  b_data: List(Float),
  rows: Int,
  inner: Int,
  cols: Int,
) -> List(Float) {
  let a_array = ffi.list_to_array(a_data)
  let b_array = ffi.list_to_array(b_data)

  list.range(0, rows - 1)
  |> list.flat_map(fn(i) {
    list.range(0, cols - 1)
    |> list.map(fn(j) {
      list.range(0, inner - 1)
      |> list.fold(0.0, fn(acc, k) {
        let a_ik = ffi.array_get(a_array, i * inner + k)
        let b_kj = ffi.array_get(b_array, k * cols + j)
        acc +. a_ik *. b_kj
      })
    })
  })
}

pub fn transpose_values(
  data: List(Float),
  rows: Int,
  cols: Int,
) -> Result(List(Float), TensorError) {
  list.range(0, cols - 1)
  |> list.fold(Ok([]), fn(acc, col) {
    use values <- result.try(acc)
    use row_values <- result.try(
      list.range(0, rows - 1)
      |> list.fold(Ok([]), fn(row_acc, row) {
        use row_result <- result.try(row_acc)
        let index = row * cols + col
        use value <- result.try(
          layout_math.at(data, index)
          |> result.map_error(fn(_) {
            IndexOutOfBounds(index, list.length(data))
          }),
        )
        Ok([value, ..row_result])
      })
      |> result.map(list.reverse),
    )
    Ok(list.append(values, row_values))
  })
}

pub fn outer_values(a_data: List(Float), b_data: List(Float)) -> List(Float) {
  list.flat_map(a_data, fn(ai) { list.map(b_data, fn(bj) { ai *. bj }) })
}
