//// Walsh-Hadamard preprocessing for low-bit quantization.
////
//// The implementation is pure Gleam and deterministic. Hot loops can move to a
//// NIF/CUDA kernel later without changing this contract.

import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape}
import viva_tensor/tensor.{type Tensor}

@external(erlang, "math", "sqrt")
fn sqrt(x: Float) -> Float

@external(erlang, "erlang", "phash2")
fn phash2(term: #(Int, Int), range: Int) -> Int

pub type HadamardPreprocess {
  HadamardPreprocess(
    tensor: Tensor,
    original_dim: Int,
    padded_dim: Int,
    seed: Int,
  )
}

pub fn try_preprocess(
  input: Tensor,
  seed: Int,
) -> Result(HadamardPreprocess, TensorError) {
  case input.shape {
    [_] -> {
      use values <- result.try(tensor.try_to_list(input))
      case values {
        [] ->
          Error(InvalidShape(
            "Hadamard preprocessing requires a non-empty vector",
          ))
        _ -> {
          let original_dim = list.length(values)
          let padded_dim = next_power_of_two(original_dim)
          let rotated =
            values
            |> pad_to(padded_dim)
            |> randomized_hadamard(seed)

          Ok(HadamardPreprocess(
            tensor: tensor.from_list(rotated),
            original_dim: original_dim,
            padded_dim: padded_dim,
            seed: seed,
          ))
        }
      }
    }
    shape ->
      Error(InvalidShape(
        "Hadamard preprocessing expects a vector, got "
        <> shape_to_string(shape),
      ))
  }
}

pub fn inverse(
  preprocessed: HadamardPreprocess,
) -> Result(Tensor, TensorError) {
  use values <- result.try(tensor.try_to_list(preprocessed.tensor))
  values
  |> inverse_randomized_hadamard(preprocessed.seed)
  |> list.take(preprocessed.original_dim)
  |> tensor.from_list
  |> Ok
}

pub fn try_walsh_hadamard(
  values: List(Float),
) -> Result(List(Float), TensorError) {
  case values {
    [] ->
      Error(InvalidShape("Walsh-Hadamard transform requires a non-empty vector"))
    _ ->
      case is_power_of_two(list.length(values)) {
        True -> Ok(hadamard(values))
        False ->
          Error(InvalidShape(
            "Walsh-Hadamard transform requires power-of-two length",
          ))
      }
  }
}

pub fn try_normalized_walsh_hadamard(
  values: List(Float),
) -> Result(List(Float), TensorError) {
  use transformed <- result.try(try_walsh_hadamard(values))
  Ok(normalize_hadamard(transformed))
}

pub fn randomized_hadamard(values: List(Float), seed: Int) -> List(Float) {
  let signed =
    values
    |> list.index_map(fn(value, index) {
      value *. int.to_float(random_sign(seed, index))
    })

  normalize_hadamard(hadamard(signed))
}

pub fn inverse_randomized_hadamard(
  values: List(Float),
  seed: Int,
) -> List(Float) {
  hadamard(values)
  |> normalize_hadamard
  |> list.index_map(fn(value, index) {
    value *. int.to_float(random_sign(seed, index))
  })
}

pub fn hadamard(values: List(Float)) -> List(Float) {
  case values {
    [] -> []
    [_] -> values
    _ -> {
      let half = list.length(values) / 2
      let left = list.take(values, half)
      let right = list.drop(values, half)
      let sum = list.map2(left, right, fn(a, b) { a +. b })
      let diff = list.map2(left, right, fn(a, b) { a -. b })
      list.append(hadamard(sum), hadamard(diff))
    }
  }
}

pub fn normalize_hadamard(values: List(Float)) -> List(Float) {
  case values {
    [] -> []
    _ -> {
      let scale = sqrt(int.to_float(list.length(values)))
      list.map(values, fn(value) { value /. scale })
    }
  }
}

pub fn pad_to(values: List(Float), target_size: Int) -> List(Float) {
  let missing = target_size - list.length(values)
  case missing <= 0 {
    True -> values
    False -> list.append(values, list.repeat(0.0, missing))
  }
}

pub fn next_power_of_two(value: Int) -> Int {
  next_power_of_two_loop(value, 1)
}

fn next_power_of_two_loop(value: Int, current: Int) -> Int {
  case current >= value {
    True -> current
    False -> next_power_of_two_loop(value, current * 2)
  }
}

fn is_power_of_two(value: Int) -> Bool {
  value > 0 && next_power_of_two(value) == value
}

fn random_sign(seed: Int, index: Int) -> Int {
  case phash2(#(seed, index), 2) {
    0 -> -1
    _ -> 1
  }
}

fn shape_to_string(shape: List(Int)) -> String {
  let body =
    shape
    |> list.map(int.to_string)
    |> list.fold("", fn(acc, dim) {
      case acc {
        "" -> dim
        _ -> acc <> ", " <> dim
      }
    })

  "[" <> body <> "]"
}
