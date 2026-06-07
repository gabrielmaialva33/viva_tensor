//// TurboQuant-inspired online vector quantization.
////
//// Based on Google Research's TurboQuant direction: data-oblivious random
//// rotation, scalar quantization in the rotated basis, and an optional 1-bit
//// residual correction for inner-product workloads.
////
//// This module is a pure Gleam reference implementation. It is intentionally
//// simple and deterministic so the algorithmic contract can be tested before
//// moving the hot loops into a NIF/CUDA kernel.

import gleam/float
import gleam/int
import gleam/list
import viva_tensor/core/error.{type TensorError, InvalidShape, ShapeMismatch}
import viva_tensor/core/ffi
import viva_tensor/tensor.{type Tensor}

@external(erlang, "math", "sqrt")
fn sqrt(x: Float) -> Float

@external(erlang, "erlang", "phash2")
fn phash2(term: #(Int, Int), range: Int) -> Int

/// TurboQuant configuration.
pub type Config {
  Config(
    /// Main scalar quantizer bit-width. Google reports strong KV-cache results
    /// around 2.5-3.5 total bits; here the residual bit is configured
    /// separately with `use_qjl_residual`.
    bits: Int,
    /// Seed for deterministic data-oblivious random rotation.
    seed: Int,
    /// Add a 1-bit residual correction in the rotated basis.
    use_qjl_residual: Bool,
  )
}

/// Compressed vector.
pub type QuantizedVector {
  QuantizedVector(
    codes: List(Int),
    residual_signs: List(Int),
    scale: Float,
    residual_scale: Float,
    original_dim: Int,
    padded_dim: Int,
    bits: Int,
    seed: Int,
    use_qjl_residual: Bool,
    memory_bytes: Int,
  )
}

/// Default 3-bit main quantizer plus 1 residual bit.
pub fn default_config() -> Config {
  Config(bits: 3, seed: 0, use_qjl_residual: True)
}

/// Quantize a vector with randomized Hadamard rotation and scalar levels.
pub fn quantize(
  values: List(Float),
  config: Config,
) -> Result(QuantizedVector, TensorError) {
  case list.length(values), valid_bits(config.bits) {
    0, _ -> Error(InvalidShape("TurboQuant requires a non-empty vector"))
    _, False -> Error(InvalidShape("TurboQuant bits must be between 1 and 8"))
    dim, True -> {
      let padded_dim = next_power_of_two(dim)
      let padded = pad_to(values, padded_dim)
      let rotated = randomized_hadamard(padded, config.seed)
      let scale = max_abs(rotated)
      let codes =
        list.map(rotated, fn(value) { nearest_code(value, config.bits, scale) })
      let main =
        list.map(codes, fn(code) { decode_code(code, config.bits, scale) })
      let residual =
        list.map2(rotated, main, fn(value, approx) { value -. approx })

      let residual_signs = case config.use_qjl_residual {
        True -> list.map(residual, sign_bit)
        False -> []
      }

      let residual_scale = case config.use_qjl_residual {
        True -> mean_abs(residual)
        False -> 0.0
      }

      Ok(QuantizedVector(
        codes: codes,
        residual_signs: residual_signs,
        scale: scale,
        residual_scale: residual_scale,
        original_dim: dim,
        padded_dim: padded_dim,
        bits: config.bits,
        seed: config.seed,
        use_qjl_residual: config.use_qjl_residual,
        memory_bytes: estimate_memory_bytes(
          padded_dim,
          config.bits,
          config.use_qjl_residual,
        ),
      ))
    }
  }
}

/// Quantize a rank-1 tensor.
pub fn quantize_tensor(
  tensor: Tensor,
  config: Config,
) -> Result(QuantizedVector, TensorError) {
  case tensor.shape {
    [_] -> quantize(tensor.to_list(tensor), config)
    shape ->
      Error(InvalidShape(
        "TurboQuant currently expects a vector tensor, got "
        <> shape_to_string(shape),
      ))
  }
}

/// Dequantize back to a plain vector.
pub fn dequantize(vector: QuantizedVector) -> List(Float) {
  dequantize_rotated(vector)
  |> inverse_randomized_hadamard(vector.seed)
  |> list.take(vector.original_dim)
}

/// Dequantize back to a rank-1 tensor.
pub fn dequantize_tensor(vector: QuantizedVector) -> Tensor {
  tensor.from_list(dequantize(vector))
}

/// Native fast path — the NIF this module's header promises ("moving the hot
/// loops into a NIF/CUDA kernel"). Does the full TurboQuant_mse round-trip
/// (randomized Hadamard rotation + **Lloyd-Max** optimal-MSE scalar quantize +
/// dequantize + inverse rotation) in one C call.
///
/// Two upgrades over the pure-Gleam path: it uses the optimal-MSE Lloyd-Max
/// codebook for the rotated-basis Normal distribution (vs. uniform levels), and
/// it runs orders of magnitude faster (FWHT in C). Operates per row on a 1D/2D
/// native tensor and returns the reconstructed FP64 tensor — for weight
/// fake-quant and distortion measurement. `bits` in 1..8.
pub fn quantize_dequantize_native(
  input: Tensor,
  bits: Int,
  seed: Int,
) -> Result(Tensor, TensorError) {
  case tensor.native_ref(input) {
    Ok(ref) ->
      case ffi.nt_turboquant(ref, bits, seed) {
        Ok(q) -> Ok(tensor.from_native_ref(q, tensor.shape(input)))
        Error(msg) -> Error(InvalidShape("turboquant nif: " <> msg))
      }
    Error(_) ->
      Error(InvalidShape(
        "quantize_dequantize_native requires a native tensor (viva_tensor.native_*)",
      ))
  }
}

/// Estimate the inner product between an uncompressed query and a compressed
/// vector without materializing the full dequantized vector.
pub fn inner_product(
  query: List(Float),
  vector: QuantizedVector,
) -> Result(Float, TensorError) {
  case list.length(query) == vector.original_dim {
    False ->
      Error(
        ShapeMismatch(expected: [vector.original_dim], got: [list.length(query)]),
      )
    True -> {
      let rotated_query =
        query
        |> pad_to(vector.padded_dim)
        |> randomized_hadamard(vector.seed)

      Ok(dot(rotated_query, dequantize_rotated(vector)))
    }
  }
}

/// Effective compression ratio relative to FP32 storage.
pub fn compression_ratio(vector: QuantizedVector) -> Float {
  let original_bytes = vector.original_dim * 4
  int.to_float(original_bytes) /. int.to_float(vector.memory_bytes)
}

fn dequantize_rotated(vector: QuantizedVector) -> List(Float) {
  let main =
    list.map(vector.codes, fn(code) {
      decode_code(code, vector.bits, vector.scale)
    })

  case vector.use_qjl_residual {
    False -> main
    True ->
      list.map2(main, vector.residual_signs, fn(value, sign) {
        value +. int.to_float(sign) *. vector.residual_scale
      })
  }
}

fn randomized_hadamard(values: List(Float), seed: Int) -> List(Float) {
  let signed =
    values
    |> list.index_map(fn(value, index) {
      value *. int.to_float(random_sign(seed, index))
    })

  normalize_hadamard(hadamard(signed))
}

fn inverse_randomized_hadamard(values: List(Float), seed: Int) -> List(Float) {
  hadamard(values)
  |> normalize_hadamard
  |> list.index_map(fn(value, index) {
    value *. int.to_float(random_sign(seed, index))
  })
}

fn hadamard(values: List(Float)) -> List(Float) {
  case values {
    [] -> []
    [_] -> values
    _ -> {
      let half = list.length(values) / 2
      let left = values |> list.take(half) |> hadamard
      let right = values |> list.drop(half) |> hadamard
      list.append(
        list.map2(left, right, fn(a, b) { a +. b }),
        list.map2(left, right, fn(a, b) { a -. b }),
      )
    }
  }
}

fn normalize_hadamard(values: List(Float)) -> List(Float) {
  let n = list.length(values)
  case n {
    0 -> []
    _ -> {
      let scale = 1.0 /. sqrt(int.to_float(n))
      list.map(values, fn(value) { value *. scale })
    }
  }
}

fn nearest_code(value: Float, bits: Int, scale: Float) -> Int {
  let levels = int_pow2(bits)
  range_int(0, levels - 1)
  |> list.fold(#(0, 1.0e308), fn(best, code) {
    let decoded = decode_code(code, bits, scale)
    let distance = float.absolute_value(value -. decoded)
    case distance <. best.1 {
      True -> #(code, distance)
      False -> best
    }
  })
  |> fn(best) { best.0 }
}

fn decode_code(code: Int, bits: Int, scale: Float) -> Float {
  case scale == 0.0 {
    True -> 0.0
    False -> {
      let levels = int_pow2(bits)
      -1.0
      *. scale
      +. 2.0
      *. scale
      *. int.to_float(code)
      /. int.to_float(levels - 1)
    }
  }
}

fn random_sign(seed: Int, index: Int) -> Int {
  case phash2(#(seed, index), 2) {
    0 -> -1
    _ -> 1
  }
}

fn sign_bit(value: Float) -> Int {
  case value <. 0.0 {
    True -> -1
    False -> 1
  }
}

fn max_abs(values: List(Float)) -> Float {
  values
  |> list.map(float.absolute_value)
  |> list.fold(0.0, float.max)
}

fn mean_abs(values: List(Float)) -> Float {
  case list.length(values) {
    0 -> 0.0
    n ->
      values
      |> list.map(float.absolute_value)
      |> list.fold(0.0, fn(acc, value) { acc +. value })
      |> fn(total) { total /. int.to_float(n) }
  }
}

fn dot(a: List(Float), b: List(Float)) -> Float {
  list.map2(a, b, fn(x, y) { x *. y })
  |> list.fold(0.0, fn(acc, value) { acc +. value })
}

fn pad_to(values: List(Float), size: Int) -> List(Float) {
  let missing = size - list.length(values)
  case missing <= 0 {
    True -> values
    False -> list.append(values, list.repeat(0.0, missing))
  }
}

fn next_power_of_two(n: Int) -> Int {
  next_power_of_two_loop(1, n)
}

fn next_power_of_two_loop(current: Int, n: Int) -> Int {
  case current >= n {
    True -> current
    False -> next_power_of_two_loop(current * 2, n)
  }
}

fn int_pow2(exp: Int) -> Int {
  case exp <= 0 {
    True -> 1
    False -> 2 * int_pow2(exp - 1)
  }
}

fn valid_bits(bits: Int) -> Bool {
  bits >= 1 && bits <= 8
}

fn estimate_memory_bytes(dim: Int, bits: Int, use_residual: Bool) -> Int {
  let residual_bits = case use_residual {
    True -> dim
    False -> 0
  }
  let payload_bits = dim * bits + residual_bits
  // Payload plus two fp32 scales. Metadata is not counted here, matching the
  // existing quantization modules' focus on tensor storage footprint.
  { payload_bits + 7 } / 8 + 8
}

fn shape_to_string(shape: List(Int)) -> String {
  let body =
    shape
    |> list.map(int.to_string)
    |> list.fold("", fn(acc, part) {
      case acc == "" {
        True -> part
        False -> acc <> ", " <> part
      }
    })

  "[" <> body <> "]"
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
