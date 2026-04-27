import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor/quant/turboquant
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

fn assert_close(actual: Float, expected: Float, tolerance: Float) -> Nil {
  let _ =
    { float.absolute_value(actual -. expected) <. tolerance }
    |> should.be_true()
  Nil
}

fn dot(a: List(Float), b: List(Float)) -> Float {
  list.map2(a, b, fn(x, y) { x *. y })
  |> list.fold(0.0, fn(acc, value) { acc +. value })
}

pub fn turboquant_roundtrip_shape_test() {
  let config = turboquant.default_config()

  case turboquant.quantize([1.0, -2.0, 3.0, 0.5, -0.25], config) {
    Ok(quantized) -> {
      quantized.original_dim |> should.equal(5)
      quantized.padded_dim |> should.equal(8)
      turboquant.dequantize(quantized) |> list.length |> should.equal(5)
    }
    Error(_) -> should.fail()
  }
}

pub fn turboquant_compresses_vector_test() {
  let config = turboquant.Config(bits: 3, seed: 7, use_qjl_residual: True)
  let values = list.repeat(0.25, 128)

  case turboquant.quantize(values, config) {
    Ok(quantized) -> {
      let _ = { quantized.memory_bytes < 128 * 4 } |> should.be_true()
      let _ =
        { turboquant.compression_ratio(quantized) >. 4.0 } |> should.be_true()
      Nil
    }
    Error(_) -> should.fail()
  }
}

pub fn turboquant_inner_product_tracks_dequantized_dot_test() {
  let config = turboquant.Config(bits: 4, seed: 11, use_qjl_residual: True)
  let values = [0.5, -1.0, 2.0, 0.25, -0.75, 1.25]
  let query = [1.0, 0.5, -0.25, 2.0, -1.0, 0.75]

  case turboquant.quantize(values, config) {
    Ok(quantized) -> {
      let recovered = turboquant.dequantize(quantized)
      let expected = dot(query, recovered)
      case turboquant.inner_product(query, quantized) {
        Ok(estimated) -> assert_close(estimated, expected, 0.0001)
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn turboquant_tensor_roundtrip_test() {
  let config = turboquant.Config(bits: 3, seed: 3, use_qjl_residual: False)
  let vector = tensor.from_list([1.0, 2.0, 3.0, 4.0])

  case turboquant.quantize_tensor(vector, config) {
    Ok(quantized) -> {
      let recovered = turboquant.dequantize_tensor(quantized)
      tensor.shape(recovered) |> should.equal([4])
      tensor.to_list(recovered) |> list.length |> should.equal(4)
    }
    Error(_) -> should.fail()
  }
}

pub fn turboquant_rejects_invalid_bits_test() {
  let config = turboquant.Config(bits: 0, seed: 0, use_qjl_residual: False)
  turboquant.quantize([1.0, 2.0], config) |> should.be_error()
}
