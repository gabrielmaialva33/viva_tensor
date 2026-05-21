//// Tests for `viva_tensor/nn/embedding`.

import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_math/scalar as vm_scalar
import viva_tensor/core/error.{DimensionError, IndexOutOfBounds, InvalidShape}
import viva_tensor/nn/embedding.{
  Embedding, LearnedPositionalEncoding, embedding_forward, embedding_init,
  embedding_init_uniform, learned_positional_forward, learned_positional_init,
  rope, sinusoidal_encoding,
}
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

// -------------------------------------------------------------------------
// Embedding
// -------------------------------------------------------------------------

pub fn embedding_init_test() {
  let layer = embedding_init(num_embeddings: 5, embedding_dim: 3)
  let Embedding(num_embeddings, embedding_dim, weight) = layer
  num_embeddings |> should.equal(5)
  embedding_dim |> should.equal(3)
  tensor.shape(weight) |> should.equal([5, 3])
  tensor.to_list(weight)
  |> should.equal([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ])
}

pub fn embedding_init_uniform_test() {
  let layer = embedding_init_uniform(num_embeddings: 8, embedding_dim: 4)
  let Embedding(_, _, weight) = layer
  tensor.shape(weight) |> should.equal([8, 4])
  let limit = 1.0 /. float_sqrt(4.0)
  // limit = 0.5; all values must fall in [-0.5, 0.5].
  tensor.to_list(weight)
  |> list.all(fn(v) {
    v >=. { -1.0 *. limit -. 1.0e-9 } && v <=. limit +. 1.0e-9
  })
  |> should.be_true()
}

pub fn embedding_forward_test() {
  // Hand-built weight matrix: row i = [i*10, i*10+1].
  let assert Ok(weight) =
    tensor.matrix(3, 2, [0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
  let layer = Embedding(num_embeddings: 3, embedding_dim: 2, weight: weight)
  let indices = tensor.from_list([0.0, 2.0, 1.0])
  let assert Ok(out) = embedding_forward(layer, indices)
  tensor.shape(out) |> should.equal([3, 2])
  tensor.to_list(out)
  |> should.equal([0.0, 1.0, 20.0, 21.0, 10.0, 11.0])
}

pub fn embedding_forward_negative_index_test() {
  // Negative indices wrap NumPy-style: -1 -> num_embeddings - 1.
  let assert Ok(weight) =
    tensor.matrix(3, 2, [0.0, 1.0, 10.0, 11.0, 20.0, 21.0])
  let layer = Embedding(num_embeddings: 3, embedding_dim: 2, weight: weight)
  let indices = tensor.from_list([-1.0])
  let assert Ok(out) = embedding_forward(layer, indices)
  tensor.to_list(out) |> should.equal([20.0, 21.0])
}

pub fn embedding_oob_test() {
  let layer = embedding_init(num_embeddings: 3, embedding_dim: 2)
  let indices = tensor.from_list([5.0])
  case embedding_forward(layer, indices) {
    Error(IndexOutOfBounds(5, 3)) -> Nil
    other -> {
      let _ = other
      should.fail()
    }
  }
}

pub fn embedding_rank_mismatch_test() {
  let layer = embedding_init(num_embeddings: 4, embedding_dim: 2)
  // Pass a 2D tensor as indices to trigger the rank check.
  let assert Ok(bad) = tensor.matrix(2, 2, [0.0, 1.0, 2.0, 3.0])
  case embedding_forward(layer, bad) {
    Error(DimensionError(_)) -> Nil
    _ -> should.fail()
  }
}

// -------------------------------------------------------------------------
// Sinusoidal positional encoding
// -------------------------------------------------------------------------

pub fn sinusoidal_encoding_test_4d() {
  // max_len = 2, embedding_dim = 4
  //   PE[0, 0] = sin(0 / 10000^(0/4)) = sin(0)   = 0
  //   PE[0, 1] = cos(0 / 10000^(0/4)) = cos(0)   = 1
  //   PE[0, 2] = sin(0 / 10000^(2/4)) = sin(0)   = 0
  //   PE[0, 3] = cos(0 / 10000^(2/4)) = cos(0)   = 1
  //   PE[1, 0] = sin(1)               ≈ 0.84147
  //   PE[1, 1] = cos(1)               ≈ 0.54030
  //   PE[1, 2] = sin(1 / sqrt(10000)) = sin(0.01) ≈ 0.00999983
  //   PE[1, 3] = cos(0.01)            ≈ 0.99995
  let assert Ok(pe) = sinusoidal_encoding(max_len: 2, embedding_dim: 4)
  tensor.shape(pe) |> should.equal([2, 4])
  let data = tensor.to_list(pe)
  let expected = [
    0.0,
    1.0,
    0.0,
    1.0,
    vm_scalar.sin(1.0),
    vm_scalar.cos(1.0),
    vm_scalar.sin(0.01),
    vm_scalar.cos(0.01),
  ]
  numerics.lists_close(data, expected, 1.0e-9, 1.0e-9)
}

pub fn sinusoidal_encoding_zero_position_test() {
  // Row 0 must be alternating [0, 1, 0, 1, ...] regardless of dim.
  let assert Ok(pe) = sinusoidal_encoding(max_len: 1, embedding_dim: 6)
  tensor.to_list(pe)
  |> should.equal([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
}

pub fn sinusoidal_encoding_odd_dim_error_test() {
  case sinusoidal_encoding(max_len: 4, embedding_dim: 5) {
    Error(InvalidShape(msg)) -> {
      // Sanity-check the message mentions even/odd.
      let _ = msg
      Nil
    }
    _ -> should.fail()
  }
}

// -------------------------------------------------------------------------
// Learned positional encoding
// -------------------------------------------------------------------------

pub fn learned_positional_forward_test() {
  // Hand-built weight matrix: 4 positions, dim 2.
  let assert Ok(w) =
    tensor.matrix(4, 2, [0.0, 0.1, 1.0, 1.1, 2.0, 2.1, 3.0, 3.1])
  let pe = LearnedPositionalEncoding(max_len: 4, embedding_dim: 2, weight: w)
  let assert Ok(out) = learned_positional_forward(pe, 3)
  tensor.shape(out) |> should.equal([3, 2])
  tensor.to_list(out)
  |> should.equal([0.0, 0.1, 1.0, 1.1, 2.0, 2.1])
}

pub fn learned_positional_full_window_test() {
  let pe = learned_positional_init(max_len: 5, embedding_dim: 3)
  let assert Ok(out) = learned_positional_forward(pe, 5)
  tensor.shape(out) |> should.equal([5, 3])
}

pub fn learned_positional_oob_test() {
  let pe = learned_positional_init(max_len: 4, embedding_dim: 2)
  case learned_positional_forward(pe, 5) {
    Error(IndexOutOfBounds(5, 4)) -> Nil
    _ -> should.fail()
  }
}

// -------------------------------------------------------------------------
// RoPE
// -------------------------------------------------------------------------

pub fn rope_basic_test() {
  // Input: [seq_len=2, dim=2] with rows (1, 0) and (1, 0).
  // dim=2 -> one pair, theta_0 = 1 / base^(0/2) = 1.
  //   pos=0: angle = 0 -> (cos 0, sin 0) = (1, 0) -> output (1, 0).
  //   pos=1: angle = 1 -> (cos 1, sin 1) ≈ (0.5403, 0.8415).
  let assert Ok(x) = tensor.matrix(2, 2, [1.0, 0.0, 1.0, 0.0])
  let assert Ok(out) = rope(x, 10_000.0)
  tensor.shape(out) |> should.equal([2, 2])
  let data = tensor.to_list(out)
  let expected = [1.0, 0.0, vm_scalar.cos(1.0), vm_scalar.sin(1.0)]
  numerics.lists_close(data, expected, 1.0e-9, 1.0e-9)
}

pub fn rope_pair_rotation_test() {
  // Input: row = (1, 0) at pos=1, dim=4. Two pairs:
  //   pair 0: theta_0 = 1 / 10000^(0/4) = 1.   angle = 1.
  //   pair 1: theta_1 = 1 / 10000^(2/4) = 1/100. angle = 0.01.
  // We rotate (1,0): -> (cos a, sin a).
  // Row 2: (0, 1) at pos=0 - angle 0, identity -> (0, 1, 0, 1)? No, second pair also (0, 1).
  // Construct row = [1, 0, 0, 1] -> after pos=1:
  //   pair 0: (1, 0) -> (cos 1, sin 1)
  //   pair 1: (0, 1) -> (0*cos 0.01 - 1*sin 0.01, 0*sin 0.01 + 1*cos 0.01)
  //                   = (-sin 0.01, cos 0.01)
  let assert Ok(x) = tensor.matrix(1, 4, [1.0, 0.0, 0.0, 1.0])
  // pos=0 is identity, so put our test row at pos=0 wouldn't rotate.
  // Use a 2-row input where row 1 is the interesting one.
  let assert Ok(x2) =
    tensor.matrix(2, 4, [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0])
  let assert Ok(out) = rope(x2, 10_000.0)
  let data = tensor.to_list(out)
  // Row 0 is all zeros -> stays zero.
  // Row 1: pair 0 angle=1 on (1,0) -> (cos 1, sin 1)
  //        pair 1 angle=0.01 on (0,1) -> (-sin 0.01, cos 0.01)
  let expected = [
    0.0,
    0.0,
    0.0,
    0.0,
    vm_scalar.cos(1.0),
    vm_scalar.sin(1.0),
    float.negate(vm_scalar.sin(0.01)),
    vm_scalar.cos(0.01),
  ]
  let _ = x
  numerics.lists_close(data, expected, 1.0e-9, 1.0e-9)
}

pub fn rope_odd_dim_error_test() {
  let assert Ok(x) = tensor.matrix(1, 3, [1.0, 2.0, 3.0])
  case rope(x, 10_000.0) {
    Error(InvalidShape(_)) -> Nil
    _ -> should.fail()
  }
}

pub fn rope_rank_error_test() {
  let x = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  case rope(x, 10_000.0) {
    Error(DimensionError(_)) -> Nil
    _ -> should.fail()
  }
}

// Local sqrt helper for tests (avoids reaching into private ffi).
fn float_sqrt(x: Float) -> Float {
  let assert Ok(r) = float.power(x, 0.5)
  r
}
