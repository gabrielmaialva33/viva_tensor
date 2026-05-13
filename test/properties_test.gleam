//// Property-based tests for `viva_tensor`.
////
//// These exercise algebraic invariants (commutativity, associativity,
//// distributivity, identity, transpose-invariance) and shape invariants
//// (reshape, layer-norm) over randomly generated tensors.
////
//// Non-determinism
//// ---------------
//// The harness in `support/properties` uses `int.random/1` — the BEAM PRNG —
//// which is NOT seedable. Every run draws a fresh sample, so a failure
//// surfaced today reflects a real bug, but reproducing the exact same input
//// twice in a row is unlikely. See the module doc on `support/properties`
//// for the full rationale.
////
//// Tolerances follow the rest of the suite: `rtol=1e-5`, `atol=1e-7`. They
//// are looser than the deterministic reference tests because random
//// compositions accumulate floating-point error.

import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import support/properties as prop
import viva_tensor as vt

pub fn main() -> Nil {
  gleeunit.main()
}

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-7

const iters: Int = 30

// --- Helpers ----------------------------------------------------------------

fn tensors_close(a: vt.Tensor, b: vt.Tensor) -> Bool {
  case vt.shape(a) == vt.shape(b) {
    False -> False
    True -> numerics.lists_close(vt.to_list(a), vt.to_list(b), rtol, atol)
  }
}

fn unwrap(r: Result(a, e)) -> a {
  let assert Ok(value) = r
  value
}

fn random_2d_shape() -> List(Int) {
  prop.random_shape(2, 1, 6)
}

// --- Tensor algebra invariants ---------------------------------------------

pub fn prop_add_commutative_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let b = prop.random_tensor(shape, 5.0)
    let ab = unwrap(vt.add(a, b))
    let ba = unwrap(vt.add(b, a))
    tensors_close(ab, ba)
  })
  |> should.be_true
}

pub fn prop_mul_commutative_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let b = prop.random_tensor(shape, 5.0)
    let ab = unwrap(vt.mul(a, b))
    let ba = unwrap(vt.mul(b, a))
    tensors_close(ab, ba)
  })
  |> should.be_true
}

pub fn prop_add_associative_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let b = prop.random_tensor(shape, 5.0)
    let c = prop.random_tensor(shape, 5.0)
    let left = unwrap(vt.add(unwrap(vt.add(a, b)), c))
    let right = unwrap(vt.add(a, unwrap(vt.add(b, c))))
    tensors_close(left, right)
  })
  |> should.be_true
}

pub fn prop_distributive_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let b = prop.random_tensor(shape, 5.0)
    let c = prop.random_tensor(shape, 5.0)
    // a * (b + c)
    let sum_bc = unwrap(vt.add(b, c))
    let left = unwrap(vt.mul(a, sum_bc))
    // a*b + a*c
    let ab = unwrap(vt.mul(a, b))
    let ac = unwrap(vt.mul(a, c))
    let right = unwrap(vt.add(ab, ac))
    tensors_close(left, right)
  })
  |> should.be_true
}

pub fn prop_add_zero_identity_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let z = vt.zeros(shape)
    let result = unwrap(vt.add(a, z))
    // exact equality is fine: add by 0.0 should not change the value.
    vt.shape(result) == vt.shape(a) && vt.to_list(result) == vt.to_list(a)
  })
  |> should.be_true
}

pub fn prop_mul_one_identity_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let one = vt.ones(shape)
    let result = unwrap(vt.mul(a, one))
    tensors_close(result, a)
  })
  |> should.be_true
}

pub fn prop_negate_double_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let neg_neg = vt.negate(vt.negate(a))
    tensors_close(neg_neg, a)
  })
  |> should.be_true
}

pub fn prop_sum_after_transpose_invariant_test() {
  prop.for_all(iters, fn() {
    let shape = random_2d_shape()
    let a = prop.random_tensor(shape, 5.0)
    let at = unwrap(vt.transpose(a))
    numerics.floats_close(vt.sum(at), vt.sum(a), rtol, atol)
  })
  |> should.be_true
}

// --- Matmul properties ------------------------------------------------------

pub fn prop_matmul_with_identity_test() {
  prop.for_all(iters, fn() {
    // pick a 2-D tensor A of shape [m, n]; use I = eye(n).
    let m = prop.int_in_range(1, 5)
    let n = prop.int_in_range(1, 5)
    let a = prop.random_tensor([m, n], 5.0)
    let i = vt.identity(n)
    let ai = unwrap(vt.matmul(a, i))
    tensors_close(ai, a)
  })
  |> should.be_true
}

pub fn prop_matmul_with_zero_test() {
  prop.for_all(iters, fn() {
    let m = prop.int_in_range(1, 5)
    let n = prop.int_in_range(1, 5)
    let p = prop.int_in_range(1, 5)
    let a = prop.random_tensor([m, n], 5.0)
    let z = vt.zeros([n, p])
    let result = unwrap(vt.matmul(a, z))
    let expected = vt.zeros([m, p])
    tensors_close(result, expected)
  })
  |> should.be_true
}

pub fn prop_matmul_associative_when_compatible_test() {
  // Hardest property — has to sample four chained dims so that
  //   A: [m,n]  B: [n,p]  C: [p,q]  are mutually compatible.
  prop.for_all(iters, fn() {
    let m = prop.int_in_range(1, 4)
    let n = prop.int_in_range(1, 4)
    let p = prop.int_in_range(1, 4)
    let q = prop.int_in_range(1, 4)
    let a = prop.random_tensor([m, n], 3.0)
    let b = prop.random_tensor([n, p], 3.0)
    let c = prop.random_tensor([p, q], 3.0)
    // (A @ B) @ C
    let ab = unwrap(vt.matmul(a, b))
    let left = unwrap(vt.matmul(ab, c))
    // A @ (B @ C)
    let bc = unwrap(vt.matmul(b, c))
    let right = unwrap(vt.matmul(a, bc))
    tensors_close(left, right)
  })
  |> should.be_true
}

// --- Activations ------------------------------------------------------------

pub fn prop_softmax_sums_to_one_test() {
  prop.for_all(iters, fn() {
    let n = prop.int_in_range(1, 8)
    let x = prop.random_tensor([n], 5.0)
    let sm = unwrap(vt.softmax(x, 0))
    let total = vt.sum(sm)
    numerics.floats_close(total, 1.0, rtol, atol)
  })
  |> should.be_true
}

pub fn prop_softmax_invariant_under_shift_test() {
  prop.for_all(iters, fn() {
    let n = prop.int_in_range(1, 8)
    let x = prop.random_tensor([n], 3.0)
    let c = prop.float_in_range(-2.0, 2.0)
    let sm_x = unwrap(vt.softmax(x, 0))
    let sm_xc = unwrap(vt.softmax(vt.add_scalar(x, c), 0))
    // shift invariance is exact in math but floats are not; allow a bit more
    // headroom because exp can stretch the error.
    numerics.lists_close(vt.to_list(sm_xc), vt.to_list(sm_x), 1.0e-4, 1.0e-6)
  })
  |> should.be_true
}

pub fn prop_sigmoid_in_unit_interval_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let x = prop.random_tensor(shape, 5.0)
    let y = vt.sigmoid(x)
    list.all(vt.to_list(y), fn(v) { v >=. 0.0 && v <=. 1.0 })
  })
  |> should.be_true
}

pub fn prop_relu_non_negative_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let x = prop.random_tensor(shape, 5.0)
    let y = vt.relu(x)
    list.all(vt.to_list(y), fn(v) { v >=. 0.0 })
  })
  |> should.be_true
}

// --- Shape preservation -----------------------------------------------------

pub fn prop_reshape_preserves_size_test() {
  prop.for_all(iters, fn() {
    let shape = prop.random_shape(prop.int_in_range(1, 3), 1, 5)
    let a = prop.random_tensor(shape, 5.0)
    let total = vt.size(a)
    // flatten preserves total size; reshape back to original preserves it too.
    let flat = vt.flatten(a)
    let reshaped = unwrap(vt.reshape(flat, shape))
    vt.size(flat) == total && vt.size(reshaped) == total
  })
  |> should.be_true
}

pub fn prop_layer_norm_preserves_shape_test() {
  prop.for_all(iters, fn() {
    // LayerNorm normalizes the last dimension; pick [batch, features].
    let batch = prop.int_in_range(1, 4)
    let features = prop.int_in_range(2, 6)
    let x = prop.random_tensor([batch, features], 3.0)
    let layer = vt.layer_norm_init(features)
    let y = unwrap(vt.layer_norm_forward(layer, x))
    vt.shape(y) == vt.shape(x)
  })
  |> should.be_true
}

// --- Sanity check: harness builds correctly ---------------------------------

pub fn prop_harness_for_all_short_circuits_test() {
  // A failing body should make `for_all` return False.
  let always_false = prop.for_all(5, fn() { False })
  always_false |> should.be_false

  // A passing body returns True over many iterations.
  let always_true = prop.for_all(5, fn() { True })
  always_true |> should.be_true
}
