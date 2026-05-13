//// Tests for `viva_tensor/nn/init`.
////
//// These tests are statistical because the underlying PRNG is
//// non-deterministic. Sample sizes and tolerances are chosen to keep
//// flake probability negligible while still catching gross regressions.

import gleam/float
import gleam/int
import gleam/list
import gleeunit/should
import support/numerics.{floats_close, lists_close}
import viva_tensor as t
import viva_tensor/nn/init
import viva_tensor/tensor

// =============================================================================
// uniform
// =============================================================================

pub fn uniform_in_range_test() {
  let low = -0.5
  let high = 1.5
  let xs = tensor.to_list(init.uniform([50], low, high))

  list.length(xs) |> should.equal(50)

  list.all(xs, fn(x) { x >=. low && x <. high }) |> should.be_true
}

// =============================================================================
// normal
// =============================================================================

pub fn normal_approx_mean_test() {
  let target_mean = 1.0
  let std = 0.5
  let xs = tensor.to_list(init.normal([1000], target_mean, std))

  let n = list.length(xs)
  n |> should.equal(1000)

  let sum = list.fold(xs, 0.0, fn(acc, x) { acc +. x })
  let mean = sum /. int.to_float(n)
  // 1000 samples of std=0.5 -> standard error = 0.5/sqrt(1000) ~= 0.016.
  // Tolerance of 0.2 is ~12 sigma: borderline impossible to flake.
  floats_close(mean, target_mean, 0.0, 0.2) |> should.be_true
}

// =============================================================================
// truncated_normal
// =============================================================================

pub fn truncated_normal_in_range_test() {
  let a = -1.0
  let b = 1.0
  let xs = tensor.to_list(init.truncated_normal([200], 0.0, 0.5, a, b))

  list.length(xs) |> should.equal(200)
  list.all(xs, fn(x) { x >=. a && x <=. b }) |> should.be_true
}

// =============================================================================
// xavier_uniform / xavier_normal
// =============================================================================

pub fn xavier_uniform_test() {
  let fan_in = 32
  let fan_out = 16
  let w = init.xavier_uniform(fan_in, fan_out)

  tensor.shape(w) |> should.equal([fan_in, fan_out])

  let assert Ok(bound) =
    float.square_root(6.0 /. int.to_float(fan_in + fan_out))
  let xs = tensor.to_list(w)
  list.length(xs) |> should.equal(fan_in * fan_out)
  list.all(xs, fn(x) { x >=. 0.0 -. bound && x <. bound }) |> should.be_true
}

pub fn xavier_normal_test() {
  let fan_in = 64
  let fan_out = 64
  let w = init.xavier_normal(fan_in, fan_out)

  tensor.shape(w) |> should.equal([fan_in, fan_out])

  let assert Ok(target_std) =
    float.square_root(2.0 /. int.to_float(fan_in + fan_out))
  let xs = tensor.to_list(w)
  let n = list.length(xs)
  let mean = list.fold(xs, 0.0, fn(acc, x) { acc +. x }) /. int.to_float(n)
  let var =
    list.fold(xs, 0.0, fn(acc, x) {
      let d = x -. mean
      acc +. d *. d
    })
    /. int.to_float(n)
  let assert Ok(observed_std) = float.square_root(var)
  // 4096 samples -> std estimate is within ~1% in expectation.
  // 25% tolerance is generous to absorb PRNG jitter without flaking.
  floats_close(observed_std, target_std, 0.25, 0.0) |> should.be_true
}

// =============================================================================
// kaiming_uniform / kaiming_normal
// =============================================================================

pub fn kaiming_uniform_test() {
  let fan_in = 32
  let fan_out = 16
  let gain = init.relu_gain()
  let w = init.kaiming_uniform(fan_in, fan_out, gain)

  tensor.shape(w) |> should.equal([fan_in, fan_out])

  let assert Ok(inner) = float.square_root(3.0 /. int.to_float(fan_in))
  let bound = gain *. inner
  let xs = tensor.to_list(w)
  list.all(xs, fn(x) { x >=. 0.0 -. bound && x <. bound }) |> should.be_true
}

pub fn kaiming_normal_test() {
  let fan_in = 128
  let fan_out = 32
  let gain = init.relu_gain()
  let w = init.kaiming_normal(fan_in, fan_out, gain)

  tensor.shape(w) |> should.equal([fan_in, fan_out])

  let assert Ok(inner) = float.square_root(1.0 /. int.to_float(fan_in))
  let target_std = gain *. inner

  let xs = tensor.to_list(w)
  let n = list.length(xs)
  let mean = list.fold(xs, 0.0, fn(acc, x) { acc +. x }) /. int.to_float(n)
  let var =
    list.fold(xs, 0.0, fn(acc, x) {
      let d = x -. mean
      acc +. d *. d
    })
    /. int.to_float(n)
  let assert Ok(observed_std) = float.square_root(var)
  floats_close(observed_std, target_std, 0.3, 0.0) |> should.be_true
}

// =============================================================================
// orthogonal
// =============================================================================

pub fn orthogonal_columns_orthonormal_test() {
  // rows >= cols -> Q has orthonormal columns -> Q^T @ Q = I_cols
  let rows = 6
  let cols = 3
  let assert Ok(q) = init.orthogonal(rows, cols, 1.0)
  tensor.shape(q) |> should.equal([rows, cols])

  let assert Ok(qt) = t.transpose(q)
  let assert Ok(qtq) = t.matmul(qt, q)

  let identity =
    list.range(0, cols - 1)
    |> list.flat_map(fn(i) {
      list.range(0, cols - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  lists_close(tensor.to_list(qtq), identity, 1.0e-6, 1.0e-8)
  |> should.be_true
}

pub fn orthogonal_rows_orthonormal_test() {
  // rows < cols -> Q has orthonormal rows -> Q @ Q^T = I_rows
  let rows = 3
  let cols = 6
  let assert Ok(q) = init.orthogonal(rows, cols, 1.0)
  tensor.shape(q) |> should.equal([rows, cols])

  let assert Ok(qt) = t.transpose(q)
  let assert Ok(qqt) = t.matmul(q, qt)

  let identity =
    list.range(0, rows - 1)
    |> list.flat_map(fn(i) {
      list.range(0, rows - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  lists_close(tensor.to_list(qqt), identity, 1.0e-6, 1.0e-8)
  |> should.be_true
}

// =============================================================================
// identity
// =============================================================================

pub fn identity_test() {
  let n = 4
  let id = init.identity(n)
  tensor.shape(id) |> should.equal([n, n])

  let expected =
    list.range(0, n - 1)
    |> list.flat_map(fn(i) {
      list.range(0, n - 1)
      |> list.map(fn(j) {
        case i == j {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  tensor.to_list(id) |> should.equal(expected)
}

// =============================================================================
// gain helpers
// =============================================================================

pub fn relu_gain_test() {
  let assert Ok(expected) = float.square_root(2.0)
  floats_close(init.relu_gain(), expected, 0.0, 1.0e-12) |> should.be_true
}
