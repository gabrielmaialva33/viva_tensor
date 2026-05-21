//// Minimal hand-rolled QuickCheck-style harness for property-based tests.
////
//// Tests live next to `test/support/numerics.gleam` and are test-time only.
//// They are NOT part of the published library.
////
//// Non-determinism notice
//// ----------------------
//// `int.random/1` from `gleam_stdlib` is the only RNG primitive available on
//// the BEAM and it is NOT seedable. As a consequence every run of the
//// property tests draws a fresh, unseeded sample. Practical impact:
////
//// * A property test that passes in CI today can still fail tomorrow if a
////   new sample exposes a regression that was always there (this is the
////   *good* kind of flakiness — it surfaces real bugs).
//// * Reproducing a failure may require re-running the test until the same
////   class of input shows up again. Iteration counts are sized to keep that
////   loop short (30–50 iterations per property).
//// * Failures print the offending shape / values inline so triage does not
////   require re-running.
////
//// When `gleam_stdlib` ships a seedable PRNG, swap `int.random` here for the
//// seeded API and add a `--seed` flag at the runner level.

import gleam/float
import gleam/int
import gleam/list
import viva_tensor/tensor.{type Tensor, Tensor}

/// Generate a random integer in `[low, high]` (inclusive).
///
/// Falls back to `low` when `high < low` to keep callers total.
pub fn int_in_range(low: Int, high: Int) -> Int {
  case high < low {
    True -> low
    False -> {
      let span = high - low + 1
      low + int.random(span)
    }
  }
}

/// Generate a random float in `[low, high)`.
///
/// Uses `int.random` over a 1_000_000-bucket grid so we get reasonable
/// coverage without floating-point trickery.
pub fn float_in_range(low: Float, high: Float) -> Float {
  let resolution = 1_000_000
  let r = int.to_float(int.random(resolution)) /. int.to_float(resolution)
  low +. r *. { high -. low }
}

/// Generate a random shape with `rank` dimensions, each in `[min_dim, max_dim]`.
pub fn random_shape(rank: Int, min_dim: Int, max_dim: Int) -> List(Int) {
  case rank <= 0 {
    True -> []
    False ->
      range_int(0, rank - 1)
      |> list.map(fn(_) { int_in_range(min_dim, max_dim) })
  }
}

fn shape_size(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, d) { acc * d })
}

/// Generate a random tensor of the given shape with values in `[-bound, bound]`.
pub fn random_tensor(shape: List(Int), bound: Float) -> Tensor {
  random_tensor_in_range(shape, float.negate(bound), bound)
}

/// Generate a random tensor with values in `[low, high)`.
pub fn random_tensor_in_range(
  shape: List(Int),
  low: Float,
  high: Float,
) -> Tensor {
  let n = shape_size(shape)
  let data =
    range_int(0, n - 1)
    |> list.map(fn(_) { float_in_range(low, high) })
  Tensor(data, shape)
}

/// Run a property `n_iters` times.
///
/// `body` runs in a fresh random state on each iteration. If any iteration
/// returns `False`, `for_all` short-circuits and returns `False`. Callers
/// should pipe the result into `should.be_true` so gleeunit fails loudly.
///
/// Iterations >= 1; otherwise treats as a single pass.
pub fn for_all(n_iters: Int, body: fn() -> Bool) -> Bool {
  let iters = case n_iters < 1 {
    True -> 1
    False -> n_iters
  }
  loop(iters, body)
}

fn loop(remaining: Int, body: fn() -> Bool) -> Bool {
  case remaining {
    0 -> True
    _ ->
      case body() {
        False -> False
        True -> loop(remaining - 1, body)
      }
  }
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
