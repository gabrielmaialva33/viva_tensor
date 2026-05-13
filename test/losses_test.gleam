//// Forward-pass tests for `viva_tensor/nn/losses`.
////
//// Hand-computed expected values are documented inline above each
//// assertion so a maintainer can sanity-check without re-deriving them.

import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/core/error.{
  InvalidShape, OperandShapeMismatch, RankMismatch, ShapeMismatch,
}
import viva_tensor/nn/losses

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// HELPERS
// =============================================================================

const tol: Float = 1.0e-6

fn assert_close(actual: Float, expected: Float) -> Nil {
  let diff = float.absolute_value(actual -. expected)
  case diff <. tol {
    True -> Nil
    False -> {
      should.equal(actual, expected)
      Nil
    }
  }
}

fn assert_list_close(actual: List(Float), expected: List(Float)) -> Nil {
  list.length(actual) |> should.equal(list.length(expected))
  list.zip(actual, expected)
  |> list.each(fn(pair) {
    let #(a, e) = pair
    assert_close(a, e)
  })
}

// =============================================================================
// MSE
// =============================================================================

pub fn mse_loss_mean_test() {
  // pred=[1,2,3], target=[1.5,2.5,2]
  // sq_err = [0.25, 0.25, 1.0] -> mean = 1.5 / 3 = 0.5
  let pred = t.from_list([1.0, 2.0, 3.0])
  let target = t.from_list([1.5, 2.5, 2.0])
  let assert Ok(loss) = losses.mse_loss(pred, target, losses.ReductionMean)
  t.shape(loss) |> should.equal([1])
  let assert [v] = t.to_list(loss)
  assert_close(v, 0.5)
}

pub fn mse_loss_sum_test() {
  // sq_err = [0.25, 0.25, 1.0] -> sum = 1.5
  let pred = t.from_list([1.0, 2.0, 3.0])
  let target = t.from_list([1.5, 2.5, 2.0])
  let assert Ok(loss) = losses.mse_loss(pred, target, losses.ReductionSum)
  t.shape(loss) |> should.equal([1])
  let assert [v] = t.to_list(loss)
  assert_close(v, 1.5)
}

pub fn mse_loss_none_test() {
  // Per-element keeps shape and values.
  let pred = t.from_list([1.0, 2.0, 3.0])
  let target = t.from_list([1.5, 2.5, 2.0])
  let assert Ok(loss) = losses.mse_loss(pred, target, losses.ReductionNone)
  t.shape(loss) |> should.equal([3])
  assert_list_close(t.to_list(loss), [0.25, 0.25, 1.0])
}

// =============================================================================
// L1
// =============================================================================

pub fn l1_loss_mean_test() {
  // pred=[1,2,3], target=[1.5,1.5,4]
  // abs_err = [0.5, 0.5, 1.0] -> mean = 2/3
  let pred = t.from_list([1.0, 2.0, 3.0])
  let target = t.from_list([1.5, 1.5, 4.0])
  let assert Ok(loss) = losses.l1_loss(pred, target, losses.ReductionMean)
  t.shape(loss) |> should.equal([1])
  let assert [v] = t.to_list(loss)
  assert_close(v, 2.0 /. 3.0)
}

// =============================================================================
// BCE
// =============================================================================

pub fn bce_loss_test() {
  // pred=[0.9, 0.2], target=[1.0, 0.0]
  // loss_0 = -log(0.9)            ≈ 0.10536051565782628
  // loss_1 = -log(1 - 0.2) = -log(0.8) ≈ 0.22314355131420976
  // mean ≈ 0.16425203348601802
  let pred = t.from_list([0.9, 0.2])
  let target = t.from_list([1.0, 0.0])
  let assert Ok(loss) = losses.bce_loss(pred, target, losses.ReductionMean)
  let assert [v] = t.to_list(loss)
  assert_close(v, 0.16425203348601802)
}

pub fn bce_loss_clamp_test() {
  // Without clamping, log(0) would crash. With clamp to [eps, 1-eps] the
  // loss is large but finite.
  let pred = t.from_list([0.0, 1.0])
  let target = t.from_list([1.0, 0.0])
  let assert Ok(loss) = losses.bce_loss(pred, target, losses.ReductionMean)
  let assert [v] = t.to_list(loss)
  // -log(1e-7) ≈ 16.118 on each side, mean ≈ 16.118
  { v >. 10.0 } |> should.be_true()
  { v <. 1.0e6 } |> should.be_true()
}

pub fn bce_loss_invalid_target_test() {
  let pred = t.from_list([0.5, 0.5])
  let target = t.from_list([1.5, -0.5])
  losses.bce_loss(pred, target, losses.ReductionMean)
  |> should.equal(
    Error(InvalidShape("bce_loss: target contains values outside [0, 1]")),
  )
}

// =============================================================================
// Cross-Entropy
// =============================================================================

pub fn cross_entropy_basic_test() {
  // 3 samples, 2 classes; argmax of each row matches target -> small loss.
  // Row 0: [3.0, 0.0], target 0 -> loss = -3 + logsumexp([3,0])
  //   logsumexp ≈ 3 + log(1 + e^-3) ≈ 3.0485873515737583
  //   loss ≈ 0.0485873515737583
  // Row 1: [0.0, 3.0], target 1 -> same ≈ 0.0485873515737583
  // Row 2: [1.0, 2.0], target 1 -> loss = -2 + logsumexp([1,2])
  //   logsumexp = 2 + log(1 + e^-1) ≈ 2.3132616875182228
  //   loss ≈ 0.3132616875182228
  // mean ≈ (0.04858735 + 0.04858735 + 0.31326169) / 3 ≈ 0.13681213
  let assert Ok(logits) = t.from_list2d([[3.0, 0.0], [0.0, 3.0], [1.0, 2.0]])
  let targets = t.from_list([0.0, 1.0, 1.0])
  let assert Ok(loss) =
    losses.cross_entropy_loss(logits, targets, losses.ReductionMean)
  t.shape(loss) |> should.equal([1])
  let assert [v] = t.to_list(loss)
  assert_close(v, 0.13681213089393463)
}

pub fn cross_entropy_shape_error_test() {
  let logits = t.from_list([1.0, 2.0, 3.0])
  let targets = t.from_list([0.0])
  let result = losses.cross_entropy_loss(logits, targets, losses.ReductionMean)
  case result {
    Error(RankMismatch(op, 2, [3])) -> op |> should.equal("cross_entropy_loss")
    _ -> should.fail()
  }
}

pub fn cross_entropy_targets_shape_error_test() {
  let assert Ok(logits) = t.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(targets) = t.from_list2d([[0.0], [1.0]])
  let result = losses.cross_entropy_loss(logits, targets, losses.ReductionMean)
  case result {
    Error(OperandShapeMismatch("cross_entropy_loss", "targets", "[batch]", _)) ->
      Nil
    _ -> {
      should.fail()
      Nil
    }
  }
}

// =============================================================================
// Huber
// =============================================================================

pub fn huber_loss_quadratic_region_test() {
  // |error| < delta -> 0.5 * error^2
  // pred=[0.1, 0.2], target=[0.0, 0.0], delta=1.0
  // errors = [0.1, 0.2] -> losses = [0.005, 0.02] -> mean = 0.0125
  let pred = t.from_list([0.1, 0.2])
  let target = t.from_list([0.0, 0.0])
  let assert Ok(loss) =
    losses.huber_loss(pred, target, 1.0, losses.ReductionMean)
  let assert [v] = t.to_list(loss)
  assert_close(v, 0.0125)
}

pub fn huber_loss_linear_region_test() {
  // |error| >= delta -> delta * (|error| - 0.5 * delta)
  // pred=[5.0], target=[0.0], delta=1.0
  // |error|=5 -> loss = 1.0 * (5.0 - 0.5) = 4.5
  let pred = t.from_list([5.0])
  let target = t.from_list([0.0])
  let assert Ok(loss) =
    losses.huber_loss(pred, target, 1.0, losses.ReductionSum)
  let assert [v] = t.to_list(loss)
  assert_close(v, 4.5)
}

// =============================================================================
// Shared error path
// =============================================================================

pub fn loss_shape_mismatch_test() {
  let pred = t.from_list([1.0, 2.0])
  let target = t.from_list([1.0, 2.0, 3.0])

  losses.mse_loss(pred, target, losses.ReductionMean)
  |> should.equal(Error(ShapeMismatch(expected: [2], got: [3])))

  losses.l1_loss(pred, target, losses.ReductionMean)
  |> should.equal(Error(ShapeMismatch(expected: [2], got: [3])))

  losses.bce_loss(pred, target, losses.ReductionMean)
  |> should.equal(Error(ShapeMismatch(expected: [2], got: [3])))

  losses.huber_loss(pred, target, 1.0, losses.ReductionMean)
  |> should.equal(Error(ShapeMismatch(expected: [2], got: [3])))
}
