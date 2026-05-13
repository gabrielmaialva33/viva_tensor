//// Tests for `viva_tensor/nn/norm`.

import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as vt
import viva_tensor/core/error.{InvalidShape, ShapeMismatch}
import viva_tensor/nn/norm
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// Default tolerance — normalization sums-of-squares accumulate fp noise.
const rtol: Float = 1.0e-5

const atol: Float = 1.0e-6

// ----- LayerNorm -----------------------------------------------------------

pub fn layer_norm_init_test() {
  let layer = norm.layer_norm_init(4)
  tensor.shape(layer.scale) |> should.equal([4])
  tensor.shape(layer.bias) |> should.equal([4])
  tensor.to_list(layer.scale) |> should.equal([1.0, 1.0, 1.0, 1.0])
  tensor.to_list(layer.bias) |> should.equal([0.0, 0.0, 0.0, 0.0])
  numerics.floats_close(layer.eps, 1.0e-5, rtol, atol) |> should.be_true
}

pub fn layer_norm_init_with_eps_test() {
  let layer = norm.layer_norm_init_with_eps(3, 1.0e-3)
  numerics.floats_close(layer.eps, 1.0e-3, rtol, atol) |> should.be_true
}

pub fn layer_norm_forward_test() {
  // Input row [1, 2, 3, 4]: mean = 2.5, var = 1.25.
  // Output should have mean ~ 0 and variance ~ 1 along the last axis.
  let layer = norm.layer_norm_init(4)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0]])
  let assert Ok(y) = norm.layer_norm_forward(layer, x)
  let data = tensor.to_list(y)
  // Check mean ~ 0.
  let mean = sum(data) /. 4.0
  numerics.floats_close(mean, 0.0, rtol, atol) |> should.be_true
  // Check variance ~ 1.
  let var =
    list.fold(data, 0.0, fn(acc, v) {
      let d = v -. mean
      acc +. d *. d
    })
    /. 4.0
  numerics.floats_close(var, 1.0, rtol, 1.0e-4) |> should.be_true
  // Output preserves input shape.
  tensor.shape(y) |> should.equal([1, 4])
}

pub fn layer_norm_shape_error_test() {
  let layer = norm.layer_norm_init(4)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0]])
  norm.layer_norm_forward(layer, x)
  |> should.equal(Error(ShapeMismatch(expected: [4], got: [3])))
}

// ----- RMSNorm -------------------------------------------------------------

pub fn rms_norm_init_test() {
  let layer = norm.rms_norm_init(4)
  tensor.shape(layer.scale) |> should.equal([4])
  tensor.to_list(layer.scale) |> should.equal([1.0, 1.0, 1.0, 1.0])
  numerics.floats_close(layer.eps, 1.0e-6, rtol, atol) |> should.be_true
}

pub fn rms_norm_forward_test() {
  // mean(x^2) for [1,2,3,4] = 7.5
  // rms = sqrt(7.5 + 1e-6)
  // output_i = x_i / rms
  let layer = norm.rms_norm_init(4)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0]])
  let assert Ok(y) = norm.rms_norm_forward(layer, x)
  let data = tensor.to_list(y)
  // Expected rms ~ sqrt(7.5) ~ 2.738612787525831.
  let rms = 2.738612787525831
  let expected = [
    1.0 /. rms,
    2.0 /. rms,
    3.0 /. rms,
    4.0 /. rms,
  ]
  numerics.lists_close(data, expected, rtol, atol) |> should.be_true
  tensor.shape(y) |> should.equal([1, 4])
}

// ----- BatchNorm1d ---------------------------------------------------------

pub fn batch_norm_training_test() {
  // Two training steps on the same batch update running_mean each step.
  // batch_mean over axis 0 of [[1,2],[3,4]] = [2, 3]; batch_var = [1, 1].
  // running_mean: 0 -> 0.1*[2,3] = [0.2, 0.3]
  //                -> 0.9*[0.2,0.3] + 0.1*[2,3] = [0.38, 0.57]
  let layer = norm.batch_norm_1d_init(2)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(#(layer1, _)) = norm.batch_norm_1d_forward(layer, x, True)
  let after1 = tensor.to_list(layer1.running_mean)
  numerics.lists_close(after1, [0.2, 0.3], rtol, atol)
  |> should.be_true

  let assert Ok(#(layer2, _)) = norm.batch_norm_1d_forward(layer1, x, True)
  let after2 = tensor.to_list(layer2.running_mean)
  numerics.lists_close(after2, [0.38, 0.57], rtol, atol)
  |> should.be_true

  // running_var with batch_var = 1: (1-0.1)*1 + 0.1*1 = 1.0 — stays 1.
  let var2 = tensor.to_list(layer2.running_var)
  numerics.lists_close(var2, [1.0, 1.0], rtol, atol) |> should.be_true
}

pub fn batch_norm_eval_test() {
  // Pre-seed running stats by doing one training pass, then run eval and
  // confirm running stats are unchanged.
  let init = norm.batch_norm_1d_init(2)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(#(trained, _)) = norm.batch_norm_1d_forward(init, x, True)
  let before_mean = tensor.to_list(trained.running_mean)
  let before_var = tensor.to_list(trained.running_var)

  let assert Ok(#(after_eval, _y)) =
    norm.batch_norm_1d_forward(trained, x, False)
  tensor.to_list(after_eval.running_mean) |> should.equal(before_mean)
  tensor.to_list(after_eval.running_var) |> should.equal(before_var)
}

pub fn batch_norm_eval_uses_running_stats_test() {
  // Eval normalizes using running stats. With fresh init
  // (running_mean=0, running_var=1, eps=1e-5), eval on [[1,2],[3,4]] gives
  // y = (x - 0) / sqrt(1 + 1e-5) * 1 + 0 ~ x.
  let layer = norm.batch_norm_1d_init(2)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(#(_, y)) = norm.batch_norm_1d_forward(layer, x, False)
  let data = tensor.to_list(y)
  numerics.lists_close(data, [1.0, 2.0, 3.0, 4.0], 1.0e-4, 1.0e-4)
  |> should.be_true
}

// ----- GroupNorm -----------------------------------------------------------

pub fn group_norm_init_test() {
  let layer = norm.group_norm_init(2, 4)
  layer.num_groups |> should.equal(2)
  tensor.shape(layer.scale) |> should.equal([4])
  tensor.shape(layer.bias) |> should.equal([4])
}

pub fn group_norm_forward_test() {
  // num_groups=2, num_channels=4, input shape [1,4] = [[1,2,3,4]].
  // Group 0 = {1,2}: mean=1.5, var=0.25, denom=sqrt(0.25+1e-5) ~ 0.50000999...
  // Group 1 = {3,4}: mean=3.5, var=0.25
  // Output approx [-1, 1, -1, 1].
  let layer = norm.group_norm_init(2, 4)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0]])
  let assert Ok(y) = norm.group_norm_forward(layer, x)
  let data = tensor.to_list(y)
  numerics.lists_close(data, [-1.0, 1.0, -1.0, 1.0], 1.0e-4, 1.0e-4)
  |> should.be_true
  tensor.shape(y) |> should.equal([1, 4])
}

pub fn group_norm_divisibility_error_test() {
  let layer = norm.group_norm_init(2, 5)
  let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0, 5.0]])
  norm.group_norm_forward(layer, x)
  |> should.equal(
    Error(InvalidShape("group_norm: channels (5) not divisible by groups (2)")),
  )
}

// ----- Re-export sanity check ---------------------------------------------

pub fn facade_reexports_test() {
  // Sanity check the viva_tensor facade re-exports.
  let layer = vt.layer_norm_init(2)
  tensor.shape(layer.scale) |> should.equal([2])
  let _rms = vt.rms_norm_init(2)
  let _bn = vt.batch_norm_1d_init(2)
  let _gn = vt.group_norm_init(1, 2)
  Nil
}

// ----- helpers -------------------------------------------------------------

fn sum(xs: List(Float)) -> Float {
  list.fold(xs, 0.0, fn(acc, x) { acc +. x })
}
