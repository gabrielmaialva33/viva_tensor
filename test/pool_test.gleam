//// Tests for `viva_tensor/nn/pool` — Dropout, MaxPool1d/AvgPool1d,
//// AdaptiveAvgPool2d/1d, and Upsample (nearest + bilinear) forward passes.

import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/nn/pool
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

fn tensor_with_shape(data: List(Float), shape: List(Int)) -> tensor.Tensor {
  let assert Ok(t) = tensor.reshape(tensor.from_list(data), shape)
  t
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

pub fn dropout_eval_test() {
  // training = False: pure passthrough regardless of p.
  let layer = pool.dropout_init(0.5)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  let out = pool.dropout_forward(layer, x, False)
  tensor.shape(out) |> should.equal([1, 1, 4])
  numerics.lists_close(
    tensor.to_list(out),
    [1.0, 2.0, 3.0, 4.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn dropout_p_zero_train_test() {
  // p = 0.0 in training: passthrough, no scaling.
  let layer = pool.dropout_init(0.0)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  let out = pool.dropout_forward(layer, x, True)
  tensor.shape(out) |> should.equal([1, 1, 4])
  numerics.lists_close(
    tensor.to_list(out),
    [1.0, 2.0, 3.0, 4.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn dropout_p_one_train_test() {
  // p = 1.0 in training: every element zeroed.
  // Edge case: inverted-dropout scale factor 1 / (1 - p) is undefined here,
  // so we short-circuit to zeros (matches PyTorch).
  let layer = pool.dropout_init(1.0)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  let out = pool.dropout_forward(layer, x, True)
  tensor.shape(out) |> should.equal([1, 1, 4])
  numerics.lists_close(
    tensor.to_list(out),
    [0.0, 0.0, 0.0, 0.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn dropout_scaling_test() {
  // With p = 0.5 and training = True, retained values are scaled by
  // 1 / (1 - 0.5) = 2.0; dropped values are 0.0. So every element is either
  // 0.0 or 2x its original. We don't seed RNG, so we just check the
  // distribution of outputs over a larger tensor.
  let layer = pool.dropout_init(0.5)
  let xs = list.repeat(1.0, 512)
  let x = tensor_with_shape(xs, [1, 1, 512])
  let out = pool.dropout_forward(layer, x, True)
  tensor.shape(out) |> should.equal([1, 1, 512])
  // Every output element must be either 0.0 or 2.0 (1.0 * 1/0.5).
  tensor.to_list(out)
  |> list.all(fn(v) {
    numerics.floats_close(v, 0.0, 1.0e-9, 1.0e-9)
    || numerics.floats_close(v, 2.0, 1.0e-9, 1.0e-9)
  })
  |> should.be_true
}

// ---------------------------------------------------------------------------
// MaxPool1d
// ---------------------------------------------------------------------------

pub fn max_pool_1d_basic_test() {
  // [1,1,4] = [1, 2, 3, 4], kernel = 2, stride = 2 -> [1, 1, 2] = [2, 4]
  let cfg = pool.MaxPool1dConfig(kernel_size: 2, stride: 2, padding: 0)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  let assert Ok(out) = pool.max_pool_1d_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 2])
  numerics.lists_close(tensor.to_list(out), [2.0, 4.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn max_pool_1d_with_padding_test() {
  // [1, 2, 3, 4, 5], kernel = 3, stride = 1, padding = 1
  // padded: [0, 1, 2, 3, 4, 5, 0]
  // windows: max(0,1,2)=2, max(1,2,3)=3, max(2,3,4)=4, max(3,4,5)=5, max(4,5,0)=5
  let cfg = pool.MaxPool1dConfig(kernel_size: 3, stride: 1, padding: 1)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])
  let assert Ok(out) = pool.max_pool_1d_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 5])
  numerics.lists_close(
    tensor.to_list(out),
    [2.0, 3.0, 4.0, 5.0, 5.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

// ---------------------------------------------------------------------------
// AvgPool1d
// ---------------------------------------------------------------------------

pub fn avg_pool_1d_test() {
  // [1, 2, 3, 4], kernel = 2, stride = 2 -> [(1+2)/2, (3+4)/2] = [1.5, 3.5]
  let cfg = pool.AvgPool1dConfig(kernel_size: 2, stride: 2, padding: 0)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 4])
  let assert Ok(out) = pool.avg_pool_1d_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 2])
  numerics.lists_close(tensor.to_list(out), [1.5, 3.5], 1.0e-9, 1.0e-9)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

pub fn adaptive_avg_pool_2d_test() {
  // [1,1,4,4] -> [1,1,2,2]: each output cell is the mean of a 2x2 block.
  //  1  2 |  3  4
  //  5  6 |  7  8
  //  ----+-----
  //  9 10 | 11 12
  // 13 14 | 15 16
  // means: (1+2+5+6)/4=3.5, (3+4+7+8)/4=5.5,
  //        (9+10+13+14)/4=11.5, (11+12+15+16)/4=13.5
  let cfg = pool.AdaptiveAvgPool2dConfig(output_h: 2, output_w: 2)
  let x =
    tensor_with_shape(
      [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        14.0, 15.0, 16.0,
      ],
      [1, 1, 4, 4],
    )
  let assert Ok(out) = pool.adaptive_avg_pool_2d_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 2, 2])
  numerics.lists_close(
    tensor.to_list(out),
    [3.5, 5.5, 11.5, 13.5],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn adaptive_avg_pool_2d_identity_test() {
  // output_h == input_h and output_w == input_w: each cell is its own window
  // (size 1) so the output equals the input.
  let cfg = pool.AdaptiveAvgPool2dConfig(output_h: 3, output_w: 3)
  let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
  let x = tensor_with_shape(data, [1, 1, 3, 3])
  let assert Ok(out) = pool.adaptive_avg_pool_2d_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 3, 3])
  numerics.lists_close(tensor.to_list(out), data, 1.0e-9, 1.0e-9)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

pub fn upsample_nearest_test() {
  // [1,1,2,2] with scale=2 -> [1,1,4,4]; nearest-neighbor replicates each
  // input pixel into a 2x2 block.
  // input:    out:
  //  1 2      1 1 2 2
  //  3 4      1 1 2 2
  //           3 3 4 4
  //           3 3 4 4
  let cfg = pool.UpsampleConfig(scale_factor: 2, mode: pool.Nearest)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
  let assert Ok(out) = pool.upsample_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 4, 4])
  numerics.lists_close(
    tensor.to_list(out),
    [
      1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0,
      4.0,
    ],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn upsample_bilinear_basic_test() {
  // 2x2 -> 4x4 bilinear with align_corners=False:
  //   src coord for output i: (i + 0.5) / 2 - 0.5
  //   i=0 -> -0.25  -> clamp to 0
  //   i=1 ->  0.25
  //   i=2 ->  0.75
  //   i=3 ->  1.25  -> clamp to 1
  // Corners replicate the input corner values; interior pixels interpolate
  // smoothly between them. Verify a couple of specific cells.
  let cfg = pool.UpsampleConfig(scale_factor: 2, mode: pool.Bilinear)
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
  let assert Ok(out) = pool.upsample_forward(cfg, x)
  tensor.shape(out) |> should.equal([1, 1, 4, 4])
  let values = tensor.to_list(out)
  // Corner [0,0] -> input [0,0] = 1.0
  let assert Ok(c00) = list_at(values, 0)
  numerics.floats_close(c00, 1.0, 1.0e-9, 1.0e-9) |> should.be_true
  // Corner [0,3] -> input [0,1] = 2.0
  let assert Ok(c03) = list_at(values, 3)
  numerics.floats_close(c03, 2.0, 1.0e-9, 1.0e-9) |> should.be_true
  // Corner [3,0] -> input [1,0] = 3.0
  let assert Ok(c30) = list_at(values, 12)
  numerics.floats_close(c30, 3.0, 1.0e-9, 1.0e-9) |> should.be_true
  // Corner [3,3] -> input [1,1] = 4.0
  let assert Ok(c33) = list_at(values, 15)
  numerics.floats_close(c33, 4.0, 1.0e-9, 1.0e-9) |> should.be_true
  // Cell [1,1] -> src (0.25, 0.25): 0.75 * (0.75*1 + 0.25*2) + 0.25 * (0.75*3 + 0.25*4) = 1.75
  let assert Ok(c11) = list_at(values, 5)
  numerics.floats_close(c11, 1.75, 1.0e-9, 1.0e-9) |> should.be_true
  // Cell [1,2] -> src (0.25, 0.75): 0.75 * (0.25*1 + 0.75*2) + 0.25 * (0.25*3 + 0.75*4) = 2.25
  let assert Ok(c12) = list_at(values, 6)
  numerics.floats_close(c12, 2.25, 1.0e-9, 1.0e-9) |> should.be_true
  // Sanity: every interior value strictly between min and max input
  list.all(values, fn(v) { v >=. 1.0 -. 1.0e-9 && v <=. 4.0 +. 1.0e-9 })
  |> should.be_true
}

fn list_at(xs: List(Float), i: Int) -> Result(Float, Nil) {
  case xs, i {
    [], _ -> Error(Nil)
    [x, ..], 0 -> Ok(x)
    [_, ..rest], n -> list_at(rest, n - 1)
  }
}
