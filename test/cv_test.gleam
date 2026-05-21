//// Tests for `viva_tensor/nn/cv` — MaxUnpool2d, NMS, ROIAlign,
//// batched matmul and BatchNorm2d.

import gleam/int
import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/nn/cv
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

const rtol: Float = 1.0e-4

const atol: Float = 1.0e-5

fn t4(data: List(Float), shape: List(Int)) -> tensor.Tensor {
  let assert Ok(t) = tensor.reshape(tensor.from_list(data), shape)
  t
}

// ---------------------------------------------------------------------------
// MaxPool2d-with-indices / MaxUnpool2d
// ---------------------------------------------------------------------------

pub fn max_pool_2d_with_indices_test() {
  // [1, 1, 2, 2] input [[1, 3], [2, 4]] with kernel=2, stride=2 → [[4]],
  // indices [[3]] (flat index for value 4 in row-major [H, W]).
  let x = t4([1.0, 3.0, 2.0, 4.0], [1, 1, 2, 2])
  let assert Ok(#(pooled, indices)) = cv.max_pool_2d_with_indices(x, 2, 2, 0)
  tensor.shape(pooled) |> should.equal([1, 1, 1, 1])
  tensor.shape(indices) |> should.equal([1, 1, 1, 1])
  numerics.lists_close(tensor.to_list(pooled), [4.0], rtol, atol)
  |> should.be_true
  numerics.lists_close(tensor.to_list(indices), [3.0], rtol, atol)
  |> should.be_true
}

pub fn max_unpool_2d_test() {
  // Roundtrip: pool then unpool. Output should have the max placed at the
  // stored index and zeros everywhere else.
  let x = t4([1.0, 3.0, 2.0, 4.0], [1, 1, 2, 2])
  let assert Ok(#(pooled, indices)) = cv.max_pool_2d_with_indices(x, 2, 2, 0)
  let cfg = cv.MaxUnpool2dConfig(kernel_size: 2, stride: 2, padding: 0)
  let assert Ok(unpooled) =
    cv.max_unpool_2d_forward(cfg, pooled, indices, #(2, 2))
  tensor.shape(unpooled) |> should.equal([1, 1, 2, 2])
  // Value 4.0 sits at flat index 3 (row 1, col 1); everywhere else = 0.
  numerics.lists_close(
    tensor.to_list(unpooled),
    [0.0, 0.0, 0.0, 4.0],
    rtol,
    atol,
  )
  |> should.be_true
}

// ---------------------------------------------------------------------------
// NMS
// ---------------------------------------------------------------------------

pub fn nms_basic_test() {
  // 3 boxes:
  //   0: [0, 0, 10, 10]   score 0.9
  //   1: [1, 1, 11, 11]   score 0.8   (heavy overlap with 0 → suppressed)
  //   2: [50, 50, 60, 60] score 0.7   (no overlap → kept)
  let assert Ok(boxes) =
    tensor.reshape(
      tensor.from_list([
        0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 11.0, 11.0, 50.0, 50.0, 60.0, 60.0,
      ]),
      [3, 4],
    )
  let scores = tensor.from_list([0.9, 0.8, 0.7])
  let assert Ok(kept) = cv.nms(boxes, scores, 0.5)
  kept |> should.equal([0, 2])
}

pub fn nms_no_overlap_test() {
  // 3 disjoint boxes — all should be kept in descending-score order.
  let assert Ok(boxes) =
    tensor.reshape(
      tensor.from_list([
        0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0,
      ]),
      [3, 4],
    )
  let scores = tensor.from_list([0.1, 0.9, 0.5])
  let assert Ok(kept) = cv.nms(boxes, scores, 0.5)
  // Sorted by descending score: indices 1, 2, 0.
  kept |> should.equal([1, 2, 0])
}

pub fn nms_shape_error_test() {
  // boxes shape [3, 5] is not [N, 4].
  let assert Ok(boxes) =
    tensor.reshape(
      tensor.from_list([
        0.0, 0.0, 1.0, 1.0, 0.0, 2.0, 2.0, 3.0, 3.0, 0.0, 4.0, 4.0, 5.0, 5.0,
        0.0,
      ]),
      [3, 5],
    )
  let scores = tensor.from_list([0.1, 0.9, 0.5])
  case cv.nms(boxes, scores, 0.5) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

// ---------------------------------------------------------------------------
// ROIAlign
// ---------------------------------------------------------------------------

pub fn roi_align_identity_test() {
  // Single ROI covering the whole 4x4 feature map at spatial_scale 1.0 with
  // output 4x4 (sampling_ratio=2). With `aligned=False` semantics, the
  // output is a bilinear resampling whose top-left sample lands in the
  // first half-bin and whose right/bottom edges clamp at the last pixel.
  // Expected values match a hand-traced TorchVision-style reference.
  let data =
    range_int(0, 15)
    |> list.map(fn(i) { int.to_float(i) })
  let feat = t4(data, [1, 1, 4, 4])
  let assert Ok(rois) =
    tensor.reshape(tensor.from_list([0.0, 0.0, 0.0, 4.0, 4.0]), [1, 5])
  let cfg =
    cv.RoiAlignConfig(
      output_h: 4,
      output_w: 4,
      spatial_scale: 1.0,
      sampling_ratio: 2,
    )
  let assert Ok(out) = cv.roi_align(cfg, feat, rois)
  tensor.shape(out) |> should.equal([1, 1, 4, 4])
  let expected = [
    2.5, 3.5, 4.5, 5.0, 6.5, 7.5, 8.5, 9.0, 10.5, 11.5, 12.5, 13.0, 12.5, 13.5,
    14.5, 15.0,
  ]
  numerics.lists_close(tensor.to_list(out), expected, rtol, atol)
  |> should.be_true
}

pub fn roi_align_basic_test() {
  // 1x1 feature map of value 7.0; any ROI inside should return 7.0 everywhere.
  let feat = t4([7.0], [1, 1, 1, 1])
  let assert Ok(rois) =
    tensor.reshape(tensor.from_list([0.0, 0.0, 0.0, 1.0, 1.0]), [1, 5])
  let cfg =
    cv.RoiAlignConfig(
      output_h: 2,
      output_w: 2,
      spatial_scale: 1.0,
      sampling_ratio: 2,
    )
  let assert Ok(out) = cv.roi_align(cfg, feat, rois)
  tensor.shape(out) |> should.equal([1, 1, 2, 2])
  numerics.lists_close(tensor.to_list(out), [7.0, 7.0, 7.0, 7.0], rtol, atol)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// Batched matmul
// ---------------------------------------------------------------------------

pub fn batched_matmul_test() {
  // [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]. Use ones for shape + value sanity:
  // every output element should equal the inner contraction length (4).
  let a = t4(list.repeat(1.0, 24), [2, 3, 4])
  let b = t4(list.repeat(1.0, 40), [2, 4, 5])
  let assert Ok(c) = cv.batched_matmul(a, b)
  tensor.shape(c) |> should.equal([2, 3, 5])
  let expected = list.repeat(4.0, 30)
  numerics.lists_close(tensor.to_list(c), expected, rtol, atol)
  |> should.be_true
}

pub fn batched_matmul_broadcast_test() {
  // [1, 3, 4] @ [2, 4, 5] broadcasts the left batch dim → [2, 3, 5].
  let a = t4(list.repeat(1.0, 12), [1, 3, 4])
  let b = t4(list.repeat(2.0, 40), [2, 4, 5])
  let assert Ok(c) = cv.batched_matmul(a, b)
  tensor.shape(c) |> should.equal([2, 3, 5])
  // sum over k=4 of 1*2 = 8.0 for every output.
  let expected = list.repeat(8.0, 30)
  numerics.lists_close(tensor.to_list(c), expected, rtol, atol)
  |> should.be_true
}

pub fn batched_matmul_shape_error_test() {
  // [2, 3, 4] @ [2, 5, 6] — inner dims 4 vs 5 mismatch.
  let a = t4(list.repeat(1.0, 24), [2, 3, 4])
  let b = t4(list.repeat(1.0, 60), [2, 5, 6])
  case cv.batched_matmul(a, b) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

// ---------------------------------------------------------------------------
// BatchNorm2d
// ---------------------------------------------------------------------------

pub fn batch_norm_2d_training_test() {
  // Use 2 channels, 2 batches, 2x2 spatial. After training-mode normalization
  // the per-channel mean should be ~0 and variance ~1.
  let layer = cv.batch_norm_2d_init(2)
  let data = [
    // batch 0
    1.0, 2.0, 3.0, 4.0,
    // channel 0 (2x2)
    5.0, 6.0, 7.0, 8.0,
    // channel 1
    // batch 1
    2.0, 3.0, 4.0, 5.0,
    // channel 0
    6.0, 7.0, 8.0, 9.0,
    // channel 1
  ]
  let x = t4(data, [2, 2, 2, 2])
  let assert Ok(#(_updated, y)) = cv.batch_norm_2d_forward(layer, x, True)
  tensor.shape(y) |> should.equal([2, 2, 2, 2])

  let out = tensor.to_list(y)
  // Split per channel: positions 0..3 + 8..11 are channel 0, rest channel 1.
  let chan0 = list.append(list.take(out, 4), list.take(list.drop(out, 8), 4))
  let chan1 =
    list.append(
      list.take(list.drop(out, 4), 4),
      list.take(list.drop(out, 12), 4),
    )

  let mean0 = list_mean(chan0)
  let mean1 = list_mean(chan1)
  numerics.floats_close(mean0, 0.0, 1.0e-6, 1.0e-5) |> should.be_true
  numerics.floats_close(mean1, 0.0, 1.0e-6, 1.0e-5) |> should.be_true

  let var0 = list_var(chan0, mean0)
  let var1 = list_var(chan1, mean1)
  // var should be ~1.0 (biased estimator: matches normalization formula).
  numerics.floats_close(var0, 1.0, 1.0e-3, 1.0e-3) |> should.be_true
  numerics.floats_close(var1, 1.0, 1.0e-3, 1.0e-3) |> should.be_true
}

pub fn batch_norm_2d_eval_test() {
  // Eval mode: should use running stats and *not* update them.
  let layer = cv.batch_norm_2d_init(2)
  let x = t4(list.repeat(1.0, 16), [2, 2, 2, 2])
  let assert Ok(#(updated, _y)) = cv.batch_norm_2d_forward(layer, x, False)
  // Running mean still zeros, running var still ones.
  tensor.to_list(updated.running_mean)
  |> should.equal([0.0, 0.0])
  tensor.to_list(updated.running_var)
  |> should.equal([1.0, 1.0])
}

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

fn list_mean(xs: List(Float)) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ -> list.fold(xs, 0.0, fn(acc, x) { acc +. x }) /. int.to_float(n)
  }
}

fn list_var(xs: List(Float), mean: Float) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ ->
      list.fold(xs, 0.0, fn(acc, x) {
        let d = x -. mean
        acc +. d *. d
      })
      /. int.to_float(n)
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
