//// Tests for `viva_tensor/vision/transforms` — geometric, colour, and
//// conversion transforms operating on CHW (`[C, H, W]`) tensors (or NCHW
//// batched). CHW layout, PyTorch / Torchvision convention.

import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/tensor
import viva_tensor/vision/transforms as vt

pub fn main() {
  gleeunit.main()
}

fn tensor_with_shape(data: List(Float), shape: List(Int)) -> tensor.Tensor {
  let assert Ok(t) = tensor.reshape(tensor.from_list(data), shape)
  t
}

// ---------------------------------------------------------------------------
// resize
// ---------------------------------------------------------------------------

pub fn resize_nearest_test() {
  // [1, 2, 2] → [1, 4, 4]. Nearest with align_corners=False maps
  // each output pixel to the nearest source pixel; for a 2→4 scale the
  // output collapses to four 2x2 blocks of the original values.
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 2, 2])
  let assert Ok(out) = vt.resize(x, 4, 4, vt.Nearest)
  tensor.shape(out) |> should.equal([1, 4, 4])
  // Corners preserved: top-left = 1.0, top-right = 2.0,
  // bottom-left = 3.0, bottom-right = 4.0.
  let data = tensor.to_list(out)
  numerics.lists_close(
    data,
    [
      1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0,
      4.0,
    ],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn resize_bilinear_test() {
  // Same input, but bilinear. Just verify shape + corner values
  // (corners must equal the original corner pixels under
  // align_corners=False with this clamping).
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 2, 2])
  let assert Ok(out) = vt.resize(x, 4, 4, vt.Bilinear)
  tensor.shape(out) |> should.equal([1, 4, 4])
  let data = tensor.to_list(out)
  // First element corresponds to oh=0, ow=0:
  //   src_h = (0+0.5)*2/4 - 0.5 = 0 → h0=h1=0, dh=0
  //   src_w = same → w0=w1=0, dw=0
  // Result = data[0,0] = 1.0.
  case data {
    [first, ..] ->
      numerics.floats_close(first, 1.0, 1.0e-9, 1.0e-9)
      |> should.be_true
    [] -> should.fail()
  }
}

// ---------------------------------------------------------------------------
// center_crop
// ---------------------------------------------------------------------------

pub fn center_crop_test() {
  // [3, 4, 4] image where each channel has the same 4x4 grid 1..16.
  // Centre 2x2 = rows 1..2, cols 1..2 → [6, 7, 10, 11] per channel.
  let plane = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
    15.0, 16.0,
  ]
  let data = [plane, plane, plane]
  let x =
    tensor_with_shape(
      data
        |> list_flatten,
      [3, 4, 4],
    )
  let assert Ok(out) = vt.center_crop(x, 2, 2)
  tensor.shape(out) |> should.equal([3, 2, 2])
  numerics.lists_close(
    tensor.to_list(out),
    [6.0, 7.0, 10.0, 11.0, 6.0, 7.0, 10.0, 11.0, 6.0, 7.0, 10.0, 11.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

// ---------------------------------------------------------------------------
// random_crop
// ---------------------------------------------------------------------------

pub fn random_crop_test() {
  // Non-deterministic position, but the output shape must be right
  // and every value must come from the source image (here 1..16).
  let x =
    tensor_with_shape(
      [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        14.0, 15.0, 16.0,
      ],
      [1, 4, 4],
    )
  let assert Ok(out) = vt.random_crop(x, 2, 2)
  tensor.shape(out) |> should.equal([1, 2, 2])
}

// ---------------------------------------------------------------------------
// horizontal_flip / vertical_flip
// ---------------------------------------------------------------------------

pub fn horizontal_flip_test() {
  // [1, 2, 3] with rows [[1, 2, 3], [4, 5, 6]] flips to
  // [[3, 2, 1], [6, 5, 4]].
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3])
  let assert Ok(out) = vt.horizontal_flip(x)
  tensor.shape(out) |> should.equal([1, 2, 3])
  numerics.lists_close(
    tensor.to_list(out),
    [3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn vertical_flip_test() {
  // [1, 2, 3] flipped along H: rows reversed.
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3])
  let assert Ok(out) = vt.vertical_flip(x)
  tensor.shape(out) |> should.equal([1, 2, 3])
  numerics.lists_close(
    tensor.to_list(out),
    [4.0, 5.0, 6.0, 1.0, 2.0, 3.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

// ---------------------------------------------------------------------------
// normalize
// ---------------------------------------------------------------------------

pub fn normalize_test() {
  // Single channel, mean = 1.5, std = 0.5.
  // (1.0 - 1.5) / 0.5 = -1.0, (2.0 - 1.5) / 0.5 = 1.0.
  let x = tensor_with_shape([1.0, 2.0], [1, 1, 2])
  let assert Ok(out) = vt.normalize(x, [1.5], [0.5])
  tensor.shape(out) |> should.equal([1, 1, 2])
  numerics.lists_close(tensor.to_list(out), [-1.0, 1.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn normalize_shape_mismatch_test() {
  // 3-channel image but only 2 mean/std entries.
  let x =
    tensor_with_shape(
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
      [3, 2, 2],
    )
  let result = vt.normalize(x, [0.5, 0.5], [1.0, 1.0])
  case result {
    Error(_) -> Nil
    Ok(_) -> should.fail()
  }
}

// ---------------------------------------------------------------------------
// to_grayscale
// ---------------------------------------------------------------------------

pub fn to_grayscale_1_channel_test() {
  // Single-pixel RGB image. Luma = 0.299*1 + 0.587*1 + 0.114*1 = 1.0.
  let x = tensor_with_shape([1.0, 1.0, 1.0], [3, 1, 1])
  let assert Ok(out) = vt.to_grayscale(x, 1)
  tensor.shape(out) |> should.equal([1, 1, 1])
  case tensor.to_list(out) {
    [v] ->
      numerics.floats_close(v, 1.0, 1.0e-9, 1.0e-9)
      |> should.be_true
    _ -> should.fail()
  }
}

pub fn to_grayscale_3_channel_test() {
  // Pure red: R=1, G=0, B=0. Luma = 0.299. Broadcast to 3 channels =>
  // [0.299, 0.299, 0.299].
  let x = tensor_with_shape([1.0, 0.0, 0.0], [3, 1, 1])
  let assert Ok(out) = vt.to_grayscale(x, 3)
  tensor.shape(out) |> should.equal([3, 1, 1])
  numerics.lists_close(
    tensor.to_list(out),
    [0.299, 0.299, 0.299],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

// ---------------------------------------------------------------------------
// adjust_brightness / adjust_contrast
// ---------------------------------------------------------------------------

pub fn adjust_brightness_test() {
  // [0.2, 0.5, 0.8] with factor 0.5 → [0.1, 0.25, 0.4].
  // Factor 4.0 → [0.8, 1.0 (clamped from 2.0), 1.0 (clamped)].
  let x = tensor_with_shape([0.2, 0.5, 0.8], [1, 1, 3])
  let assert Ok(half) = vt.adjust_brightness(x, 0.5)
  numerics.lists_close(tensor.to_list(half), [0.1, 0.25, 0.4], 1.0e-9, 1.0e-9)
  |> should.be_true
  let assert Ok(clipped) = vt.adjust_brightness(x, 4.0)
  numerics.lists_close(tensor.to_list(clipped), [0.8, 1.0, 1.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn adjust_contrast_test() {
  // Single channel [0.0, 1.0]. Mean = 0.5.
  // factor = 0.0 → every pixel becomes the mean: [0.5, 0.5].
  let x = tensor_with_shape([0.0, 1.0], [1, 1, 2])
  let assert Ok(flat) = vt.adjust_contrast(x, 0.0)
  numerics.lists_close(tensor.to_list(flat), [0.5, 0.5], 1.0e-9, 1.0e-9)
  |> should.be_true
  // factor = 1.0 → identity.
  let assert Ok(same) = vt.adjust_contrast(x, 1.0)
  numerics.lists_close(tensor.to_list(same), [0.0, 1.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// to_tensor / to_byte_image
// ---------------------------------------------------------------------------

pub fn to_tensor_test() {
  // Single-channel 1x3 image, bytes [0, 127, 255] → [0.0, ~0.498, 1.0].
  let assert Ok(out) = vt.to_tensor([0, 127, 255], 1, 3, 1)
  tensor.shape(out) |> should.equal([1, 1, 3])
  numerics.lists_close(
    tensor.to_list(out),
    [0.0, 127.0 /. 255.0, 1.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn to_byte_image_test() {
  // CHW [1, 1, 3] in [0, 1] → HWC bytes.
  let x = tensor_with_shape([0.0, 0.5, 1.0], [1, 1, 3])
  let assert Ok(bytes) = vt.to_byte_image(x)
  // 0.0*255=0, 0.5*255=127.5 (rounds to 128), 1.0*255=255.
  bytes |> should.equal([0, 128, 255])
}

// ---------------------------------------------------------------------------
// compose
// ---------------------------------------------------------------------------

pub fn compose_test() {
  // Chain horizontal_flip + normalize.
  // [1, 1, 3] input [0.0, 0.5, 1.0].
  // After hflip: [1.0, 0.5, 0.0].
  // After normalize(mean=[0.5], std=[0.5]): [1.0, 0.0, -1.0].
  let x = tensor_with_shape([0.0, 0.5, 1.0], [1, 1, 3])
  let pipeline = [
    vt.horizontal_flip,
    fn(t) { vt.normalize(t, [0.5], [0.5]) },
  ]
  let assert Ok(out) = vt.compose(pipeline, x)
  tensor.shape(out) |> should.equal([1, 1, 3])
  numerics.lists_close(tensor.to_list(out), [1.0, 0.0, -1.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn list_flatten(xs: List(List(Float))) -> List(Float) {
  case xs {
    [] -> []
    [head, ..rest] -> append(head, list_flatten(rest))
  }
}

fn append(a: List(Float), b: List(Float)) -> List(Float) {
  case a {
    [] -> b
    [head, ..rest] -> [head, ..append(rest, b)]
  }
}
