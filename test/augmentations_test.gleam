//// Tests for `viva_tensor/vision/augmentations`. Six tests in line with
//// the task spec: ColorJitter zero-strength, shape preservation, MixUp
//// degeneracy when alpha → 0, MixUp/CutMix shape, CutMix zero-box no-op.

import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/tensor.{type Tensor, Tensor}

pub fn main() -> Nil {
  gleeunit.main()
}

// --- Helpers ----------------------------------------------------------------

fn sample_rgb_batch() -> Tensor {
  // Shape [B=2, C=3, H=2, W=2]. Distinct, deterministic, non-symmetric values
  // so that hue rotation and saturation actually move things around.
  Tensor(
    data: [
      // sample 0: R, G, B planes
      0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.05, 0.15, 0.25,
      // sample 1
      0.2, 0.4, 0.6, 0.8, 0.1, 0.3, 0.5, 0.7, 0.9, 0.6, 0.3, 0.1,
    ],
    shape: [2, 3, 2, 2],
  )
}

// --- ColorJitter ------------------------------------------------------------

pub fn color_jitter_zero_strength_test() {
  let image = sample_rgb_batch()
  let config = t.color_jitter_init(0.0, 0.0, 0.0, 0.0)
  case t.color_jitter_forward(config, image) {
    Ok(result) -> {
      t.shape(result) |> should.equal([2, 3, 2, 2])
      numerics.lists_close(
        t.to_list(result),
        t.to_list(image),
        0.000_000_1,
        0.000_000_1,
      )
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn color_jitter_shape_test() {
  let image = sample_rgb_batch()
  let config = t.color_jitter_init(0.2, 0.2, 0.2, 0.1)
  case t.color_jitter_forward(config, image) {
    Ok(result) -> {
      t.shape(result) |> should.equal([2, 3, 2, 2])
      list.length(t.to_list(result)) |> should.equal(24)
    }
    Error(_) -> should.fail()
  }
}

// --- MixUp ------------------------------------------------------------------

pub fn mixup_lambda_one_test() {
  // alpha = 0 is treated as the degenerate Beta where λ = 1.0, so the
  // mixed image must equal the original byte-for-byte.
  let image = sample_rgb_batch()
  let labels = t.from_list([0.0, 1.0])
  case t.mixup(image, labels, 2, 0.0) {
    Ok(#(mixed_images, mixed_labels)) -> {
      t.shape(mixed_images) |> should.equal([2, 3, 2, 2])
      t.shape(mixed_labels) |> should.equal([2, 2])
      numerics.lists_close(
        t.to_list(mixed_images),
        t.to_list(image),
        0.000_000_1,
        0.000_000_1,
      )
      |> should.be_true()
      // Labels should be the pristine one-hot rows for class 0 then class 1.
      numerics.lists_close(
        t.to_list(mixed_labels),
        [1.0, 0.0, 0.0, 1.0],
        0.000_000_1,
        0.000_000_1,
      )
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn mixup_shape_test() {
  let image = sample_rgb_batch()
  let labels = t.from_list([0.0, 1.0])
  case t.mixup(image, labels, 3, 0.4) {
    Ok(#(mixed_images, mixed_labels)) -> {
      t.shape(mixed_images) |> should.equal([2, 3, 2, 2])
      t.shape(mixed_labels) |> should.equal([2, 3])
    }
    Error(_) -> should.fail()
  }
}

// --- CutMix -----------------------------------------------------------------

pub fn cutmix_shape_test() {
  let image = sample_rgb_batch()
  let labels = t.from_list([0.0, 1.0])
  case t.cutmix(image, labels, 4, 0.5) {
    Ok(#(mixed_images, mixed_labels)) -> {
      t.shape(mixed_images) |> should.equal([2, 3, 2, 2])
      t.shape(mixed_labels) |> should.equal([2, 4])
    }
    Error(_) -> should.fail()
  }
}

pub fn cutmix_zero_box_test() {
  // alpha = 0 ⇒ sample_beta returns 1.0 ⇒ cut_ratio = sqrt(1 - 1) = 0
  // ⇒ box has zero area ⇒ images are returned unchanged.
  let image = sample_rgb_batch()
  let labels = t.from_list([0.0, 1.0])
  case t.cutmix(image, labels, 2, 0.0) {
    Ok(#(mixed_images, _mixed_labels)) -> {
      t.shape(mixed_images) |> should.equal([2, 3, 2, 2])
      numerics.lists_close(
        t.to_list(mixed_images),
        t.to_list(image),
        0.000_000_1,
        0.000_000_1,
      )
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}
