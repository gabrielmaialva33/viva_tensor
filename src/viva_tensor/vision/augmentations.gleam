//// Higher-order vision augmentations operating on batches of images.
////
//// Inputs follow the NCHW convention: `[B, C, H, W]` or `[C, H, W]`
//// (single image). All ops are pure Gleam (no NIF, no native fallback).
////
//// The math behind each augmentation lives next to the function it powers.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{
  type TensorError, InvalidShape, OperandShapeMismatch,
}
import viva_tensor/core/ffi
import viva_tensor/tensor.{type Tensor, Tensor}

// =============================================================================
// COLOR JITTER
// =============================================================================

/// Configuration for `color_jitter_forward`.
///
/// Each strength is the half-width of the uniform interval used to draw the
/// per-call factor:
///   - `brightness` ∈ [1 - b, 1 + b]
///   - `contrast`   ∈ [1 - c, 1 + c]
///   - `saturation` ∈ [1 - s, 1 + s]
///   - `hue`        ∈ [-h, h]   (rotation angle in radians)
///
/// A strength of `0.0` disables that adjustment (and the corresponding pass
/// becomes the identity).
pub type ColorJitterConfig {
  ColorJitterConfig(
    brightness: Float,
    contrast: Float,
    saturation: Float,
    hue: Float,
  )
}

/// Construct a `ColorJitterConfig`. Mirrors `torchvision.transforms.ColorJitter`.
pub fn color_jitter_init(
  brightness: Float,
  contrast: Float,
  saturation: Float,
  hue: Float,
) -> ColorJitterConfig {
  ColorJitterConfig(
    brightness: brightness,
    contrast: contrast,
    saturation: saturation,
    hue: hue,
  )
}

/// Apply random brightness/contrast/saturation/hue adjustments to an image
/// tensor with shape `[C, H, W]` or `[B, C, H, W]`.
///
/// The four passes run sequentially and each draws an independent uniform
/// factor from `[1-strength, 1+strength]` (or `[-hue, hue]` for hue).
///
/// - **Brightness**: `x' = x * b` where `b ∈ [1 - brightness, 1 + brightness]`.
/// - **Contrast**:   `x' = (x - mean) * c + mean` where `mean` is the
///   per-image grand mean across all channels/pixels, `c ∈ [1 - contrast,
///   1 + contrast]`.
/// - **Saturation**: `x' = lerp(luma, x, s)` where the BT.601 luma is
///   `Y = 0.299 R + 0.587 G + 0.114 B` and `s ∈ [1 - saturation,
///   1 + saturation]`. `s = 1` keeps the image, `s = 0` collapses to
///   grayscale.
/// - **Hue**: rotate the RGB color vector around the gray axis
///   `n = (1, 1, 1) / sqrt(3)` by `θ ∈ [-hue, hue]` radians, using
///   Rodrigues' rotation formula
///   `R = I cos θ + (1 - cos θ) n nᵀ + sin θ [n]_x`. This preserves the
///   luma direction and shifts hue along the chrominance plane.
///
/// Requires `C = 3` (RGB). If the image already has zero strength on every
/// channel, the input is returned untouched (no random draws).
pub fn color_jitter_forward(
  config: ColorJitterConfig,
  image: Tensor,
) -> Result(Tensor, TensorError) {
  let shape = tensor.shape(image)
  use #(batch, channels, height, width) <- result.try(parse_image_shape(
    "color_jitter_forward",
    shape,
  ))

  case channels == 3 {
    True -> {
      let data = tensor.to_list(image)
      let plane = height * width
      let stride = channels * plane

      let total_strength =
        float.absolute_value(config.brightness)
        +. float.absolute_value(config.contrast)
        +. float.absolute_value(config.saturation)
        +. float.absolute_value(config.hue)

      case total_strength <=. 0.0 {
        True -> Ok(image)
        False -> {
          let processed =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              let start = b * stride
              let r = slice(data, start, plane)
              let g = slice(data, start + plane, plane)
              let b_ch = slice(data, start + 2 * plane, plane)
              let #(r1, g1, b1) =
                apply_brightness(r, g, b_ch, config.brightness)
              let #(r2, g2, b2) = apply_contrast(r1, g1, b1, config.contrast)
              let #(r3, g3, b3) = apply_saturation(r2, g2, b2, config.saturation)
              let #(r4, g4, b4) = apply_hue(r3, g3, b3, config.hue)
              list.append(r4, list.append(g4, b4))
            })
          Ok(Tensor(data: processed, shape: shape))
        }
      }
    }
    False ->
      Error(OperandShapeMismatch(
        operation: "color_jitter_forward",
        operand: "image",
        expected: "channel dim = 3 (RGB)",
        got: shape,
      ))
  }
}

fn apply_brightness(
  r: List(Float),
  g: List(Float),
  b: List(Float),
  strength: Float,
) -> #(List(Float), List(Float), List(Float)) {
  case strength <=. 0.0 {
    True -> #(r, g, b)
    False -> {
      let factor = uniform_in(1.0 -. strength, 1.0 +. strength)
      #(
        list.map(r, fn(v) { v *. factor }),
        list.map(g, fn(v) { v *. factor }),
        list.map(b, fn(v) { v *. factor }),
      )
    }
  }
}

fn apply_contrast(
  r: List(Float),
  g: List(Float),
  b: List(Float),
  strength: Float,
) -> #(List(Float), List(Float), List(Float)) {
  case strength <=. 0.0 {
    True -> #(r, g, b)
    False -> {
      let factor = uniform_in(1.0 -. strength, 1.0 +. strength)
      let n = int.to_float(list.length(r) * 3)
      let total =
        list.fold(r, 0.0, fn(acc, v) { acc +. v })
        +. list.fold(g, 0.0, fn(acc, v) { acc +. v })
        +. list.fold(b, 0.0, fn(acc, v) { acc +. v })
      let mean = case n >. 0.0 {
        True -> total /. n
        False -> 0.0
      }
      let shift = fn(v: Float) -> Float { { v -. mean } *. factor +. mean }
      #(list.map(r, shift), list.map(g, shift), list.map(b, shift))
    }
  }
}

fn apply_saturation(
  r: List(Float),
  g: List(Float),
  b: List(Float),
  strength: Float,
) -> #(List(Float), List(Float), List(Float)) {
  case strength <=. 0.0 {
    True -> #(r, g, b)
    False -> {
      let s = uniform_in(1.0 -. strength, 1.0 +. strength)
      // BT.601 luma: Y = 0.299 R + 0.587 G + 0.114 B
      let luma_for =
        list.map(list.zip(r, list.zip(g, b)), fn(triple) {
          let #(rv, #(gv, bv)) = triple
          0.299 *. rv +. 0.587 *. gv +. 0.114 *. bv
        })
      let lerp_one = fn(channel: List(Float)) -> List(Float) {
        list.map(list.zip(channel, luma_for), fn(pair) {
          let #(v, y) = pair
          y +. s *. { v -. y }
        })
      }
      #(lerp_one(r), lerp_one(g), lerp_one(b))
    }
  }
}

fn apply_hue(
  r: List(Float),
  g: List(Float),
  b: List(Float),
  strength: Float,
) -> #(List(Float), List(Float), List(Float)) {
  case strength <=. 0.0 {
    True -> #(r, g, b)
    False -> {
      let theta = uniform_in(0.0 -. strength, strength)
      // Rodrigues rotation around gray axis n = (1,1,1)/sqrt(3).
      // R = I cos θ + (1 - cos θ) n nᵀ + sin θ [n]_x
      //
      // With n = (1,1,1)/√3, the 3×3 matrix expands to (Wikipedia: axis–angle
      // representation):
      //   row_r = [cos θ + (1-cos θ)/3,      (1-cos θ)/3 - sin θ/√3, (1-cos θ)/3 + sin θ/√3]
      //   row_g = [(1-cos θ)/3 + sin θ/√3,   cos θ + (1-cos θ)/3,    (1-cos θ)/3 - sin θ/√3]
      //   row_b = [(1-cos θ)/3 - sin θ/√3,   (1-cos θ)/3 + sin θ/√3, cos θ + (1-cos θ)/3]
      let c = ffi.cos(theta)
      let s = ffi.sin(theta)
      let one_minus_c_third = { 1.0 -. c } /. 3.0
      let sqrt3 = ffi.sqrt(3.0)
      let s_over_sqrt3 = s /. sqrt3
      let diag = c +. one_minus_c_third
      let off1 = one_minus_c_third -. s_over_sqrt3
      let off2 = one_minus_c_third +. s_over_sqrt3

      // Pre-zip to walk all three channels in a single pass.
      let triples = list.zip(r, list.zip(g, b))
      let mixed =
        list.map(triples, fn(t) {
          let #(rv, #(gv, bv)) = t
          let rn = diag *. rv +. off1 *. gv +. off2 *. bv
          let gn = off2 *. rv +. diag *. gv +. off1 *. bv
          let bn = off1 *. rv +. off2 *. gv +. diag *. bv
          #(rn, gn, bn)
        })
      let r_out = list.map(mixed, fn(t) { t.0 })
      let g_out = list.map(mixed, fn(t) { t.1 })
      let b_out = list.map(mixed, fn(t) { t.2 })
      #(r_out, g_out, b_out)
    }
  }
}

// =============================================================================
// MIXUP
// =============================================================================

/// MixUp augmentation (Zhang et al., 2018, https://arxiv.org/abs/1710.09412).
///
/// Draws `λ ~ Beta(alpha, alpha)` and a random permutation `σ` of the batch,
/// then produces
/// ```
/// mixed_images[i] = λ · images[i] + (1 - λ) · images[σ(i)]
/// mixed_labels[i] = λ · one_hot(labels[i]) + (1 - λ) · one_hot(labels[σ(i)])
/// ```
/// Labels may be supplied either as class indices (`[B]`) or as a one-hot/soft
/// label matrix (`[B, num_classes]`). In both cases the output is a soft-label
/// matrix `[B, num_classes]`.
///
/// `alpha ≤ 0` is treated as a degenerate beta and short-circuits to
/// `λ = 1.0`, leaving the inputs unchanged.
pub fn mixup(
  images: Tensor,
  labels: Tensor,
  num_classes: Int,
  alpha: Float,
) -> Result(#(Tensor, Tensor), TensorError) {
  let image_shape = tensor.shape(images)
  use #(batch, _channels, _h, _w) <- result.try(parse_image_shape(
    "mixup",
    image_shape,
  ))
  use label_matrix <- result.try(normalize_labels(
    "mixup",
    labels,
    batch,
    num_classes,
  ))

  let lambda = sample_beta(alpha, alpha)
  let perm = random_permutation(batch)

  let image_data = tensor.to_list(images)
  let per_image_size = element_count(image_shape) / batch
  let mixed_image_data =
    mix_batched(image_data, perm, per_image_size, lambda)

  let mixed_label_data =
    mix_batched(label_matrix, perm, num_classes, lambda)

  Ok(#(
    Tensor(data: mixed_image_data, shape: image_shape),
    Tensor(data: mixed_label_data, shape: [batch, num_classes]),
  ))
}

// =============================================================================
// CUTMIX
// =============================================================================

/// CutMix augmentation (Yun et al., 2019, https://arxiv.org/abs/1905.04899).
///
/// Pastes a random axis-aligned rectangle from a partner image into each
/// sample. The rectangle width/height are scaled by `sqrt(1 - λ)` with
/// `λ ~ Beta(alpha, alpha)`, and `λ` is then **recomputed** from the actual
/// pasted area so the label mixing reflects the real overlap:
/// ```
/// cut_w   = round(W · sqrt(1 - λ))
/// cut_h   = round(H · sqrt(1 - λ))
/// rx, ry  ~ Uniform{0..W}, Uniform{0..H}
/// box     = [max(rx - cut_w/2, 0), min(rx + cut_w/2, W)] × ...
/// λ_real  = 1 - (box_w · box_h) / (W · H)
/// images[i, ..., box] = images[σ(i), ..., box]
/// labels[i]           = λ_real · one_hot(labels[i])
///                     + (1 - λ_real) · one_hot(labels[σ(i)])
/// ```
/// When the random rectangle has zero area, `λ_real = 1` and the input image
/// is returned unchanged.
pub fn cutmix(
  images: Tensor,
  labels: Tensor,
  num_classes: Int,
  alpha: Float,
) -> Result(#(Tensor, Tensor), TensorError) {
  let image_shape = tensor.shape(images)
  use #(batch, channels, height, width) <- result.try(parse_image_shape(
    "cutmix",
    image_shape,
  ))
  use label_matrix <- result.try(normalize_labels(
    "cutmix",
    labels,
    batch,
    num_classes,
  ))

  let lambda_initial = sample_beta(alpha, alpha)
  let cut_ratio = ffi.sqrt(float.max(0.0, 1.0 -. lambda_initial))
  let cut_w = float_to_round(int.to_float(width) *. cut_ratio)
  let cut_h = float_to_round(int.to_float(height) *. cut_ratio)
  let rx = case width <= 0 {
    True -> 0
    False -> int.random(width)
  }
  let ry = case height <= 0 {
    True -> 0
    False -> int.random(height)
  }
  let x1 = int_clamp(rx - cut_w / 2, 0, width)
  let x2 = int_clamp(rx + cut_w / 2, 0, width)
  let y1 = int_clamp(ry - cut_h / 2, 0, height)
  let y2 = int_clamp(ry + cut_h / 2, 0, height)
  let box_w = x2 - x1
  let box_h = y2 - y1
  let lambda = case width * height <= 0 {
    True -> 1.0
    False ->
      1.0 -. int.to_float(box_w * box_h) /. int.to_float(width * height)
  }

  let perm = random_permutation(batch)
  let image_data = tensor.to_list(images)
  let mixed_image_data = case box_w == 0 || box_h == 0 {
    True -> image_data
    False ->
      paste_box(image_data, perm, batch, channels, height, width, x1, x2, y1, y2)
  }

  let mixed_label_data = mix_batched(label_matrix, perm, num_classes, lambda)
  Ok(#(
    Tensor(data: mixed_image_data, shape: image_shape),
    Tensor(data: mixed_label_data, shape: [batch, num_classes]),
  ))
}

// =============================================================================
// HELPERS
// =============================================================================

fn parse_image_shape(
  op: String,
  shape: List(Int),
) -> Result(#(Int, Int, Int, Int), TensorError) {
  case shape {
    [b, c, h, w] -> Ok(#(b, c, h, w))
    [c, h, w] -> Ok(#(1, c, h, w))
    _ ->
      Error(OperandShapeMismatch(
        operation: op,
        operand: "image",
        expected: "[C, H, W] or [B, C, H, W]",
        got: shape,
      ))
  }
}

fn normalize_labels(
  op: String,
  labels: Tensor,
  batch: Int,
  num_classes: Int,
) -> Result(List(Float), TensorError) {
  case num_classes <= 0 {
    True ->
      Error(InvalidShape(
        op <> ": num_classes must be positive, got " <> int.to_string(num_classes),
      ))
    False -> {
      let label_shape = tensor.shape(labels)
      let label_data = tensor.to_list(labels)
      case label_shape {
        [b] if b == batch ->
          Ok(one_hot_from_indices(label_data, num_classes))
        [b, k] if b == batch && k == num_classes -> Ok(label_data)
        _ ->
          Error(OperandShapeMismatch(
            operation: op,
            operand: "labels",
            expected: "[B] indices or [B, num_classes] one-hot",
            got: label_shape,
          ))
      }
    }
  }
}

fn one_hot_from_indices(
  indices: List(Float),
  num_classes: Int,
) -> List(Float) {
  list.flat_map(indices, fn(idx_f) {
    let idx = float.round(idx_f)
    list.range(0, num_classes - 1)
    |> list.map(fn(k) {
      case k == idx {
        True -> 1.0
        False -> 0.0
      }
    })
  })
}

fn element_count(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(acc, dim) { acc * dim })
}

fn slice(data: List(Float), start: Int, length: Int) -> List(Float) {
  data |> list.drop(start) |> list.take(length)
}

fn mix_batched(
  data: List(Float),
  perm: List(Int),
  per_sample: Int,
  lambda: Float,
) -> List(Float) {
  list.index_map(perm, fn(p, i) {
    let a = slice(data, i * per_sample, per_sample)
    let b = slice(data, p * per_sample, per_sample)
    list.map(list.zip(a, b), fn(pair) {
      let #(va, vb) = pair
      lambda *. va +. { 1.0 -. lambda } *. vb
    })
  })
  |> list.flatten
}

fn paste_box(
  data: List(Float),
  perm: List(Int),
  batch: Int,
  channels: Int,
  height: Int,
  width: Int,
  x1: Int,
  x2: Int,
  y1: Int,
  y2: Int,
) -> List(Float) {
  let plane = height * width
  let stride = channels * plane
  list.range(0, batch - 1)
  |> list.flat_map(fn(i) {
    let p = case list.drop(perm, i) {
      [head, ..] -> head
      [] -> i
    }
    list.range(0, channels - 1)
    |> list.flat_map(fn(c) {
      let dst_base = i * stride + c * plane
      let src_base = p * stride + c * plane
      list.range(0, height - 1)
      |> list.flat_map(fn(y) {
        list.range(0, width - 1)
        |> list.map(fn(x) {
          let pick_src = y >= y1 && y < y2 && x >= x1 && x < x2
          let offset = y * width + x
          let base = case pick_src {
            True -> src_base
            False -> dst_base
          }
          case list.drop(data, base + offset) {
            [v, ..] -> v
            [] -> 0.0
          }
        })
      })
    })
  })
}

// --- Sampling utilities -----------------------------------------------------

fn uniform_in(lo: Float, hi: Float) -> Float {
  lo +. { hi -. lo } *. ffi.random_uniform()
}

fn random_permutation(n: Int) -> List(Int) {
  // Fisher–Yates shuffle on [0 .. n-1].
  let initial = list.range(0, n - 1)
  fisher_yates(initial, n - 1)
}

fn fisher_yates(xs: List(Int), i: Int) -> List(Int) {
  case i <= 0 {
    True -> xs
    False -> {
      let j = int.random(i + 1)
      let swapped = swap_at(xs, i, j)
      fisher_yates(swapped, i - 1)
    }
  }
}

fn swap_at(xs: List(Int), i: Int, j: Int) -> List(Int) {
  case i == j {
    True -> xs
    False -> {
      let xi = list_get(xs, i, 0)
      let xj = list_get(xs, j, 0)
      list.index_map(xs, fn(v, k) {
        case k == i, k == j {
          True, _ -> xj
          _, True -> xi
          _, _ -> v
        }
      })
    }
  }
}

fn list_get(xs: List(Int), i: Int, default: Int) -> Int {
  case list.drop(xs, i) {
    [v, ..] -> v
    [] -> default
  }
}

/// Sample `lambda ~ Beta(a, a)`. Uses the gamma-ratio identity
/// `Beta(a, b) = X / (X + Y)` with `X ~ Gamma(a, 1)`, `Y ~ Gamma(b, 1)`.
///
/// - `a ≤ 0` short-circuits to `1.0` so callers can disable mixing with
///   `alpha = 0`.
/// - For `a < 1` we use the boosting trick:
///   `Gamma(a) = Gamma(a + 1) · U^(1/a)`, with `U ∈ (0, 1)`.
/// - For `a ≥ 1` we use Marsaglia & Tsang's rejection method ("A simple
///   method for generating gamma variables", 2000), which is the same scheme
///   NumPy uses.
fn sample_beta(a: Float, b: Float) -> Float {
  case a <=. 0.0 || b <=. 0.0 {
    True -> 1.0
    False -> {
      let x = sample_gamma(a)
      let y = sample_gamma(b)
      let total = x +. y
      case total <=. 0.0 {
        True -> 0.5
        False -> x /. total
      }
    }
  }
}

fn sample_gamma(shape: Float) -> Float {
  case shape <. 1.0 {
    True -> {
      // Boosting: if X ~ Gamma(shape+1) and U ~ Uniform(0,1), then
      // X · U^(1/shape) ~ Gamma(shape).
      let boosted = marsaglia_tsang(shape +. 1.0)
      let u = float.max(ffi.random_uniform(), 1.0e-12)
      boosted *. ffi.pow(u, 1.0 /. shape)
    }
    False -> marsaglia_tsang(shape)
  }
}

fn marsaglia_tsang(shape: Float) -> Float {
  let d = shape -. 1.0 /. 3.0
  let c = 1.0 /. ffi.sqrt(9.0 *. d)
  marsaglia_loop(d, c, 0)
}

fn marsaglia_loop(d: Float, c: Float, attempts: Int) -> Float {
  case attempts >= 100 {
    True -> d
    // Hard guard: should never trigger statistically.
    False -> {
      let x = standard_normal()
      let v_base = 1.0 +. c *. x
      case v_base <=. 0.0 {
        True -> marsaglia_loop(d, c, attempts + 1)
        False -> {
          let v = v_base *. v_base *. v_base
          let u = float.max(ffi.random_uniform(), 1.0e-12)
          let lhs = u
          let rhs =
            ffi.exp(0.5 *. x *. x +. d -. d *. v +. d *. ffi.log(v))
          case lhs <. rhs {
            True -> d *. v
            False -> marsaglia_loop(d, c, attempts + 1)
          }
        }
      }
    }
  }
}

fn standard_normal() -> Float {
  // Box–Muller: z = sqrt(-2 ln u1) · cos(2π u2).
  let u1 = float.max(ffi.random_uniform(), 1.0e-12)
  let u2 = ffi.random_uniform()
  ffi.sqrt(-2.0 *. ffi.log(u1)) *. ffi.cos(2.0 *. ffi.pi *. u2)
}

fn int_clamp(value: Int, lo: Int, hi: Int) -> Int {
  case value < lo {
    True -> lo
    False ->
      case value > hi {
        True -> hi
        False -> value
      }
  }
}

fn float_to_round(value: Float) -> Int {
  float.round(value)
}
