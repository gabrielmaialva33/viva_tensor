//// Image preprocessing transforms operating on CHW (`[C, H, W]`) tensors,
//// or batched NCHW (`[B, C, H, W]`).
////
//// This layout matches PyTorch / Torchvision: channels first, single image
//// is 3-D, a batch is 4-D. Pixel intensities are expected to live in
//// `[0.0, 1.0]` (use `to_tensor` / `to_byte_image` to round-trip with
//// byte buffers in HWC order — the PIL convention on disk).
////
//// All transforms are pure functions that return a fresh `Tensor`. Random
//// transforms use `int.random` from the BEAM PRNG, which is not seedable —
//// outputs are non-deterministic across calls. They are documented per
//// function.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape, RankMismatch}
import viva_tensor/tensor.{type Tensor}

// --- Types ------------------------------------------------------------------

/// Resampling mode for `resize`.
///
/// - `Nearest`: nearest-neighbour. Cheap, blocky, exact pixel values.
/// - `Bilinear`: linear interpolation along both spatial axes using
///   `align_corners=False` (centers mapped as `src = (out + 0.5) * (in/out) - 0.5`,
///   clamped to `[0, in - 1]`).
pub type ResizeMode {
  Nearest
  Bilinear
}

// --- Internal helpers -------------------------------------------------------

/// Resolve `[C, H, W]` from a 3-D shape, or `[B, C, H, W]` from a 4-D shape.
/// Returns `#(batch, c, h, w)` where `batch = 1` for the 3-D case.
fn chw_or_bchw(
  op: String,
  shp: List(Int),
) -> Result(#(Int, Int, Int, Int, Bool), TensorError) {
  case shp {
    [c, h, w] -> Ok(#(1, c, h, w, False))
    [b, c, h, w] -> Ok(#(b, c, h, w, True))
    _ -> Error(RankMismatch(op, 3, shp))
  }
}

fn make_shape(c: Int, h: Int, w: Int, batched: Bool, batch: Int) -> List(Int) {
  case batched {
    True -> [batch, c, h, w]
    False -> [c, h, w]
  }
}

fn min_float(a: Float, b: Float) -> Float {
  case a <. b {
    True -> a
    False -> b
  }
}

fn max_float(a: Float, b: Float) -> Float {
  case a >. b {
    True -> a
    False -> b
  }
}

fn clamp_f(x: Float, lo: Float, hi: Float) -> Float {
  max_float(lo, min_float(hi, x))
}

fn clamp_i(x: Int, lo: Int, hi: Int) -> Int {
  case x < lo {
    True -> lo
    False ->
      case x > hi {
        True -> hi
        False -> x
      }
  }
}

fn int_to_float(i: Int) -> Float {
  int.to_float(i)
}

fn list_get(xs: List(Float), idx: Int) -> Float {
  case xs {
    [] -> 0.0
    [head, ..rest] ->
      case idx {
        0 -> head
        _ -> list_get(rest, idx - 1)
      }
  }
}

// --- Sample/gather over flat CHW data ---------------------------------------

/// Fetch a pixel from the flat CHW buffer for image `b` (always 0 for 3-D),
/// channel `c`, row `h`, col `w`. Clamps `h`/`w` to the image bounds.
fn sample_clamped(
  data: List(Float),
  b: Int,
  c: Int,
  h: Int,
  w: Int,
  channels: Int,
  height: Int,
  width: Int,
) -> Float {
  let h_c = clamp_i(h, 0, height - 1)
  let w_c = clamp_i(w, 0, width - 1)
  let idx =
    b * channels * height * width + c * height * width + h_c * width + w_c
  list_get(data, idx)
}

// --- 1. resize --------------------------------------------------------------

/// Resize an image to `[..., C, new_h, new_w]` using `mode`.
///
/// - `image` must be `[C, H, W]` (3-D) or `[B, C, H, W]` (4-D).
/// - Output rank matches input rank; only the spatial axes change.
/// - Errors: `RankMismatch("resize", 3, shape)` if the rank is not 3 or 4.
///   `InvalidShape` if `new_h`/`new_w` is non-positive.
pub fn resize(
  image: Tensor,
  new_h: Int,
  new_w: Int,
  mode: ResizeMode,
) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "resize",
    tensor.shape(image),
  ))
  case new_h <= 0 || new_w <= 0 {
    True ->
      Error(InvalidShape(
        "resize: new_h and new_w must be positive, got "
        <> int.to_string(new_h)
        <> "x"
        <> int.to_string(new_w),
      ))
    False -> {
      let data = tensor.to_list(image)
      let out_data =
        list.range(0, batch - 1)
        |> list.flat_map(fn(b) {
          list.range(0, c - 1)
          |> list.flat_map(fn(ch) {
            list.range(0, new_h - 1)
            |> list.flat_map(fn(oh) {
              list.range(0, new_w - 1)
              |> list.map(fn(ow) {
                case mode {
                  Nearest ->
                    sample_nearest(data, b, ch, oh, ow, c, h, w, new_h, new_w)
                  Bilinear ->
                    sample_bilinear(data, b, ch, oh, ow, c, h, w, new_h, new_w)
                }
              })
            })
          })
        })
      tensor.reshape(
        tensor.from_list(out_data),
        make_shape(c, new_h, new_w, batched, batch),
      )
    }
  }
}

fn sample_nearest(
  data: List(Float),
  b: Int,
  ch: Int,
  oh: Int,
  ow: Int,
  channels: Int,
  in_h: Int,
  in_w: Int,
  out_h: Int,
  out_w: Int,
) -> Float {
  // `align_corners=False`: src = (out + 0.5) * in/out - 0.5
  let src_h =
    { int_to_float(oh) +. 0.5 }
    *. int_to_float(in_h)
    /. int_to_float(out_h)
    -. 0.5
  let src_w =
    { int_to_float(ow) +. 0.5 }
    *. int_to_float(in_w)
    /. int_to_float(out_w)
    -. 0.5
  let ih = float.round(src_h)
  let iw = float.round(src_w)
  sample_clamped(data, b, ch, ih, iw, channels, in_h, in_w)
}

fn sample_bilinear(
  data: List(Float),
  b: Int,
  ch: Int,
  oh: Int,
  ow: Int,
  channels: Int,
  in_h: Int,
  in_w: Int,
  out_h: Int,
  out_w: Int,
) -> Float {
  let src_h =
    { int_to_float(oh) +. 0.5 }
    *. int_to_float(in_h)
    /. int_to_float(out_h)
    -. 0.5
  let src_w =
    { int_to_float(ow) +. 0.5 }
    *. int_to_float(in_w)
    /. int_to_float(out_w)
    -. 0.5
  let h_clamped = clamp_f(src_h, 0.0, int_to_float(in_h - 1))
  let w_clamped = clamp_f(src_w, 0.0, int_to_float(in_w - 1))
  let h0 = float.truncate(h_clamped)
  let w0 = float.truncate(w_clamped)
  let h1 = clamp_i(h0 + 1, 0, in_h - 1)
  let w1 = clamp_i(w0 + 1, 0, in_w - 1)
  let dh = h_clamped -. int_to_float(h0)
  let dw = w_clamped -. int_to_float(w0)
  let v00 = sample_clamped(data, b, ch, h0, w0, channels, in_h, in_w)
  let v01 = sample_clamped(data, b, ch, h0, w1, channels, in_h, in_w)
  let v10 = sample_clamped(data, b, ch, h1, w0, channels, in_h, in_w)
  let v11 = sample_clamped(data, b, ch, h1, w1, channels, in_h, in_w)
  let top = v00 *. { 1.0 -. dw } +. v01 *. dw
  let bot = v10 *. { 1.0 -. dw } +. v11 *. dw
  top *. { 1.0 -. dh } +. bot *. dh
}

// --- 2. center_crop ---------------------------------------------------------

/// Crop the centre `target_h x target_w` region.
///
/// Input/output rank: `[C, H, W]` or `[B, C, H, W]`.
/// Errors:
/// - `RankMismatch("center_crop", 3, shape)` if rank is not 3 or 4.
/// - `InvalidShape("center_crop: target size exceeds image")` if the crop
///   doesn't fit.
pub fn center_crop(
  image: Tensor,
  target_h: Int,
  target_w: Int,
) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "center_crop",
    tensor.shape(image),
  ))
  case target_h > h || target_w > w || target_h <= 0 || target_w <= 0 {
    True -> Error(InvalidShape("center_crop: target size exceeds image"))
    False -> {
      let top = { h - target_h } / 2
      let left = { w - target_w } / 2
      crop_region(image, batch, c, h, w, top, left, target_h, target_w, batched)
    }
  }
}

// --- 3. random_crop ---------------------------------------------------------

/// Crop a `target_h x target_w` window at a random top-left corner.
///
/// Non-deterministic: uses `int.random` (BEAM PRNG, not seedable).
/// Errors mirror `center_crop`.
pub fn random_crop(
  image: Tensor,
  target_h: Int,
  target_w: Int,
) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "random_crop",
    tensor.shape(image),
  ))
  case target_h > h || target_w > w || target_h <= 0 || target_w <= 0 {
    True -> Error(InvalidShape("random_crop: target size exceeds image"))
    False -> {
      let max_top = h - target_h
      let max_left = w - target_w
      let top = case max_top {
        0 -> 0
        n -> int.random(n + 1)
      }
      let left = case max_left {
        0 -> 0
        n -> int.random(n + 1)
      }
      crop_region(image, batch, c, h, w, top, left, target_h, target_w, batched)
    }
  }
}

fn crop_region(
  image: Tensor,
  batch: Int,
  c: Int,
  h: Int,
  w: Int,
  top: Int,
  left: Int,
  target_h: Int,
  target_w: Int,
  batched: Bool,
) -> Result(Tensor, TensorError) {
  let data = tensor.to_list(image)
  let out =
    list.range(0, batch - 1)
    |> list.flat_map(fn(b) {
      list.range(0, c - 1)
      |> list.flat_map(fn(ch) {
        list.range(0, target_h - 1)
        |> list.flat_map(fn(oh) {
          list.range(0, target_w - 1)
          |> list.map(fn(ow) {
            sample_clamped(data, b, ch, top + oh, left + ow, c, h, w)
          })
        })
      })
    })
  tensor.reshape(
    tensor.from_list(out),
    make_shape(c, target_h, target_w, batched, batch),
  )
}

// --- 4. horizontal_flip -----------------------------------------------------

/// Mirror the image along the width axis. Deterministic.
///
/// Input/output rank: `[C, H, W]` or `[B, C, H, W]`.
pub fn horizontal_flip(image: Tensor) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "horizontal_flip",
    tensor.shape(image),
  ))
  let data = tensor.to_list(image)
  let out =
    list.range(0, batch - 1)
    |> list.flat_map(fn(b) {
      list.range(0, c - 1)
      |> list.flat_map(fn(ch) {
        list.range(0, h - 1)
        |> list.flat_map(fn(row) {
          list.range(0, w - 1)
          |> list.map(fn(col) {
            sample_clamped(data, b, ch, row, w - 1 - col, c, h, w)
          })
        })
      })
    })
  tensor.reshape(tensor.from_list(out), make_shape(c, h, w, batched, batch))
}

// --- 5. vertical_flip -------------------------------------------------------

/// Mirror the image along the height axis. Deterministic.
///
/// Input/output rank: `[C, H, W]` or `[B, C, H, W]`.
pub fn vertical_flip(image: Tensor) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "vertical_flip",
    tensor.shape(image),
  ))
  let data = tensor.to_list(image)
  let out =
    list.range(0, batch - 1)
    |> list.flat_map(fn(b) {
      list.range(0, c - 1)
      |> list.flat_map(fn(ch) {
        list.range(0, h - 1)
        |> list.flat_map(fn(row) {
          list.range(0, w - 1)
          |> list.map(fn(col) {
            sample_clamped(data, b, ch, h - 1 - row, col, c, h, w)
          })
        })
      })
    })
  tensor.reshape(tensor.from_list(out), make_shape(c, h, w, batched, batch))
}

// --- 6. random_horizontal_flip ----------------------------------------------

/// Flip horizontally with probability `p`. `p` is clamped to `[0.0, 1.0]`.
///
/// Non-deterministic: uses `int.random` (BEAM PRNG, not seedable).
pub fn random_horizontal_flip(
  image: Tensor,
  p: Float,
) -> Result(Tensor, TensorError) {
  let p_clamped = clamp_f(p, 0.0, 1.0)
  // 1_000_000 buckets is plenty for typical p (3-6 decimal digits).
  let resolution = 1_000_000
  let threshold = float.round(p_clamped *. int_to_float(resolution))
  case int.random(resolution) < threshold {
    True -> horizontal_flip(image)
    False -> {
      // Identity copy keeps the contract "always returns a fresh tensor".
      use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
        "random_horizontal_flip",
        tensor.shape(image),
      ))
      tensor.reshape(
        tensor.from_list(tensor.to_list(image)),
        make_shape(c, h, w, batched, batch),
      )
    }
  }
}

// --- 7. normalize -----------------------------------------------------------

/// Per-channel `(x - mean[c]) / std[c]` normalization.
///
/// - `mean` and `std` lengths must equal the channel count (`C`).
/// - Input rank: `[C, H, W]` or `[B, C, H, W]`.
/// - Errors: `RankMismatch` for bad rank, `InvalidShape` when length doesn't
///   match `C` or any `std` entry is zero.
pub fn normalize(
  image: Tensor,
  mean: List(Float),
  std: List(Float),
) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "normalize",
    tensor.shape(image),
  ))
  case list.length(mean) == c && list.length(std) == c {
    False ->
      Error(InvalidShape(
        "normalize: mean/std length must equal channels ("
        <> int.to_string(c)
        <> ")",
      ))
    True ->
      case list.any(std, fn(s) { s == 0.0 }) {
        True -> Error(InvalidShape("normalize: std entries must be non-zero"))
        False -> {
          let data = tensor.to_list(image)
          let out =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, c - 1)
              |> list.flat_map(fn(ch) {
                let m = list_get(mean, ch)
                let s = list_get(std, ch)
                list.range(0, h - 1)
                |> list.flat_map(fn(row) {
                  list.range(0, w - 1)
                  |> list.map(fn(col) {
                    let v = sample_clamped(data, b, ch, row, col, c, h, w)
                    { v -. m } /. s
                  })
                })
              })
            })
          tensor.reshape(
            tensor.from_list(out),
            make_shape(c, h, w, batched, batch),
          )
        }
      }
  }
}

// --- 8. to_grayscale --------------------------------------------------------

/// Convert a 3-channel image to grayscale via ITU-R 601 luma weights
/// (`0.299*R + 0.587*G + 0.114*B`).
///
/// `num_output_channels` must be `1` (single-channel output) or `3` (luma
/// broadcast across 3 channels).
///
/// Input must have 3 channels; rank `[3, H, W]` or `[B, 3, H, W]`.
/// Errors:
/// - `RankMismatch` for bad rank.
/// - `InvalidShape` if channels != 3 or `num_output_channels` is neither 1
///   nor 3.
pub fn to_grayscale(
  image: Tensor,
  num_output_channels: Int,
) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "to_grayscale",
    tensor.shape(image),
  ))
  case c == 3 {
    False ->
      Error(InvalidShape(
        "to_grayscale: input must have 3 channels, got " <> int.to_string(c),
      ))
    True ->
      case num_output_channels == 1 || num_output_channels == 3 {
        False ->
          Error(InvalidShape(
            "to_grayscale: num_output_channels must be 1 or 3, got "
            <> int.to_string(num_output_channels),
          ))
        True -> {
          let data = tensor.to_list(image)
          // Pre-compute luma per spatial position per image.
          let luma_data =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, h - 1)
              |> list.flat_map(fn(row) {
                list.range(0, w - 1)
                |> list.map(fn(col) {
                  let r = sample_clamped(data, b, 0, row, col, 3, h, w)
                  let g = sample_clamped(data, b, 1, row, col, 3, h, w)
                  let bl = sample_clamped(data, b, 2, row, col, 3, h, w)
                  0.299 *. r +. 0.587 *. g +. 0.114 *. bl
                })
              })
            })
          case num_output_channels {
            1 ->
              tensor.reshape(
                tensor.from_list(luma_data),
                make_shape(1, h, w, batched, batch),
              )
            _ -> {
              // Broadcast luma to 3 channels — for each image, repeat the
              // luma plane 3 times before the next image's data.
              let plane_size = h * w
              let out =
                list.range(0, batch - 1)
                |> list.flat_map(fn(b) {
                  let plane = slice_list(luma_data, b * plane_size, plane_size)
                  list.flatten([plane, plane, plane])
                })
              tensor.reshape(
                tensor.from_list(out),
                make_shape(3, h, w, batched, batch),
              )
            }
          }
        }
      }
  }
}

fn slice_list(xs: List(Float), offset: Int, length: Int) -> List(Float) {
  xs
  |> list.drop(offset)
  |> list.take(length)
}

// --- 9. adjust_brightness ---------------------------------------------------

/// Multiply every pixel by `factor`, then clamp to `[0.0, 1.0]`.
///
/// Input rank: `[C, H, W]` or `[B, C, H, W]`.
pub fn adjust_brightness(
  image: Tensor,
  factor: Float,
) -> Result(Tensor, TensorError) {
  use #(_batch, _c, _h, _w, _batched) <- result.try(chw_or_bchw(
    "adjust_brightness",
    tensor.shape(image),
  ))
  let data = tensor.to_list(image)
  let out = list.map(data, fn(v) { clamp_f(v *. factor, 0.0, 1.0) })
  tensor.reshape(tensor.from_list(out), tensor.shape(image))
}

// --- 10. adjust_contrast ----------------------------------------------------

/// Linearly interpolate each pixel toward its channel mean:
/// `out = mean + factor * (x - mean)`, then clamp to `[0.0, 1.0]`.
///
/// Per-channel mean is computed over `H * W` (and per batch element when
/// the input is 4-D).
pub fn adjust_contrast(
  image: Tensor,
  factor: Float,
) -> Result(Tensor, TensorError) {
  use #(batch, c, h, w, batched) <- result.try(chw_or_bchw(
    "adjust_contrast",
    tensor.shape(image),
  ))
  let data = tensor.to_list(image)
  let n = h * w
  let n_f = int_to_float(n)
  let out =
    list.range(0, batch - 1)
    |> list.flat_map(fn(b) {
      list.range(0, c - 1)
      |> list.flat_map(fn(ch) {
        // Per-channel, per-image mean.
        let plane =
          list.range(0, h - 1)
          |> list.flat_map(fn(row) {
            list.range(0, w - 1)
            |> list.map(fn(col) {
              sample_clamped(data, b, ch, row, col, c, h, w)
            })
          })
        let sum = list.fold(plane, 0.0, fn(acc, v) { acc +. v })
        let mean = sum /. n_f
        list.map(plane, fn(v) {
          clamp_f(mean +. factor *. { v -. mean }, 0.0, 1.0)
        })
      })
    })
  tensor.reshape(tensor.from_list(out), make_shape(c, h, w, batched, batch))
}

// --- 11. to_tensor ----------------------------------------------------------

/// Convert a flat list of bytes (PIL-style HWC, values 0..255) into a CHW
/// tensor scaled to `[0.0, 1.0]`.
///
/// Input length must equal `height * width * channels`.
/// Output shape: `[channels, height, width]`.
///
/// Errors: `InvalidShape` when dimensions are non-positive or the input
/// length doesn't match.
pub fn to_tensor(
  byte_image: List(Int),
  height: Int,
  width: Int,
  channels: Int,
) -> Result(Tensor, TensorError) {
  case height <= 0 || width <= 0 || channels <= 0 {
    True ->
      Error(InvalidShape(
        "to_tensor: dimensions must be positive (got h="
        <> int.to_string(height)
        <> ", w="
        <> int.to_string(width)
        <> ", c="
        <> int.to_string(channels)
        <> ")",
      ))
    False -> {
      let expected = height * width * channels
      case list.length(byte_image) == expected {
        False ->
          Error(InvalidShape(
            "to_tensor: byte_image length "
            <> int.to_string(list.length(byte_image))
            <> " != h*w*c = "
            <> int.to_string(expected),
          ))
        True -> {
          // HWC -> CHW transpose, then scale by 1/255.
          // HWC index: (row * width + col) * channels + ch
          // CHW index: ch * height * width + row * width + col
          let hwc = list.map(byte_image, int.to_float)
          let chw =
            list.range(0, channels - 1)
            |> list.flat_map(fn(ch) {
              list.range(0, height - 1)
              |> list.flat_map(fn(row) {
                list.range(0, width - 1)
                |> list.map(fn(col) {
                  let idx = { row * width + col } * channels + ch
                  list_get(hwc, idx) /. 255.0
                })
              })
            })
          tensor.reshape(tensor.from_list(chw), [channels, height, width])
        }
      }
    }
  }
}

// --- 12. to_byte_image ------------------------------------------------------

/// Convert a CHW tensor in `[0.0, 1.0]` to a flat byte list in HWC order
/// (PIL convention). Each value is clamped to `[0, 255]` after scaling by
/// 255.
///
/// Input shape: `[C, H, W]` (3-D only — batched is not supported).
/// Errors: `RankMismatch("to_byte_image", 3, shape)` for non-3-D input.
pub fn to_byte_image(image: Tensor) -> Result(List(Int), TensorError) {
  let shp = tensor.shape(image)
  case shp {
    [c, h, w] -> {
      let data = tensor.to_list(image)
      let bytes =
        list.range(0, h - 1)
        |> list.flat_map(fn(row) {
          list.range(0, w - 1)
          |> list.flat_map(fn(col) {
            list.range(0, c - 1)
            |> list.map(fn(ch) {
              let idx = ch * h * w + row * w + col
              let scaled = list_get(data, idx) *. 255.0
              let rounded = float.round(scaled)
              clamp_i(rounded, 0, 255)
            })
          })
        })
      Ok(bytes)
    }
    _ -> Error(RankMismatch("to_byte_image", 3, shp))
  }
}

// --- 13. compose ------------------------------------------------------------

/// Apply a list of transforms in order, threading the result through each
/// step. Bails on the first `Error`.
///
/// ```gleam
/// compose(
///   [horizontal_flip, fn(t) { normalize(t, mean, std) }],
///   img,
/// )
/// ```
pub fn compose(
  transforms: List(fn(Tensor) -> Result(Tensor, TensorError)),
  image: Tensor,
) -> Result(Tensor, TensorError) {
  list.try_fold(transforms, image, fn(acc, f) { f(acc) })
}
