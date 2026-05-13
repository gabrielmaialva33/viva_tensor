//// Pooling and regularization layers for VIVA tensor.
////
//// This module ships the pieces that the existing 2D max/avg pool helpers
//// didn't cover: 1D pooling, adaptive average pooling, nearest/bilinear
//// upsampling and inverted dropout. All implementations are pure Gleam with
//// no NIF, no autograd integration — they're meant to round out the model
//// kit so ResNet-style heads and U-Net-style decoders can be wired up.
////
//// Output shape formulas (PyTorch convention):
////   MaxPool1d / AvgPool1d:   L_out = floor((L_in + 2*P - K) / S) + 1
////   AdaptiveAvgPool1d:       L_out = config.output_size
////   AdaptiveAvgPool2d:       [H_out, W_out] = [config.output_h, config.output_w]
////   Upsample (nearest):      [H * scale, W * scale]
////   Upsample (bilinear):     [H * scale, W * scale]
////
//// Dropout uses `int.random` from gleam_stdlib, which is the BEAM's
//// non-deterministic PRNG — no seeding API is exposed. Two calls in the same
//// process produce different masks; do not rely on reproducibility.

import gleam/float
import gleam/int
import gleam/list
import viva_tensor/core/error.{type TensorError, InvalidShape}
import viva_tensor/core/ffi.{type ErlangArray}
import viva_tensor/tensor.{type Tensor}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Inverted-dropout regularization.
///
/// `p` is the probability of dropping a single element. `p = 0.0` is a no-op
/// (passthrough). `p = 1.0` zeroes every element — see `dropout_forward` for
/// the scaling convention at that edge.
pub type Dropout {
  Dropout(p: Float)
}

/// 1D max-pool configuration.
///
/// Output length: `L_out = (L_in + 2*padding - kernel_size) / stride + 1`.
pub type MaxPool1dConfig {
  MaxPool1dConfig(kernel_size: Int, stride: Int, padding: Int)
}

/// 1D average-pool configuration.
///
/// Output length: `L_out = (L_in + 2*padding - kernel_size) / stride + 1`.
pub type AvgPool1dConfig {
  AvgPool1dConfig(kernel_size: Int, stride: Int, padding: Int)
}

/// 2D adaptive average-pool configuration.
///
/// Output shape: `[batch, channels, output_h, output_w]`. Kernel/stride are
/// derived per output cell so the input is split into roughly even windows
/// (matches PyTorch's `nn.AdaptiveAvgPool2d`).
pub type AdaptiveAvgPool2dConfig {
  AdaptiveAvgPool2dConfig(output_h: Int, output_w: Int)
}

/// 1D adaptive average-pool configuration.
///
/// Output shape: `[batch, channels, output_size]`.
pub type AdaptiveAvgPool1dConfig {
  AdaptiveAvgPool1dConfig(output_size: Int)
}

/// Upsample mode — `Nearest` repeats the source pixel, `Bilinear` does linear
/// interpolation along both spatial axes.
pub type UpsampleMode {
  Nearest
  Bilinear
}

/// 2D upsample configuration.
///
/// Output shape: `[batch, channels, H * scale_factor, W * scale_factor]`.
pub type UpsampleConfig {
  UpsampleConfig(scale_factor: Int, mode: UpsampleMode)
}

// ---------------------------------------------------------------------------
// Dropout
// ---------------------------------------------------------------------------

/// Initialize a `Dropout` layer with drop probability `p`.
///
/// `p` is the probability of dropping each element; `p = 0.0` keeps
/// everything, `p = 1.0` drops everything. `p` is not validated — out-of-range
/// values fall through to `dropout_forward` and produce the natural
/// passthrough or zero output documented there.
pub fn dropout_init(p: Float) -> Dropout {
  Dropout(p: p)
}

/// Forward pass for inverted dropout.
///
/// Output shape: same as input.
///
/// - `training = False`: pure passthrough.
/// - `training = True`, `p = 0.0`: passthrough.
/// - `training = True`, `p = 1.0`: every element is zeroed. The inverted-
///   dropout scale factor `1 / (1 - p)` is undefined at `p = 1.0`, so we
///   short-circuit to zero (matches PyTorch behavior).
/// - `training = True`, `0.0 < p < 1.0`: each element is independently kept
///   with probability `1 - p` and scaled by `1 / (1 - p)` so the expected
///   value is preserved.
///
/// Randomness comes from `int.random` (BEAM PRNG). It is not seedable, so
/// repeated calls in the same process produce different masks. Tests that
/// need determinism should use `training = False` or `p = 0.0`.
pub fn dropout_forward(
  layer: Dropout,
  input: Tensor,
  training: Bool,
) -> Tensor {
  case training {
    False -> input
    True ->
      case layer.p {
        p if p <=. 0.0 -> input
        p if p >=. 1.0 -> {
          // p == 1.0 (or beyond): zero everything.
          let shp = tensor.shape(input)
          let n = list.length(tensor.to_list(input))
          let zeros = list.repeat(0.0, n)
          case tensor.reshape(tensor.from_list(zeros), shp) {
            Ok(t) -> t
            Error(_) -> input
          }
        }
        p -> {
          let keep_prob = 1.0 -. p
          let scale = 1.0 /. keep_prob
          // Use a wide-range integer roll and compare to a threshold scaled
          // by the same range. 1_000_000 gives ~6 decimal digits of resolution
          // for the keep mask, which is plenty for tests + inference.
          let resolution = 1_000_000
          let threshold = float.round(keep_prob *. int.to_float(resolution))
          let shp = tensor.shape(input)
          let masked =
            list.map(tensor.to_list(input), fn(v) {
              case int.random(resolution) < threshold {
                True -> v *. scale
                False -> 0.0
              }
            })
          case tensor.reshape(tensor.from_list(masked), shp) {
            Ok(t) -> t
            Error(_) -> input
          }
        }
      }
  }
}

// ---------------------------------------------------------------------------
// MaxPool1d
// ---------------------------------------------------------------------------

/// Forward pass for 1D max pooling.
///
/// `input` shape `[batch, channels, length]`. Output shape
/// `[batch, channels, L_out]` where
/// `L_out = (length + 2*padding - kernel_size) / stride + 1`.
pub fn max_pool_1d_forward(
  config: MaxPool1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  pool_1d_forward(input, config.kernel_size, config.stride, config.padding, True)
}

// ---------------------------------------------------------------------------
// AvgPool1d
// ---------------------------------------------------------------------------

/// Forward pass for 1D average pooling.
///
/// `input` shape `[batch, channels, length]`. Output shape
/// `[batch, channels, L_out]` where
/// `L_out = (length + 2*padding - kernel_size) / stride + 1`.
///
/// Padding is included in the averaging window count, matching PyTorch's
/// default `count_include_pad=True` for `nn.AvgPool1d`.
pub fn avg_pool_1d_forward(
  config: AvgPool1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  pool_1d_forward(
    input,
    config.kernel_size,
    config.stride,
    config.padding,
    False,
  )
}

fn pool_1d_forward(
  input: Tensor,
  kernel_size: Int,
  stride: Int,
  padding: Int,
  is_max: Bool,
) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(input)
  case shp {
    [batch, channels, length] -> {
      case stride <= 0 || kernel_size <= 0 {
        True ->
          Error(InvalidShape(
            "pool_1d: kernel_size and stride must be positive, got kernel="
            <> int.to_string(kernel_size)
            <> " stride="
            <> int.to_string(stride),
          ))
        False -> {
          let padded_length = length + 2 * padding
          let out_length = { padded_length - kernel_size } / stride + 1
          case out_length <= 0 {
            True ->
              Error(InvalidShape(
                "pool_1d: invalid output length "
                <> int.to_string(out_length)
                <> " for input length "
                <> int.to_string(length)
                <> ", kernel "
                <> int.to_string(kernel_size)
                <> ", stride "
                <> int.to_string(stride)
                <> ", padding "
                <> int.to_string(padding),
              ))
            False -> {
              let arr = ffi.list_to_array(tensor.to_list(input))
              let kf = int.to_float(kernel_size)
              let out =
                list.range(0, batch - 1)
                |> list.flat_map(fn(b) {
                  list.range(0, channels - 1)
                  |> list.flat_map(fn(c) {
                    let base = b * channels * length + c * length
                    list.range(0, out_length - 1)
                    |> list.map(fn(o) {
                      let start = o * stride - padding
                      pool_1d_window(
                        arr,
                        base,
                        start,
                        kernel_size,
                        length,
                        is_max,
                        kf,
                      )
                    })
                  })
                })
              tensor.reshape(tensor.from_list(out), [batch, channels, out_length])
            }
          }
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "pool_1d: input must have shape [batch, channels, length], got "
        <> shape_to_string(shp),
      ))
  }
}

fn pool_1d_window(
  arr: ErlangArray,
  base: Int,
  start: Int,
  kernel_size: Int,
  length: Int,
  is_max: Bool,
  kf: Float,
) -> Float {
  case is_max {
    True -> max_window(arr, base, start, 0, kernel_size, length, AccNone)
    False -> {
      let sum = sum_window(arr, base, start, 0, kernel_size, length, 0.0)
      sum /. kf
    }
  }
}

type Acc {
  AccNone
  AccSome(Float)
}

fn max_window(
  arr: ErlangArray,
  base: Int,
  start: Int,
  k: Int,
  kernel_size: Int,
  length: Int,
  acc: Acc,
) -> Float {
  case k >= kernel_size {
    True ->
      case acc {
        AccNone -> 0.0
        AccSome(v) -> v
      }
    False -> {
      let idx = start + k
      let v = case idx >= 0 && idx < length {
        True -> ffi.array_get(arr, base + idx)
        False -> 0.0
      }
      let next = case acc {
        AccNone -> AccSome(v)
        AccSome(m) ->
          case v >. m {
            True -> AccSome(v)
            False -> AccSome(m)
          }
      }
      max_window(arr, base, start, k + 1, kernel_size, length, next)
    }
  }
}

fn sum_window(
  arr: ErlangArray,
  base: Int,
  start: Int,
  k: Int,
  kernel_size: Int,
  length: Int,
  acc: Float,
) -> Float {
  case k >= kernel_size {
    True -> acc
    False -> {
      let idx = start + k
      let v = case idx >= 0 && idx < length {
        True -> ffi.array_get(arr, base + idx)
        False -> 0.0
      }
      sum_window(arr, base, start, k + 1, kernel_size, length, acc +. v)
    }
  }
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool2d
// ---------------------------------------------------------------------------

/// Forward pass for 2D adaptive average pooling.
///
/// `input` shape `[batch, channels, H, W]`. Output shape
/// `[batch, channels, output_h, output_w]`. Per output cell `(i, j)` the
/// source window is `[h_start, h_end) x [w_start, w_end)` where
/// `h_start = floor(i * H / output_h)`, `h_end = ceil((i+1) * H / output_h)`
/// (and analogously for `w`). The cell value is the unweighted mean of the
/// window. Matches PyTorch's `nn.AdaptiveAvgPool2d` for the common
/// `H % output_h == 0` and ResNet identity (`output == input`) cases.
pub fn adaptive_avg_pool_2d_forward(
  config: AdaptiveAvgPool2dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(input)
  case shp {
    [batch, channels, h_in, w_in] -> {
      case config.output_h <= 0 || config.output_w <= 0 {
        True ->
          Error(InvalidShape(
            "adaptive_avg_pool_2d: output dims must be positive, got "
            <> int.to_string(config.output_h)
            <> "x"
            <> int.to_string(config.output_w),
          ))
        False -> {
          let arr = ffi.list_to_array(tensor.to_list(input))
          let out_h = config.output_h
          let out_w = config.output_w
          let out =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, channels - 1)
              |> list.flat_map(fn(c) {
                let base = b * channels * h_in * w_in + c * h_in * w_in
                list.range(0, out_h - 1)
                |> list.flat_map(fn(oh) {
                  let #(h_start, h_end) = adaptive_range(oh, h_in, out_h)
                  list.range(0, out_w - 1)
                  |> list.map(fn(ow) {
                    let #(w_start, w_end) = adaptive_range(ow, w_in, out_w)
                    let count = { h_end - h_start } * { w_end - w_start }
                    let sum =
                      sum_2d_window(
                        arr,
                        base,
                        w_in,
                        h_start,
                        h_end,
                        w_start,
                        w_end,
                      )
                    sum /. int.to_float(count)
                  })
                })
              })
            })
          tensor.reshape(tensor.from_list(out), [batch, channels, out_h, out_w])
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "adaptive_avg_pool_2d: input must have shape [batch, channels, H, W], got "
        <> shape_to_string(shp),
      ))
  }
}

fn sum_2d_window(
  arr: ErlangArray,
  base: Int,
  w_in: Int,
  h_start: Int,
  h_end: Int,
  w_start: Int,
  w_end: Int,
) -> Float {
  sum_2d_rows(arr, base, w_in, h_start, h_end, w_start, w_end, 0.0)
}

fn sum_2d_rows(
  arr: ErlangArray,
  base: Int,
  w_in: Int,
  h: Int,
  h_end: Int,
  w_start: Int,
  w_end: Int,
  acc: Float,
) -> Float {
  case h >= h_end {
    True -> acc
    False -> {
      let row_sum = sum_2d_cols(arr, base, w_in, h, w_start, w_end, 0.0)
      sum_2d_rows(
        arr,
        base,
        w_in,
        h + 1,
        h_end,
        w_start,
        w_end,
        acc +. row_sum,
      )
    }
  }
}

fn sum_2d_cols(
  arr: ErlangArray,
  base: Int,
  w_in: Int,
  h: Int,
  w: Int,
  w_end: Int,
  acc: Float,
) -> Float {
  case w >= w_end {
    True -> acc
    False -> {
      let v = ffi.array_get(arr, base + h * w_in + w)
      sum_2d_cols(arr, base, w_in, h, w + 1, w_end, acc +. v)
    }
  }
}

// ---------------------------------------------------------------------------
// AdaptiveAvgPool1d
// ---------------------------------------------------------------------------

/// Forward pass for 1D adaptive average pooling.
///
/// `input` shape `[batch, channels, length]`. Output shape
/// `[batch, channels, output_size]`. Per output cell `i` the source window is
/// `[floor(i * L / output), ceil((i+1) * L / output))`; the cell value is
/// the unweighted mean.
pub fn adaptive_avg_pool_1d_forward(
  config: AdaptiveAvgPool1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(input)
  case shp {
    [batch, channels, length] -> {
      case config.output_size <= 0 {
        True ->
          Error(InvalidShape(
            "adaptive_avg_pool_1d: output_size must be positive, got "
            <> int.to_string(config.output_size),
          ))
        False -> {
          let arr = ffi.list_to_array(tensor.to_list(input))
          let out_size = config.output_size
          let out =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, channels - 1)
              |> list.flat_map(fn(c) {
                let base = b * channels * length + c * length
                list.range(0, out_size - 1)
                |> list.map(fn(o) {
                  let #(start, end) = adaptive_range(o, length, out_size)
                  let count = end - start
                  let sum = sum_1d_range(arr, base, start, end, 0.0)
                  sum /. int.to_float(count)
                })
              })
            })
          tensor.reshape(tensor.from_list(out), [batch, channels, out_size])
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "adaptive_avg_pool_1d: input must have shape [batch, channels, length], got "
        <> shape_to_string(shp),
      ))
  }
}

fn sum_1d_range(
  arr: ErlangArray,
  base: Int,
  i: Int,
  end: Int,
  acc: Float,
) -> Float {
  case i >= end {
    True -> acc
    False ->
      sum_1d_range(arr, base, i + 1, end, acc +. ffi.array_get(arr, base + i))
  }
}

/// Adaptive pooling window: PyTorch convention is
/// `start = floor(i * in_size / out_size)`,
/// `end   = ceil((i + 1) * in_size / out_size)`.
fn adaptive_range(i: Int, in_size: Int, out_size: Int) -> #(Int, Int) {
  let start = i * in_size / out_size
  let end = ceil_div({ i + 1 } * in_size, out_size)
  #(start, end)
}

fn ceil_div(a: Int, b: Int) -> Int {
  { a + b - 1 } / b
}

// ---------------------------------------------------------------------------
// Upsample
// ---------------------------------------------------------------------------

/// Forward pass for 2D upsampling.
///
/// `input` shape `[batch, channels, H, W]`. Output shape
/// `[batch, channels, H * scale_factor, W * scale_factor]`.
///
/// - `Nearest`: every output pixel takes the value of the nearest input
///   pixel (block-replicate).
/// - `Bilinear`: linear interpolation along both axes using the
///   `align_corners=False` convention — i.e. input/output pixel centers are
///   mapped as `src = (out + 0.5) / scale - 0.5`, clamped to `[0, H-1]`.
pub fn upsample_forward(
  config: UpsampleConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let shp = tensor.shape(input)
  case shp {
    [batch, channels, h_in, w_in] -> {
      case config.scale_factor <= 0 {
        True ->
          Error(InvalidShape(
            "upsample: scale_factor must be positive, got "
            <> int.to_string(config.scale_factor),
          ))
        False -> {
          let h_out = h_in * config.scale_factor
          let w_out = w_in * config.scale_factor
          let arr = ffi.list_to_array(tensor.to_list(input))
          let out = case config.mode {
            Nearest ->
              upsample_nearest_compute(
                arr,
                batch,
                channels,
                h_in,
                w_in,
                h_out,
                w_out,
                config.scale_factor,
              )
            Bilinear ->
              upsample_bilinear_compute(
                arr,
                batch,
                channels,
                h_in,
                w_in,
                h_out,
                w_out,
                config.scale_factor,
              )
          }
          tensor.reshape(tensor.from_list(out), [
            batch,
            channels,
            h_out,
            w_out,
          ])
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "upsample: input must have shape [batch, channels, H, W], got "
        <> shape_to_string(shp),
      ))
  }
}

fn upsample_nearest_compute(
  arr: ErlangArray,
  batch: Int,
  channels: Int,
  h_in: Int,
  w_in: Int,
  h_out: Int,
  w_out: Int,
  scale: Int,
) -> List(Float) {
  list.range(0, batch - 1)
  |> list.flat_map(fn(b) {
    list.range(0, channels - 1)
    |> list.flat_map(fn(c) {
      let base = b * channels * h_in * w_in + c * h_in * w_in
      list.range(0, h_out - 1)
      |> list.flat_map(fn(oh) {
        let ih = oh / scale
        list.range(0, w_out - 1)
        |> list.map(fn(ow) {
          let iw = ow / scale
          ffi.array_get(arr, base + ih * w_in + iw)
        })
      })
    })
  })
}

fn upsample_bilinear_compute(
  arr: ErlangArray,
  batch: Int,
  channels: Int,
  h_in: Int,
  w_in: Int,
  h_out: Int,
  w_out: Int,
  scale: Int,
) -> List(Float) {
  let scale_f = int.to_float(scale)
  list.range(0, batch - 1)
  |> list.flat_map(fn(b) {
    list.range(0, channels - 1)
    |> list.flat_map(fn(c) {
      let base = b * channels * h_in * w_in + c * h_in * w_in
      list.range(0, h_out - 1)
      |> list.flat_map(fn(oh) {
        // align_corners=False: src = (out + 0.5) / scale - 0.5
        let src_h = { int.to_float(oh) +. 0.5 } /. scale_f -. 0.5
        let #(h0, h1, dh) = interp_neighbors(src_h, h_in)
        list.range(0, w_out - 1)
        |> list.map(fn(ow) {
          let src_w = { int.to_float(ow) +. 0.5 } /. scale_f -. 0.5
          let #(w0, w1, dw) = interp_neighbors(src_w, w_in)
          let v00 = ffi.array_get(arr, base + h0 * w_in + w0)
          let v01 = ffi.array_get(arr, base + h0 * w_in + w1)
          let v10 = ffi.array_get(arr, base + h1 * w_in + w0)
          let v11 = ffi.array_get(arr, base + h1 * w_in + w1)
          let top = v00 *. { 1.0 -. dw } +. v01 *. dw
          let bot = v10 *. { 1.0 -. dw } +. v11 *. dw
          top *. { 1.0 -. dh } +. bot *. dh
        })
      })
    })
  })
}

/// Returns `(lo, hi, frac)` for a source coordinate so the interpolated value
/// is `v[lo] * (1 - frac) + v[hi] * frac`. Coordinates are clamped to
/// `[0, size - 1]` so edges replicate (matches PyTorch).
fn interp_neighbors(src: Float, size: Int) -> #(Int, Int, Float) {
  let clamped = case src <. 0.0 {
    True -> 0.0
    False ->
      case src >. int.to_float(size - 1) {
        True -> int.to_float(size - 1)
        False -> src
      }
  }
  let lo = float.truncate(clamped)
  let hi = case lo + 1 >= size {
    True -> size - 1
    False -> lo + 1
  }
  let frac = clamped -. int.to_float(lo)
  #(lo, hi, frac)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn shape_to_string(shape: List(Int)) -> String {
  "[" <> join_strings(list.map(shape, int.to_string), ", ") <> "]"
}

fn join_strings(parts: List(String), sep: String) -> String {
  case parts {
    [] -> ""
    [s] -> s
    [s, ..rest] -> s <> sep <> join_strings(rest, sep)
  }
}
