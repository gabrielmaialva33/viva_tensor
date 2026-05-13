//// Convolution variants (Conv1d, Conv3d, ConvTranspose2d) for VIVA tensor.
////
//// Pure-Gleam forward passes that match PyTorch's `nn.Conv1d`, `nn.Conv3d`
//// and `nn.ConvTranspose2d` semantics. No NIF, no autograd integration —
//// initial layer kit so models can be wired up before the native paths land.
////
//// Output shape formulas (PyTorch convention):
////   Conv1d:           L_out  = floor((L_in  + 2*P  - K)   / S) + 1
////   Conv3d:           D/H/W_out = floor((In + 2*P - K) / S) + 1
////   ConvTranspose2d:  Out    = (In - 1) * S - 2*P + (K - 1) + output_padding + 1
////
//// Weight layouts:
////   Conv1d:           [out_channels, in_channels, kernel_size]
////   Conv3d:           [out_channels, in_channels, kD, kH, kW]
////   ConvTranspose2d:  [in_channels, out_channels, kH, kW]   (transposed convention)

import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import viva_tensor/core/error.{type TensorError, InvalidShape}
import viva_tensor/core/ffi.{type ErlangArray}
import viva_tensor/tensor.{type Tensor}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Configuration for a 1D convolution layer.
///
/// `weight` is shaped `[out_channels, in_channels, kernel_size]` and `bias`,
/// when present, is shaped `[out_channels]`.
pub type Conv1dConfig {
  Conv1dConfig(
    in_channels: Int,
    out_channels: Int,
    kernel_size: Int,
    stride: Int,
    padding: Int,
    weight: Tensor,
    bias: Option(Tensor),
  )
}

/// Configuration for a 3D convolution layer.
///
/// `weight` is shaped `[out_channels, in_channels, kD, kH, kW]`.
pub type Conv3dConfig {
  Conv3dConfig(
    in_channels: Int,
    out_channels: Int,
    kernel_size: #(Int, Int, Int),
    stride: #(Int, Int, Int),
    padding: #(Int, Int, Int),
    weight: Tensor,
    bias: Option(Tensor),
  )
}

/// Configuration for a 2D transposed convolution (a.k.a. deconvolution).
///
/// Weight layout is `[in_channels, out_channels, kH, kW]` to match PyTorch's
/// `nn.ConvTranspose2d`.
pub type ConvTranspose2dConfig {
  ConvTranspose2dConfig(
    in_channels: Int,
    out_channels: Int,
    kernel_size: #(Int, Int),
    stride: #(Int, Int),
    padding: #(Int, Int),
    output_padding: #(Int, Int),
    weight: Tensor,
    bias: Option(Tensor),
  )
}

// ---------------------------------------------------------------------------
// Conv1d
// ---------------------------------------------------------------------------

/// Initialize a `Conv1dConfig` with zero weight and zero bias.
///
/// Output shape formula: `L_out = (L_in + 2*padding - kernel_size) / stride + 1`.
///
/// ## Example
///
/// ```gleam
/// let cfg = conv.conv1d_init(
///   in_channels: 3, out_channels: 8,
///   kernel_size: 3, stride: 1, padding: 1,
/// )
/// // cfg.weight has shape [8, 3, 3], cfg.bias is Some(zeros [8])
/// ```
pub fn conv1d_init(
  in_channels in_channels: Int,
  out_channels out_channels: Int,
  kernel_size kernel_size: Int,
  stride stride: Int,
  padding padding: Int,
) -> Conv1dConfig {
  Conv1dConfig(
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_size: kernel_size,
    stride: stride,
    padding: padding,
    weight: tensor.zeros([out_channels, in_channels, kernel_size]),
    bias: Some(tensor.zeros([out_channels])),
  )
}

/// Forward pass for 1D convolution.
///
/// `input` shape `[batch, in_channels, length]`, output shape
/// `[batch, out_channels, L_out]` where
/// `L_out = (length + 2*padding - kernel_size) / stride + 1`.
///
/// ## Example
///
/// ```gleam
/// let cfg = conv.conv1d_init(1, 1, 3, 1, 0)
/// let assert Ok(x) =
///   tensor.reshape(tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0]), [1, 1, 5])
/// let assert Ok(out) = conv.conv1d_forward(cfg, x)
/// // out has shape [1, 1, 3]
/// ```
pub fn conv1d_forward(
  config: Conv1dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let in_shape = tensor.shape(input)
  case in_shape {
    [batch, in_c, length] if in_c == config.in_channels -> {
      let kernel_size = config.kernel_size
      let stride = config.stride
      let padding = config.padding
      let padded_length = length + 2 * padding
      let out_length = { padded_length - kernel_size } / stride + 1
      case out_length > 0 && stride > 0 {
        False ->
          Error(InvalidShape(
            "conv1d_forward: invalid output length "
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
        True -> {
          let in_data = tensor.to_list(input)
          let w_data = tensor.to_list(config.weight)
          let bias_data = case config.bias {
            Some(b) -> tensor.to_list(b)
            None -> list.repeat(0.0, config.out_channels)
          }
          let padded = pad_1d(in_data, batch, in_c, length, padding)
          let out =
            conv1d_compute(
              padded,
              w_data,
              bias_data,
              batch,
              in_c,
              padded_length,
              config.out_channels,
              kernel_size,
              stride,
              out_length,
            )
          tensor.reshape(tensor.from_list(out), [
            batch,
            config.out_channels,
            out_length,
          ])
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "conv1d_forward: input must have shape [batch, "
        <> int.to_string(config.in_channels)
        <> ", length], got "
        <> shape_to_string(in_shape),
      ))
  }
}

fn pad_1d(
  data: List(Float),
  batch: Int,
  channels: Int,
  length: Int,
  padding: Int,
) -> List(Float) {
  case padding == 0 {
    True -> data
    False -> {
      let arr = ffi.list_to_array(data)
      let padded_length = length + 2 * padding
      list.range(0, batch - 1)
      |> list.flat_map(fn(b) {
        list.range(0, channels - 1)
        |> list.flat_map(fn(c) {
          let base = b * channels * length + c * length
          list.range(0, padded_length - 1)
          |> list.map(fn(i) {
            let src = i - padding
            case src >= 0 && src < length {
              True -> ffi.array_get(arr, base + src)
              False -> 0.0
            }
          })
        })
      })
    }
  }
}

fn conv1d_compute(
  in_data: List(Float),
  w_data: List(Float),
  bias_data: List(Float),
  batch: Int,
  in_c: Int,
  padded_length: Int,
  out_c: Int,
  kernel_size: Int,
  stride: Int,
  out_length: Int,
) -> List(Float) {
  let in_arr = ffi.list_to_array(in_data)
  let w_arr = ffi.list_to_array(w_data)
  let bias_arr = ffi.list_to_array(bias_data)
  list.range(0, batch - 1)
  |> list.flat_map(fn(b) {
    list.range(0, out_c - 1)
    |> list.flat_map(fn(oc) {
      let bias_v = ffi.array_get(bias_arr, oc)
      list.range(0, out_length - 1)
      |> list.map(fn(o) {
        let start = o * stride
        let sum =
          sum_over_range(0, in_c - 1, fn(ic) {
            sum_over_range(0, kernel_size - 1, fn(k) {
              let in_idx =
                b * in_c * padded_length + ic * padded_length + start + k
              let w_idx = oc * in_c * kernel_size + ic * kernel_size + k
              ffi.array_get(in_arr, in_idx) *. ffi.array_get(w_arr, w_idx)
            })
          })
        sum +. bias_v
      })
    })
  })
}

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

/// Initialize a `Conv3dConfig` with zero weight and zero bias.
///
/// Output shape formula (per spatial dim):
/// `D/H/W_out = (D/H/W_in + 2*pad - kernel) / stride + 1`.
///
/// ## Example
///
/// ```gleam
/// let cfg = conv.conv3d_init(
///   in_channels: 1, out_channels: 4,
///   kernel_size: #(3, 3, 3), stride: #(1, 1, 1), padding: #(0, 0, 0),
/// )
/// // cfg.weight has shape [4, 1, 3, 3, 3]
/// ```
pub fn conv3d_init(
  in_channels in_channels: Int,
  out_channels out_channels: Int,
  kernel_size kernel_size: #(Int, Int, Int),
  stride stride: #(Int, Int, Int),
  padding padding: #(Int, Int, Int),
) -> Conv3dConfig {
  let #(kd, kh, kw) = kernel_size
  Conv3dConfig(
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_size: kernel_size,
    stride: stride,
    padding: padding,
    weight: tensor.zeros([out_channels, in_channels, kd, kh, kw]),
    bias: Some(tensor.zeros([out_channels])),
  )
}

/// Forward pass for 3D convolution.
///
/// `input` shape `[batch, in_channels, depth, height, width]`, output shape
/// `[batch, out_channels, D_out, H_out, W_out]` where each spatial size is
/// `(in + 2*pad - kernel) / stride + 1`.
///
/// ## Example
///
/// ```gleam
/// let cfg = conv.conv3d_init(1, 1, #(2, 2, 2), #(1, 1, 1), #(0, 0, 0))
/// let assert Ok(x) = tensor.reshape(tensor.ones([8]), [1, 1, 2, 2, 2])
/// let assert Ok(_out) = conv.conv3d_forward(cfg, x)
/// ```
pub fn conv3d_forward(
  config: Conv3dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let in_shape = tensor.shape(input)
  case in_shape {
    [batch, in_c, depth, height, width] if in_c == config.in_channels -> {
      let #(kd, kh, kw) = config.kernel_size
      let #(sd, sh, sw) = config.stride
      let #(pd, ph, pw) = config.padding
      let padded_d = depth + 2 * pd
      let padded_h = height + 2 * ph
      let padded_w = width + 2 * pw
      let out_d = { padded_d - kd } / sd + 1
      let out_h = { padded_h - kh } / sh + 1
      let out_w = { padded_w - kw } / sw + 1
      case out_d > 0 && out_h > 0 && out_w > 0 && sd > 0 && sh > 0 && sw > 0 {
        False ->
          Error(InvalidShape(
            "conv3d_forward: invalid output dims ("
            <> int.to_string(out_d)
            <> ", "
            <> int.to_string(out_h)
            <> ", "
            <> int.to_string(out_w)
            <> ")",
          ))
        True -> {
          let in_data = tensor.to_list(input)
          let w_data = tensor.to_list(config.weight)
          let bias_data = case config.bias {
            Some(b) -> tensor.to_list(b)
            None -> list.repeat(0.0, config.out_channels)
          }
          let padded =
            pad_3d_internal(
              in_data,
              batch,
              in_c,
              depth,
              height,
              width,
              pd,
              ph,
              pw,
            )
          let in_arr = ffi.list_to_array(padded)
          let w_arr = ffi.list_to_array(w_data)
          let bias_arr = ffi.list_to_array(bias_data)
          let out =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, config.out_channels - 1)
              |> list.flat_map(fn(oc) {
                let bias_v = ffi.array_get(bias_arr, oc)
                list.range(0, out_d - 1)
                |> list.flat_map(fn(od) {
                  list.range(0, out_h - 1)
                  |> list.flat_map(fn(oh) {
                    list.range(0, out_w - 1)
                    |> list.map(fn(ow) {
                      let sd_start = od * sd
                      let sh_start = oh * sh
                      let sw_start = ow * sw
                      let sum =
                        sum_over_range(0, in_c - 1, fn(ic) {
                          sum_over_range(0, kd - 1, fn(zk) {
                            sum_over_range(0, kh - 1, fn(yk) {
                              sum_over_range(0, kw - 1, fn(xk) {
                                let in_idx =
                                  b
                                  * in_c
                                  * padded_d
                                  * padded_h
                                  * padded_w
                                  + ic
                                  * padded_d
                                  * padded_h
                                  * padded_w
                                  + { sd_start + zk }
                                  * padded_h
                                  * padded_w
                                  + { sh_start + yk }
                                  * padded_w
                                  + sw_start
                                  + xk
                                let w_idx =
                                  oc
                                  * in_c
                                  * kd
                                  * kh
                                  * kw
                                  + ic
                                  * kd
                                  * kh
                                  * kw
                                  + zk
                                  * kh
                                  * kw
                                  + yk
                                  * kw
                                  + xk
                                ffi.array_get(in_arr, in_idx)
                                *. ffi.array_get(w_arr, w_idx)
                              })
                            })
                          })
                        })
                      sum +. bias_v
                    })
                  })
                })
              })
            })
          tensor.reshape(tensor.from_list(out), [
            batch,
            config.out_channels,
            out_d,
            out_h,
            out_w,
          ])
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "conv3d_forward: input must have shape [batch, "
        <> int.to_string(config.in_channels)
        <> ", depth, height, width], got "
        <> shape_to_string(in_shape),
      ))
  }
}

fn pad_3d_internal(
  data: List(Float),
  batch: Int,
  channels: Int,
  depth: Int,
  height: Int,
  width: Int,
  pd: Int,
  ph: Int,
  pw: Int,
) -> List(Float) {
  case pd == 0 && ph == 0 && pw == 0 {
    True -> data
    False -> {
      let arr = ffi.list_to_array(data)
      let padded_d = depth + 2 * pd
      let padded_h = height + 2 * ph
      let padded_w = width + 2 * pw
      list.range(0, batch - 1)
      |> list.flat_map(fn(b) {
        list.range(0, channels - 1)
        |> list.flat_map(fn(c) {
          let base =
            b * channels * depth * height * width + c * depth * height * width
          list.range(0, padded_d - 1)
          |> list.flat_map(fn(z) {
            list.range(0, padded_h - 1)
            |> list.flat_map(fn(y) {
              list.range(0, padded_w - 1)
              |> list.map(fn(x) {
                let sz = z - pd
                let sy = y - ph
                let sx = x - pw
                case
                  sz >= 0
                  && sz < depth
                  && sy >= 0
                  && sy < height
                  && sx >= 0
                  && sx < width
                {
                  True ->
                    ffi.array_get(
                      arr,
                      base + sz * height * width + sy * width + sx,
                    )
                  False -> 0.0
                }
              })
            })
          })
        })
      })
    }
  }
}

// ---------------------------------------------------------------------------
// ConvTranspose2d
// ---------------------------------------------------------------------------

/// Initialize a `ConvTranspose2dConfig` with zero weight and zero bias.
///
/// Output shape formula:
/// `Out = (In - 1) * stride - 2*padding + (kernel - 1) + output_padding + 1`.
///
/// ## Example
///
/// ```gleam
/// let cfg = conv.conv_transpose_2d_init(
///   in_channels: 1, out_channels: 1,
///   kernel_size: #(2, 2), stride: #(1, 1),
///   padding: #(0, 0), output_padding: #(0, 0),
/// )
/// // cfg.weight has shape [1, 1, 2, 2]
/// ```
pub fn conv_transpose_2d_init(
  in_channels in_channels: Int,
  out_channels out_channels: Int,
  kernel_size kernel_size: #(Int, Int),
  stride stride: #(Int, Int),
  padding padding: #(Int, Int),
  output_padding output_padding: #(Int, Int),
) -> ConvTranspose2dConfig {
  let #(kh, kw) = kernel_size
  ConvTranspose2dConfig(
    in_channels: in_channels,
    out_channels: out_channels,
    kernel_size: kernel_size,
    stride: stride,
    padding: padding,
    output_padding: output_padding,
    weight: tensor.zeros([in_channels, out_channels, kh, kw]),
    bias: Some(tensor.zeros([out_channels])),
  )
}

/// Forward pass for 2D transposed convolution.
///
/// `input` shape `[batch, in_channels, H_in, W_in]`, output shape
/// `[batch, out_channels, H_out, W_out]` where
/// `H_out = (H_in - 1) * stride_h - 2*pad_h + (kH - 1) + out_pad_h + 1`
/// and likewise for width.
///
/// ## Example
///
/// ```gleam
/// let cfg =
///   conv.conv_transpose_2d_init(1, 1, #(2, 2), #(1, 1), #(0, 0), #(0, 0))
/// let assert Ok(x) = tensor.reshape(tensor.ones([4]), [1, 1, 2, 2])
/// let assert Ok(out) = conv.conv_transpose_2d_forward(cfg, x)
/// // out has shape [1, 1, 3, 3]
/// ```
pub fn conv_transpose_2d_forward(
  config: ConvTranspose2dConfig,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let in_shape = tensor.shape(input)
  case in_shape {
    [batch, in_c, h_in, w_in] if in_c == config.in_channels -> {
      let #(kh, kw) = config.kernel_size
      let #(sh, sw) = config.stride
      let #(ph, pw) = config.padding
      let #(oph, opw) = config.output_padding
      let h_out = { h_in - 1 } * sh - 2 * ph + { kh - 1 } + oph + 1
      let w_out = { w_in - 1 } * sw - 2 * pw + { kw - 1 } + opw + 1
      case h_out > 0 && w_out > 0 && sh > 0 && sw > 0 {
        False ->
          Error(InvalidShape(
            "conv_transpose_2d_forward: invalid output dims ("
            <> int.to_string(h_out)
            <> ", "
            <> int.to_string(w_out)
            <> ")",
          ))
        True -> {
          let in_data = tensor.to_list(input)
          let w_data = tensor.to_list(config.weight)
          let bias_data = case config.bias {
            Some(b) -> tensor.to_list(b)
            None -> list.repeat(0.0, config.out_channels)
          }
          let in_arr = ffi.list_to_array(in_data)
          let w_arr = ffi.list_to_array(w_data)
          let bias_arr = ffi.list_to_array(bias_data)
          let out_c = config.out_channels
          // Pre-crop "full" buffer dims (before padding/output_padding cropping).
          let h_full = { h_in - 1 } * sh + kh
          let w_full = { w_in - 1 } * sw + kw
          let acc =
            scatter_transpose_2d(
              in_arr,
              w_arr,
              batch,
              in_c,
              out_c,
              h_in,
              w_in,
              kh,
              kw,
              sh,
              sw,
              h_full,
              w_full,
            )
          let acc_arr = ffi.list_to_array(acc)
          let out =
            list.range(0, batch - 1)
            |> list.flat_map(fn(b) {
              list.range(0, out_c - 1)
              |> list.flat_map(fn(oc) {
                let bias_v = ffi.array_get(bias_arr, oc)
                list.range(0, h_out - 1)
                |> list.flat_map(fn(y) {
                  list.range(0, w_out - 1)
                  |> list.map(fn(x) {
                    let src_y = y + ph
                    let src_x = x + pw
                    case
                      src_y >= 0
                      && src_y < h_full
                      && src_x >= 0
                      && src_x < w_full
                    {
                      True -> {
                        let idx =
                          b
                          * out_c
                          * h_full
                          * w_full
                          + oc
                          * h_full
                          * w_full
                          + src_y
                          * w_full
                          + src_x
                        ffi.array_get(acc_arr, idx) +. bias_v
                      }
                      False -> bias_v
                    }
                  })
                })
              })
            })
          tensor.reshape(tensor.from_list(out), [batch, out_c, h_out, w_out])
        }
      }
    }
    _ ->
      Error(InvalidShape(
        "conv_transpose_2d_forward: input must have shape [batch, "
        <> int.to_string(config.in_channels)
        <> ", height, width], got "
        <> shape_to_string(in_shape),
      ))
  }
}

fn scatter_transpose_2d(
  in_arr: ErlangArray,
  w_arr: ErlangArray,
  batch: Int,
  in_c: Int,
  out_c: Int,
  h_in: Int,
  w_in: Int,
  kh: Int,
  kw: Int,
  sh: Int,
  sw: Int,
  h_full: Int,
  w_full: Int,
) -> List(Float) {
  list.range(0, batch - 1)
  |> list.flat_map(fn(b) {
    list.range(0, out_c - 1)
    |> list.flat_map(fn(oc) {
      list.range(0, h_full - 1)
      |> list.flat_map(fn(y) {
        list.range(0, w_full - 1)
        |> list.map(fn(x) {
          sum_over_range(0, in_c - 1, fn(ic) {
            // gather contributions from kernel positions
            sum_over_range(0, kh - 1, fn(ky) {
              let dy = y - ky
              case dy >= 0 && dy % sh == 0 {
                False -> 0.0
                True -> {
                  let ih = dy / sh
                  case ih >= 0 && ih < h_in {
                    False -> 0.0
                    True ->
                      sum_over_range(0, kw - 1, fn(kx) {
                        let dx = x - kx
                        case dx >= 0 && dx % sw == 0 {
                          False -> 0.0
                          True -> {
                            let iw = dx / sw
                            case iw >= 0 && iw < w_in {
                              False -> 0.0
                              True -> {
                                let in_idx =
                                  b
                                  * in_c
                                  * h_in
                                  * w_in
                                  + ic
                                  * h_in
                                  * w_in
                                  + ih
                                  * w_in
                                  + iw
                                let w_idx =
                                  ic
                                  * out_c
                                  * kh
                                  * kw
                                  + oc
                                  * kh
                                  * kw
                                  + ky
                                  * kw
                                  + kx
                                ffi.array_get(in_arr, in_idx)
                                *. ffi.array_get(w_arr, w_idx)
                              }
                            }
                          }
                        }
                      })
                  }
                }
              }
            })
          })
        })
      })
    })
  })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sum_over_range(start: Int, end: Int, f: fn(Int) -> Float) -> Float {
  sum_range_acc(start, end, f, 0.0)
}

fn sum_range_acc(
  start: Int,
  end: Int,
  f: fn(Int) -> Float,
  acc: Float,
) -> Float {
  case start > end {
    True -> acc
    False -> sum_range_acc(start + 1, end, f, acc +. f(start))
  }
}

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
