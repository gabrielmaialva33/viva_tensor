//// Quantization-Aware Training (QAT) primitives.
////
//// This module provides **standalone fake-quant forward + Straight-Through
//// Estimator (STE) backward** functions. They are designed to be wired into a
//// future Tape-based autograd; for now they expose pure Gleam helpers usable
//// from calibration scripts and unit tests.
////
//// --- The Quantization Formula ---
////
//// Symmetric uniform (int8 default):
////   qmin = -(2^(num_bits-1) - 1)        e.g. -127
////   qmax =  (2^(num_bits-1) - 1)        e.g.  127
////   scale = max(|x|) / qmax
////   zero_point = 0
////
//// Asymmetric uniform (uint8 default):
////   qmin = 0
////   qmax = 2^num_bits - 1                e.g.  255
////   scale = (max(x) - min(x)) / (qmax - qmin)
////   zero_point = round(qmin - min(x) / scale)
////
//// Forward (fake-quant):
////   q = clamp(round(x / scale + zero_point), qmin, qmax)
////   y = (q - zero_point) * scale
////
//// Backward (Straight-Through Estimator):
////   dL/dx = dL/dy * 1{x/scale + zero_point ∈ [qmin, qmax]}
////
//// This is the canonical recipe from Jacob et al. (2017) and used by
//// PyTorch's `torch.ao.quantization` fake-quant modules.

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import viva_tensor/core/error.{type TensorError, DimensionError, InvalidShape}
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Configuration --------------------------------------------------------

/// Configuration for a fake-quant op.
///
/// * `num_bits`     — width of the quantized integer (2..8 typical).
/// * `symmetric`    — when True, uses `[-qmax, qmax]` int range with
///                    `zero_point = 0`. When False, uses `[0, 2^bits - 1]`
///                    asymmetric range with computed zero point.
/// * `per_channel`  — when True, statistics are computed per-channel along
///                    `channel_axis`; otherwise a single scalar scale is used.
/// * `channel_axis` — axis to reduce over for per-channel stats (ignored
///                    when `per_channel` is False).
pub type QuantConfig {
  QuantConfig(
    num_bits: Int,
    symmetric: Bool,
    per_channel: Bool,
    channel_axis: Int,
  )
}

/// Calibration statistics produced by `observe`.
///
/// Shapes:
/// * tensor-wide → `[1]`
/// * per-channel → `[C]` (length of the `channel_axis` dimension)
pub type QuantStats {
  QuantStats(scale: Tensor, zero_point: Tensor)
}

// --- Quant range ----------------------------------------------------------

/// Return `(qmin, qmax)` as floats for the given config.
///
/// Symmetric int8 uses `[-127, 127]` (qmax = 2^(b-1) - 1) and discards the
/// `-128` value so dequantized zero remains exactly representable.
/// Asymmetric uint8 uses `[0, 255]` (qmax = 2^b - 1).
fn quant_range(config: QuantConfig) -> #(Float, Float) {
  case config.symmetric {
    True -> {
      let qmax = int.to_float(pow2(config.num_bits - 1) - 1)
      #(0.0 -. qmax, qmax)
    }
    False -> {
      let qmax = int.to_float(pow2(config.num_bits) - 1)
      #(0.0, qmax)
    }
  }
}

fn pow2(exp: Int) -> Int {
  case exp {
    e if e <= 0 -> 1
    _ -> 2 * pow2(exp - 1)
  }
}

// --- observe --------------------------------------------------------------

/// Compute scale and zero-point statistics from an input tensor.
///
/// For symmetric configs:
///   `scale = max(|x|) / qmax`, `zero_point = 0`
///
/// For asymmetric configs:
///   `scale = (max(x) - min(x)) / (qmax - qmin)`
///   `zero_point = round(qmin - min(x) / scale)` clamped to `[qmin, qmax]`
///
/// Per-channel reduces over every axis except `channel_axis`.
pub fn observe(
  input: Tensor,
  config: QuantConfig,
) -> Result(QuantStats, TensorError) {
  let data = tensor.to_list(input)
  case data {
    [] -> Error(InvalidShape("observe: empty tensor"))
    _ -> {
      case config.per_channel {
        False -> observe_tensor_wide(data, config)
        True -> observe_per_channel(input, data, config)
      }
    }
  }
}

fn observe_tensor_wide(
  data: List(Float),
  config: QuantConfig,
) -> Result(QuantStats, TensorError) {
  let #(qmin, qmax) = quant_range(config)
  let #(min_v, max_v) = min_max(data)
  let #(scale, zp) =
    compute_scale_zp(min_v, max_v, qmin, qmax, config.symmetric)
  Ok(QuantStats(
    scale: Tensor(data: [scale], shape: [1]),
    zero_point: Tensor(data: [int.to_float(zp)], shape: [1]),
  ))
}

fn observe_per_channel(
  input: Tensor,
  data: List(Float),
  config: QuantConfig,
) -> Result(QuantStats, TensorError) {
  let shape = tensor.shape(input)
  let rank = list.length(shape)
  let axis = config.channel_axis
  case axis < 0 || axis >= rank {
    True ->
      Error(DimensionError(
        "observe: channel_axis out of bounds for tensor rank",
      ))
    False -> {
      use channels <- result.try(split_along_axis(data, shape, axis))
      let #(qmin, qmax) = quant_range(config)
      let pairs =
        list.map(channels, fn(values) {
          let #(min_v, max_v) = min_max(values)
          compute_scale_zp(min_v, max_v, qmin, qmax, config.symmetric)
        })
      let scales = list.map(pairs, fn(p) { p.0 })
      let zps = list.map(pairs, fn(p) { int.to_float(p.1) })
      let c = list.length(scales)
      Ok(QuantStats(
        scale: Tensor(data: scales, shape: [c]),
        zero_point: Tensor(data: zps, shape: [c]),
      ))
    }
  }
}

fn compute_scale_zp(
  min_v: Float,
  max_v: Float,
  qmin: Float,
  qmax: Float,
  symmetric: Bool,
) -> #(Float, Int) {
  case symmetric {
    True -> {
      let abs_max =
        float.max(float.absolute_value(min_v), float.absolute_value(max_v))
      let scale = case abs_max >. 0.0 {
        True -> abs_max /. qmax
        False -> 1.0
      }
      #(scale, 0)
    }
    False -> {
      let range = max_v -. min_v
      let q_range = qmax -. qmin
      let scale = case range >. 0.0 {
        True -> range /. q_range
        False -> 1.0
      }
      let zp_float = qmin -. min_v /. scale
      let zp_rounded = float.round(zp_float)
      let zp_clamped =
        int.max(int.min(zp_rounded, float_to_int(qmax)), float_to_int(qmin))
      #(scale, zp_clamped)
    }
  }
}

fn float_to_int(f: Float) -> Int {
  float.round(f)
}

fn min_max(values: List(Float)) -> #(Float, Float) {
  case values {
    [] -> #(0.0, 0.0)
    [first, ..rest] ->
      list.fold(rest, #(first, first), fn(acc, v) {
        let #(lo, hi) = acc
        let new_lo = case v <. lo {
          True -> v
          False -> lo
        }
        let new_hi = case v >. hi {
          True -> v
          False -> hi
        }
        #(new_lo, new_hi)
      })
  }
}

/// Split flat data into `C` lists where `C = shape[axis]`. Each output list
/// contains every element whose multi-index has the matching coordinate at
/// `axis`. Used to compute per-channel statistics without rearranging memory.
fn split_along_axis(
  data: List(Float),
  shape: List(Int),
  axis: Int,
) -> Result(List(List(Float)), TensorError) {
  let c = case list.drop(shape, axis) {
    [size, ..] -> size
    [] -> 0
  }
  case c <= 0 {
    True -> Error(InvalidShape("observe: channel axis has size 0"))
    False -> {
      let outer = product_of(list.take(shape, axis))
      let inner = product_of(list.drop(shape, axis + 1))
      let buckets = list.repeat([], c)
      let indexed =
        data
        |> list.index_map(fn(value, idx) {
          let channel = idx / inner % c
          let _ = outer
          #(channel, value)
        })
      let filled =
        list.range(0, c - 1)
        |> list.map(fn(ch) {
          indexed
          |> list.filter(fn(pair) { pair.0 == ch })
          |> list.map(fn(pair) { pair.1 })
        })
      let _ = buckets
      Ok(filled)
    }
  }
}

fn product_of(dims: List(Int)) -> Int {
  list.fold(dims, 1, fn(acc, d) { acc * d })
}

// --- fake_quant_forward ---------------------------------------------------

/// Fake-quantize: quantize then dequantize.
///
/// Forward pass:
///   `q = clamp(round(x / scale + zero_point), qmin, qmax)`
///   `y = (q - zero_point) * scale`
///
/// Per-channel scales are broadcast along every axis except `channel_axis`.
pub fn fake_quant_forward(
  input: Tensor,
  stats: QuantStats,
  config: QuantConfig,
) -> Result(Tensor, TensorError) {
  let data = tensor.to_list(input)
  let shape = tensor.shape(input)
  let scales = tensor.to_list(stats.scale)
  let zps = tensor.to_list(stats.zero_point)
  let #(qmin, qmax) = quant_range(config)
  use scale_per_elem <- result.try(broadcast_per_element(
    shape,
    scales,
    zps,
    config,
  ))
  let pairs = list.zip(data, scale_per_elem)
  let out =
    list.map(pairs, fn(pair) {
      let #(x, sz) = pair
      let #(scale, zp) = sz
      let safe_scale = case scale >. 0.0 {
        True -> scale
        False -> 1.0
      }
      let q = float.round(x /. safe_scale +. zp)
      let q_clamped =
        int.max(int.min(q, float_to_int(qmax)), float_to_int(qmin))
      { int.to_float(q_clamped) -. zp } *. safe_scale
    })
  Ok(Tensor(data: out, shape: shape))
}

// --- fake_quant_backward --------------------------------------------------

/// Straight-Through Estimator backward.
///
/// `dL/dx = dL/dy` where `qmin <= round(x / scale + zero_point) <= qmax`,
/// and `0` elsewhere. This mirrors the standard PyTorch behavior of zeroing
/// the gradient on values that would have been clipped during the forward
/// pass.
pub fn fake_quant_backward(
  grad_out: Tensor,
  input: Tensor,
  stats: QuantStats,
  config: QuantConfig,
) -> Result(Tensor, TensorError) {
  let grad_data = tensor.to_list(grad_out)
  let in_data = tensor.to_list(input)
  let shape = tensor.shape(input)
  case list.length(grad_data) == list.length(in_data) {
    False ->
      Error(InvalidShape("fake_quant_backward: grad/input shape mismatch"))
    True -> {
      let scales = tensor.to_list(stats.scale)
      let zps = tensor.to_list(stats.zero_point)
      let #(qmin, qmax) = quant_range(config)
      use scale_per_elem <- result.try(broadcast_per_element(
        shape,
        scales,
        zps,
        config,
      ))
      let triples = list.zip(grad_data, list.zip(in_data, scale_per_elem))
      let out =
        list.map(triples, fn(t) {
          let #(g, rest) = t
          let #(x, sz) = rest
          let #(scale, zp) = sz
          let safe_scale = case scale >. 0.0 {
            True -> scale
            False -> 1.0
          }
          let q = x /. safe_scale +. zp
          case q >=. qmin && q <=. qmax {
            True -> g
            False -> 0.0
          }
        })
      Ok(Tensor(data: out, shape: shape))
    }
  }
}

/// Return a list of `#(scale, zero_point)` of length `prod(shape)`, broadcast
/// from per-channel or scalar stats according to `config`.
fn broadcast_per_element(
  shape: List(Int),
  scales: List(Float),
  zps: List(Float),
  config: QuantConfig,
) -> Result(List(#(Float, Float)), TensorError) {
  let total = product_of(shape)
  case config.per_channel {
    False -> {
      let scale = case scales {
        [s, ..] -> s
        [] -> 1.0
      }
      let zp = case zps {
        [z, ..] -> z
        [] -> 0.0
      }
      Ok(list.repeat(#(scale, zp), total))
    }
    True -> {
      let rank = list.length(shape)
      let axis = config.channel_axis
      case axis < 0 || axis >= rank {
        True ->
          Error(DimensionError(
            "fake_quant: channel_axis out of bounds for tensor rank",
          ))
        False -> {
          let c = case list.drop(shape, axis) {
            [size, ..] -> size
            [] -> 0
          }
          let inner = product_of(list.drop(shape, axis + 1))
          let scale_arr = scales
          let zp_arr = zps
          let result =
            list.range(0, total - 1)
            |> list.map(fn(idx) {
              let channel = idx / inner % c
              let s = nth(scale_arr, channel, 1.0)
              let z = nth(zp_arr, channel, 0.0)
              #(s, z)
            })
          Ok(result)
        }
      }
    }
  }
}

fn nth(values: List(Float), index: Int, default: Float) -> Float {
  case list.drop(values, index) {
    [v, ..] -> v
    [] -> default
  }
}

// --- compute_per_channel_scales -------------------------------------------

/// Compute per-channel symmetric scales from a weight tensor's
/// max-absolute value along the non-channel axes.
///
/// `scale[c] = max(|W[c, ...]|) / qmax` where `qmax = 2^(num_bits-1) - 1`.
///
/// Returns a 1-D tensor `[C]` where `C = shape[channel_axis]`.
pub fn compute_per_channel_scales(
  weight: Tensor,
  num_bits: Int,
  channel_axis: Int,
) -> Result(Tensor, TensorError) {
  let shape = tensor.shape(weight)
  let rank = list.length(shape)
  case channel_axis < 0 || channel_axis >= rank {
    True ->
      Error(DimensionError(
        "compute_per_channel_scales: channel_axis out of bounds",
      ))
    False -> {
      let data = tensor.to_list(weight)
      use channels <- result.try(split_along_axis(data, shape, channel_axis))
      let qmax = int.to_float(pow2(num_bits - 1) - 1)
      let scales =
        list.map(channels, fn(values) {
          let abs_max =
            values
            |> list.map(float.absolute_value)
            |> list.fold(0.0, float.max)
          case abs_max >. 0.0 {
            True -> abs_max /. qmax
            False -> 1.0
          }
        })
      Ok(Tensor(data: scales, shape: [list.length(scales)]))
    }
  }
}

// --- QatLinear ------------------------------------------------------------

/// QAT-aware Linear layer.
///
/// * `weight`         — `[out_features, in_features]` weight matrix.
/// * `bias`           — optional `[out_features]` bias vector.
/// * `weight_stats`   — per-channel scales for the weight (axis 0).
/// * `weight_config`  — quant config used to derive `weight_stats`.
/// * `input_stats`    — running activation stats (set via `qat_linear_calibrate`).
/// * `input_config`   — quant config for activations (tensor-wide by default).
pub type QatLinear {
  QatLinear(
    weight: Tensor,
    bias: Option(Tensor),
    weight_stats: QuantStats,
    weight_config: QuantConfig,
    input_stats: Option(QuantStats),
    input_config: QuantConfig,
  )
}

/// Initialize a QAT linear layer with zero weights/bias and a default
/// per-channel symmetric weight config plus tensor-wide symmetric
/// activation config. Activation stats start empty until calibration.
pub fn qat_linear_init(
  in_features: Int,
  out_features: Int,
  weight_bits: Int,
  activation_bits: Int,
) -> QatLinear {
  let weight = tensor.zeros([out_features, in_features])
  let bias = tensor.zeros([out_features])
  let weight_config =
    QuantConfig(
      num_bits: weight_bits,
      symmetric: True,
      per_channel: True,
      channel_axis: 0,
    )
  let input_config =
    QuantConfig(
      num_bits: activation_bits,
      symmetric: True,
      per_channel: False,
      channel_axis: 0,
    )
  let zero_scale =
    Tensor(data: list.repeat(1.0, out_features), shape: [
      out_features,
    ])
  let zero_zp =
    Tensor(data: list.repeat(0.0, out_features), shape: [out_features])
  let weight_stats = QuantStats(scale: zero_scale, zero_point: zero_zp)
  QatLinear(
    weight: weight,
    bias: Some(bias),
    weight_stats: weight_stats,
    weight_config: weight_config,
    input_stats: None,
    input_config: input_config,
  )
}

/// Forward pass: fake-quantize the weights (always) and the input
/// (when `input_stats` is populated), then perform a matmul plus optional
/// bias.
///
/// Output shape: `[batch, out_features]` for a `[batch, in_features]` input.
pub fn qat_linear_forward(
  layer: QatLinear,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  use fq_weight <- result.try(fake_quant_forward(
    layer.weight,
    layer.weight_stats,
    layer.weight_config,
  ))
  let fq_input_result = case layer.input_stats {
    Some(stats) -> fake_quant_forward(input, stats, layer.input_config)
    None -> Ok(input)
  }
  use fq_input <- result.try(fq_input_result)
  use w_t <- result.try(tensor.transpose(fq_weight))
  use out <- result.try(tensor.matmul(fq_input, w_t))
  case layer.bias {
    None -> Ok(out)
    Some(b) -> add_bias_row(out, b)
  }
}

fn add_bias_row(out: Tensor, bias: Tensor) -> Result(Tensor, TensorError) {
  let shape = tensor.shape(out)
  case shape {
    [batch, features] -> {
      let bias_data = tensor.to_list(bias)
      case list.length(bias_data) == features {
        False -> Error(InvalidShape("qat_linear: bias length != out_features"))
        True -> {
          let data = tensor.to_list(out)
          let rows = list.sized_chunk(data, features)
          let added =
            list.flat_map(rows, fn(row) {
              list.map2(row, bias_data, fn(x, b) { x +. b })
            })
          Ok(Tensor(data: added, shape: [batch, features]))
        }
      }
    }
    _ -> Error(DimensionError("qat_linear: expected [batch, out] output"))
  }
}

/// Update `input_stats` by observing the given calibration batch.
///
/// Returns a new `QatLinear` with refreshed activation statistics.
pub fn qat_linear_calibrate(
  layer: QatLinear,
  input: Tensor,
) -> Result(QatLinear, TensorError) {
  use stats <- result.try(observe(input, layer.input_config))
  Ok(QatLinear(..layer, input_stats: Some(stats)))
}
