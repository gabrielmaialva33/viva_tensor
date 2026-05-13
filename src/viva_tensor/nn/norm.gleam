//// Normalization layers for neural networks.
////
//// Pure Gleam forward passes. No autograd integration in this round — each
//// layer is a value carrying its learnable parameters (scale, bias) plus
//// configuration (eps, momentum, num_groups, running stats).
////
//// Provided layers:
//// - `LayerNorm`     — normalizes along the last dimension
//// - `RmsNorm`       — root-mean-square norm (no mean subtraction)
//// - `BatchNorm1d`   — normalizes along the batch axis with running stats
//// - `GroupNorm`     — splits channels into groups and normalizes per group
////
//// References:
//// - Ba, Kiros & Hinton (2016). "Layer Normalization."
//// - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization."
//// - Ioffe & Szegedy (2015). "Batch Normalization."
//// - Wu & He (2018). "Group Normalization."

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape, ShapeMismatch}
import viva_tensor/tensor.{type Tensor, Tensor}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// LayerNorm layer. Normalizes along the **last** dimension of the input.
///
/// Math: `y = (x - E[x]) / sqrt(Var[x] + eps) * scale + bias`,
/// where the mean and variance are computed over the last axis.
///
/// `scale` and `bias` are 1D tensors of size `num_features` (the size of the
/// last dimension that this layer normalizes).
pub type LayerNorm {
  LayerNorm(scale: Tensor, bias: Tensor, eps: Float)
}

/// Create a `LayerNorm` with `scale = ones([num_features])`,
/// `bias = zeros([num_features])`, and the default `eps = 1.0e-5`.
///
/// Formula: `y = (x - mean) / sqrt(var + eps) * scale + bias`.
///
/// ## Example
/// ```gleam
/// let layer = layer_norm_init(4)
/// // layer.scale == ones([4])
/// // layer.bias  == zeros([4])
/// ```
pub fn layer_norm_init(num_features: Int) -> LayerNorm {
  layer_norm_init_with_eps(num_features, 1.0e-5)
}

/// Create a `LayerNorm` with a custom `eps`.
///
/// Formula: `y = (x - mean) / sqrt(var + eps) * scale + bias`.
///
/// ## Example
/// ```gleam
/// let layer = layer_norm_init_with_eps(4, 1.0e-6)
/// ```
pub fn layer_norm_init_with_eps(num_features: Int, eps: Float) -> LayerNorm {
  LayerNorm(
    scale: tensor.ones([num_features]),
    bias: tensor.zeros([num_features]),
    eps: eps,
  )
}

/// Forward pass for `LayerNorm`. Normalizes along the last dimension.
///
/// Math: for each row along the last axis,
/// `y_i = (x_i - mean) / sqrt(var + eps) * scale_i + bias_i`.
///
/// Returns `ShapeMismatch` if the input's last dimension does not match
/// `num_features` (the size of `layer.scale`).
///
/// ## Example
/// ```gleam
/// let layer = layer_norm_init(4)
/// let x = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0]])
/// let assert Ok(x) = x
/// let assert Ok(_y) = layer_norm_forward(layer, x)
/// ```
pub fn layer_norm_forward(
  layer: LayerNorm,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let input_shape = tensor.shape(input)
  let scale_shape = tensor.shape(layer.scale)
  use num_features <- result.try(last_dim(scale_shape))
  use input_last <- result.try(last_dim(input_shape))
  case input_last == num_features {
    False -> Error(ShapeMismatch(expected: [num_features], got: [input_last]))
    True -> {
      use data <- result.try(tensor.try_to_list(input))
      use scale_data <- result.try(tensor.try_to_list(layer.scale))
      use bias_data <- result.try(tensor.try_to_list(layer.bias))
      let normalized =
        chunk_by(data, num_features)
        |> list.map(fn(chunk) {
          let mean = list_mean(chunk)
          let var = list_variance(chunk, mean)
          let denom = safe_sqrt(var +. layer.eps)
          // (x - mean) / denom * scale + bias  (per-feature scale/bias)
          list.map(list.zip(chunk, list.zip(scale_data, bias_data)), fn(t) {
            let #(x, sb) = t
            let #(s, b) = sb
            { x -. mean } /. denom *. s +. b
          })
        })
        |> list.flatten
      Ok(Tensor(data: normalized, shape: input_shape))
    }
  }
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// RMSNorm layer. Normalizes along the last dimension using the root mean
/// square, without subtracting the mean. Popular in modern LLMs (Llama, etc.).
///
/// Math: `y = x / sqrt(mean(x^2) + eps) * scale`.
pub type RmsNorm {
  RmsNorm(scale: Tensor, eps: Float)
}

/// Create an `RmsNorm` with `scale = ones([num_features])` and the default
/// `eps = 1.0e-6`.
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * scale`.
///
/// ## Example
/// ```gleam
/// let layer = rms_norm_init(4)
/// ```
pub fn rms_norm_init(num_features: Int) -> RmsNorm {
  rms_norm_init_with_eps(num_features, 1.0e-6)
}

/// Create an `RmsNorm` with a custom `eps`.
///
/// Formula: `y = x / sqrt(mean(x^2) + eps) * scale`.
///
/// ## Example
/// ```gleam
/// let layer = rms_norm_init_with_eps(4, 1.0e-5)
/// ```
pub fn rms_norm_init_with_eps(num_features: Int, eps: Float) -> RmsNorm {
  RmsNorm(scale: tensor.ones([num_features]), eps: eps)
}

/// Forward pass for `RmsNorm`. Normalizes along the last dimension.
///
/// Math: `rms = sqrt(mean(x^2) + eps)`, then `y_i = x_i / rms * scale_i`.
///
/// Returns `ShapeMismatch` if the input's last dimension does not match
/// `num_features`.
///
/// ## Example
/// ```gleam
/// let layer = rms_norm_init(4)
/// let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0]])
/// let assert Ok(_y) = rms_norm_forward(layer, x)
/// ```
pub fn rms_norm_forward(
  layer: RmsNorm,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let input_shape = tensor.shape(input)
  let scale_shape = tensor.shape(layer.scale)
  use num_features <- result.try(last_dim(scale_shape))
  use input_last <- result.try(last_dim(input_shape))
  case input_last == num_features {
    False -> Error(ShapeMismatch(expected: [num_features], got: [input_last]))
    True -> {
      use data <- result.try(tensor.try_to_list(input))
      use scale_data <- result.try(tensor.try_to_list(layer.scale))
      let normalized =
        chunk_by(data, num_features)
        |> list.map(fn(chunk) {
          let mean_sq = list_mean_squares(chunk)
          let rms = safe_sqrt(mean_sq +. layer.eps)
          list.map(list.zip(chunk, scale_data), fn(t) {
            let #(x, s) = t
            x /. rms *. s
          })
        })
        |> list.flatten
      Ok(Tensor(data: normalized, shape: input_shape))
    }
  }
}

// ---------------------------------------------------------------------------
// BatchNorm1d
// ---------------------------------------------------------------------------

/// 1D Batch Normalization layer. Normalizes along the batch axis (axis 0) for
/// inputs of shape `[batch, num_features]`.
///
/// Training mode: uses batch mean/var and updates running stats via EMA.
/// Eval mode: uses running stats and leaves them untouched.
pub type BatchNorm1d {
  BatchNorm1d(
    scale: Tensor,
    bias: Tensor,
    running_mean: Tensor,
    running_var: Tensor,
    momentum: Float,
    eps: Float,
  )
}

/// Create a `BatchNorm1d` with `scale = ones`, `bias = zeros`,
/// `running_mean = zeros`, `running_var = ones`, `momentum = 0.1`, and
/// `eps = 1.0e-5`.
///
/// Formula (training):
///   `mu  = mean(x, axis=0)`
///   `var = var(x, axis=0)`
///   `y   = (x - mu) / sqrt(var + eps) * scale + bias`
///   `running = (1 - momentum) * running + momentum * batch_stat`
///
/// Formula (eval):
///   `y = (x - running_mean) / sqrt(running_var + eps) * scale + bias`
///
/// ## Example
/// ```gleam
/// let layer = batch_norm_1d_init(4)
/// ```
pub fn batch_norm_1d_init(num_features: Int) -> BatchNorm1d {
  BatchNorm1d(
    scale: tensor.ones([num_features]),
    bias: tensor.zeros([num_features]),
    running_mean: tensor.zeros([num_features]),
    running_var: tensor.ones([num_features]),
    momentum: 0.1,
    eps: 1.0e-5,
  )
}

/// Forward pass for `BatchNorm1d`.
///
/// In `training` mode, computes the mean/variance along axis 0 (the batch
/// axis), normalizes, applies `scale`/`bias`, and updates `running_mean` /
/// `running_var` via exponential moving average:
///   `running = (1 - momentum) * running + momentum * batch_stat`.
///
/// In eval mode (`!training`), normalizes using `running_mean` / `running_var`
/// directly and returns the layer unchanged.
///
/// Returns the (possibly updated) layer and the normalized output.
///
/// Errors:
/// - `InvalidShape` if the input is not 2D.
/// - `ShapeMismatch` if the input's feature dimension does not match
///   `num_features`.
///
/// ## Example
/// ```gleam
/// let layer = batch_norm_1d_init(2)
/// let assert Ok(x) = tensor.from_list2d([[1.0, 2.0], [3.0, 4.0]])
/// let assert Ok(#(_updated, _y)) = batch_norm_1d_forward(layer, x, True)
/// ```
pub fn batch_norm_1d_forward(
  layer: BatchNorm1d,
  input: Tensor,
  training: Bool,
) -> Result(#(BatchNorm1d, Tensor), TensorError) {
  let scale_shape = tensor.shape(layer.scale)
  use num_features <- result.try(last_dim(scale_shape))
  case tensor.shape(input) {
    [batch, features] -> {
      case features == num_features {
        False ->
          Error(
            ShapeMismatch(expected: [batch, num_features], got: [
              batch,
              features,
            ]),
          )
        True -> {
          case batch <= 0 {
            True ->
              Error(InvalidShape(
                "batch_norm_1d: batch dimension must be positive",
              ))
            False ->
              batch_norm_1d_apply(layer, input, batch, features, training)
          }
        }
      }
    }
    other ->
      Error(InvalidShape(
        "batch_norm_1d expects 2D input [batch, num_features], got "
        <> shape_to_string(other),
      ))
  }
}

fn batch_norm_1d_apply(
  layer: BatchNorm1d,
  input: Tensor,
  batch: Int,
  features: Int,
  training: Bool,
) -> Result(#(BatchNorm1d, Tensor), TensorError) {
  use data <- result.try(tensor.try_to_list(input))
  use scale_data <- result.try(tensor.try_to_list(layer.scale))
  use bias_data <- result.try(tensor.try_to_list(layer.bias))
  use running_mean_data <- result.try(tensor.try_to_list(layer.running_mean))
  use running_var_data <- result.try(tensor.try_to_list(layer.running_var))

  // Rows along axis 0 — each row is a length-`features` chunk.
  let rows = chunk_by(data, features)

  // Per-feature batch mean/var over the batch axis.
  let batch_mean = column_means(rows, features, batch)
  let batch_var = column_variances(rows, batch_mean, features, batch)

  let #(use_mean, use_var) = case training {
    True -> #(batch_mean, batch_var)
    False -> #(running_mean_data, running_var_data)
  }

  let normalized_rows =
    list.map(rows, fn(row) {
      list.map(
        list.zip(
          row,
          list.zip(use_mean, list.zip(use_var, list.zip(scale_data, bias_data))),
        ),
        fn(t) {
          let #(x, rest) = t
          let #(mu, rest2) = rest
          let #(var, rest3) = rest2
          let #(s, b) = rest3
          let denom = safe_sqrt(var +. layer.eps)
          { x -. mu } /. denom *. s +. b
        },
      )
    })
  let out_data = list.flatten(normalized_rows)
  let out = Tensor(data: out_data, shape: [batch, features])

  let updated_layer = case training {
    False -> layer
    True -> {
      let new_mean = ema_update(running_mean_data, batch_mean, layer.momentum)
      let new_var = ema_update(running_var_data, batch_var, layer.momentum)
      BatchNorm1d(
        scale: layer.scale,
        bias: layer.bias,
        running_mean: Tensor(data: new_mean, shape: [features]),
        running_var: Tensor(data: new_var, shape: [features]),
        momentum: layer.momentum,
        eps: layer.eps,
      )
    }
  }

  Ok(#(updated_layer, out))
}

// ---------------------------------------------------------------------------
// GroupNorm
// ---------------------------------------------------------------------------

/// Group Normalization layer. Splits `num_channels` into `num_groups` equally
/// sized groups and normalizes per group.
///
/// v1 limitation: accepts only inputs of shape `[batch, channels]` or
/// `[batch, channels, spatial]`. Higher-rank inputs return `InvalidShape`.
///
/// Math: for each `(batch, group)` slice, compute the mean and variance over
/// all elements in that group (across channels-in-group and spatial), then
///   `y_c = (x_c - mu_g) / sqrt(var_g + eps) * scale_c + bias_c`,
/// where `g` is the group index containing channel `c`.
pub type GroupNorm {
  GroupNorm(num_groups: Int, scale: Tensor, bias: Tensor, eps: Float)
}

/// Create a `GroupNorm` with `scale = ones([num_channels])` and
/// `bias = zeros([num_channels])`. `num_channels` must be divisible by
/// `num_groups`.
///
/// Formula: `y_c = (x_c - mu_g) / sqrt(var_g + eps) * scale_c + bias_c`,
/// where `g` is the group containing channel `c`.
///
/// ## Example
/// ```gleam
/// let layer = group_norm_init(2, 4)
/// // 2 groups of 2 channels each
/// ```
pub fn group_norm_init(num_groups: Int, num_channels: Int) -> GroupNorm {
  GroupNorm(
    num_groups: num_groups,
    scale: tensor.ones([num_channels]),
    bias: tensor.zeros([num_channels]),
    eps: 1.0e-5,
  )
}

/// Forward pass for `GroupNorm`.
///
/// Accepts inputs of shape `[batch, channels]` or `[batch, channels, spatial]`.
/// Splits `channels` into `num_groups` groups, computes per-`(batch, group)`
/// mean and variance, then normalizes and applies the per-channel
/// `scale` and `bias`.
///
/// Errors:
/// - `InvalidShape("group_norm: channels (<C>) not divisible by groups (<G>)")`
///   when `channels` is not a multiple of `num_groups`.
/// - `InvalidShape` if the input is not 2D or 3D, or if the channel dimension
///   does not match `scale`.
///
/// ## Example
/// ```gleam
/// let layer = group_norm_init(2, 4)
/// let assert Ok(x) = tensor.from_list2d([[1.0, 2.0, 3.0, 4.0]])
/// let assert Ok(_y) = group_norm_forward(layer, x)
/// ```
pub fn group_norm_forward(
  layer: GroupNorm,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  let scale_shape = tensor.shape(layer.scale)
  use num_channels <- result.try(last_dim(scale_shape))
  case num_channels % layer.num_groups == 0 {
    False ->
      Error(InvalidShape(
        "group_norm: channels ("
        <> int.to_string(num_channels)
        <> ") not divisible by groups ("
        <> int.to_string(layer.num_groups)
        <> ")",
      ))
    True -> {
      case tensor.shape(input) {
        [batch, channels] ->
          group_norm_apply(layer, input, batch, channels, 1, num_channels)
        [batch, channels, spatial] ->
          group_norm_apply(layer, input, batch, channels, spatial, num_channels)
        other ->
          Error(InvalidShape(
            "group_norm expects [batch, channels] or [batch, channels, spatial], got "
            <> shape_to_string(other),
          ))
      }
    }
  }
}

fn group_norm_apply(
  layer: GroupNorm,
  input: Tensor,
  batch: Int,
  channels: Int,
  spatial: Int,
  num_channels: Int,
) -> Result(Tensor, TensorError) {
  case channels == num_channels {
    False ->
      Error(
        ShapeMismatch(expected: [batch, num_channels], got: [
          batch,
          channels,
        ]),
      )
    True -> {
      use data <- result.try(tensor.try_to_list(input))
      use scale_data <- result.try(tensor.try_to_list(layer.scale))
      use bias_data <- result.try(tensor.try_to_list(layer.bias))

      let channels_per_group = channels / layer.num_groups
      let group_size = channels_per_group * spatial
      let row_size = channels * spatial
      let rows = chunk_by(data, row_size)

      let normalized_rows =
        list.map(rows, fn(row) {
          // Split a [channels, spatial]-flat row into one chunk per group.
          let groups = chunk_by(row, group_size)
          let normalized_groups =
            list.index_map(groups, fn(group_data, group_idx) {
              let mean = list_mean(group_data)
              let var = list_variance(group_data, mean)
              let denom = safe_sqrt(var +. layer.eps)
              // Within this group, walk channels-in-group then spatial.
              // Each consecutive `spatial` elements belong to one channel.
              let channels_in_group = chunk_by(group_data, spatial)
              list.index_map(channels_in_group, fn(channel_data, c_in_group) {
                let global_channel = group_idx * channels_per_group + c_in_group
                let s = list_at(scale_data, global_channel)
                let b = list_at(bias_data, global_channel)
                list.map(channel_data, fn(x) {
                  { x -. mean } /. denom *. s +. b
                })
              })
              |> list.flatten
            })
          list.flatten(normalized_groups)
        })
      Ok(Tensor(data: list.flatten(normalized_rows), shape: tensor.shape(input)))
    }
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn last_dim(shape: List(Int)) -> Result(Int, TensorError) {
  case list.last(shape) {
    Ok(d) -> Ok(d)
    Error(_) -> Error(InvalidShape("expected non-empty shape, got []"))
  }
}

fn chunk_by(data: List(Float), n: Int) -> List(List(Float)) {
  case n <= 0 {
    True -> []
    False -> chunk_by_loop(data, n, [])
  }
}

fn chunk_by_loop(
  data: List(Float),
  n: Int,
  acc: List(List(Float)),
) -> List(List(Float)) {
  case data {
    [] -> list.reverse(acc)
    _ -> {
      let chunk = list.take(data, n)
      let rest = list.drop(data, n)
      chunk_by_loop(rest, n, [chunk, ..acc])
    }
  }
}

fn list_mean(xs: List(Float)) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ -> list.fold(xs, 0.0, fn(acc, x) { acc +. x }) /. int.to_float(n)
  }
}

fn list_variance(xs: List(Float), mean: Float) -> Float {
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

fn list_mean_squares(xs: List(Float)) -> Float {
  let n = list.length(xs)
  case n {
    0 -> 0.0
    _ -> list.fold(xs, 0.0, fn(acc, x) { acc +. x *. x }) /. int.to_float(n)
  }
}

fn safe_sqrt(x: Float) -> Float {
  case float.square_root(x) {
    Ok(v) -> v
    Error(_) -> 0.0
  }
}

fn column_means(
  rows: List(List(Float)),
  features: Int,
  batch: Int,
) -> List(Float) {
  let zeros = list.repeat(0.0, features)
  let sums =
    list.fold(rows, zeros, fn(acc, row) {
      list.map(list.zip(acc, row), fn(p) {
        let #(a, b) = p
        a +. b
      })
    })
  let n = int.to_float(batch)
  list.map(sums, fn(s) { s /. n })
}

fn column_variances(
  rows: List(List(Float)),
  means: List(Float),
  features: Int,
  batch: Int,
) -> List(Float) {
  let zeros = list.repeat(0.0, features)
  let sumsq =
    list.fold(rows, zeros, fn(acc, row) {
      list.map(list.zip(acc, list.zip(row, means)), fn(p) {
        let #(a, rest) = p
        let #(x, mu) = rest
        let d = x -. mu
        a +. d *. d
      })
    })
  let n = int.to_float(batch)
  list.map(sumsq, fn(s) { s /. n })
}

fn ema_update(
  running: List(Float),
  batch: List(Float),
  momentum: Float,
) -> List(Float) {
  let one_minus = 1.0 -. momentum
  list.map(list.zip(running, batch), fn(p) {
    let #(r, b) = p
    one_minus *. r +. momentum *. b
  })
}

fn list_at(xs: List(Float), idx: Int) -> Float {
  case xs, idx {
    [], _ -> 0.0
    [x, ..], 0 -> x
    [_, ..rest], i -> list_at(rest, i - 1)
  }
}

fn shape_to_string(shape: List(Int)) -> String {
  "[" <> join_with(list.map(shape, int.to_string), ", ") <> "]"
}

fn join_with(parts: List(String), sep: String) -> String {
  case parts {
    [] -> ""
    [s] -> s
    [s, ..rest] -> s <> sep <> join_with(rest, sep)
  }
}
