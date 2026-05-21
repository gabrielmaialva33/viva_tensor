//// HuggingFace SafeTensors -> viva_tensor weight loader.
////
//// Reads a `.safetensors` file produced by the `transformers` library and
//// maps its flat `name -> tensor` dictionary into viva_tensor's strongly
//// typed `Embedding`, `LayerNorm`, `MultiHeadAttention`, `FeedForward`,
//// `EncoderBlock`, `DecoderBlock`, and `Transformer` records.
////
//// ### Naming convention
////
//// This loader targets **BERT/BART-style flat hierarchies** with one tensor
//// per projection. For an encoder block at index `L`:
////
//// ```
//// encoder.layers.{L}.self_attn.q_proj.weight   [embed_dim, embed_dim]
//// encoder.layers.{L}.self_attn.q_proj.bias     [embed_dim]
//// encoder.layers.{L}.self_attn.k_proj.weight   [embed_dim, embed_dim]
//// encoder.layers.{L}.self_attn.k_proj.bias     [embed_dim]
//// encoder.layers.{L}.self_attn.v_proj.weight   [embed_dim, embed_dim]
//// encoder.layers.{L}.self_attn.v_proj.bias     [embed_dim]
//// encoder.layers.{L}.self_attn.out_proj.weight [embed_dim, embed_dim]
//// encoder.layers.{L}.self_attn.out_proj.bias   [embed_dim]
//// encoder.layers.{L}.norm1.weight              [embed_dim]
//// encoder.layers.{L}.norm1.bias                [embed_dim]
//// encoder.layers.{L}.norm2.weight              [embed_dim]
//// encoder.layers.{L}.norm2.bias                [embed_dim]
//// encoder.layers.{L}.ffn.linear1.weight        [embed_dim, hidden_dim]
//// encoder.layers.{L}.ffn.linear1.bias          [hidden_dim]
//// encoder.layers.{L}.ffn.linear2.weight        [hidden_dim, embed_dim]
//// encoder.layers.{L}.ffn.linear2.bias          [embed_dim]
//// ```
////
//// Decoder blocks mirror the encoder, plus a cross-attention sublayer:
////
//// ```
//// decoder.layers.{L}.self_attn.*    (same 8 names as encoder.self_attn.*)
//// decoder.layers.{L}.cross_attn.*   (same 8 names; cross-attention)
//// decoder.layers.{L}.norm1.weight / bias       [embed_dim]
//// decoder.layers.{L}.norm2.weight / bias       [embed_dim]
//// decoder.layers.{L}.norm3.weight / bias       [embed_dim]
//// decoder.layers.{L}.ffn.linear1.weight / bias
//// decoder.layers.{L}.ffn.linear2.weight / bias
//// ```
////
//// Top-level (when `config.has_embedding` is `True`):
////
//// ```
//// embedding.weight    [vocab_size, embed_dim]
//// positional.weight   [max_len, embed_dim]   (loaded by caller, not here)
//// ```
////
//// ### Conventions NOT supported
////
//// - **GPT-2-style fused QKV** (`attn.c_attn.weight` of shape
////   `[embed_dim, 3*embed_dim]`). Callers must split manually.
//// - **T5-style relative attention bias** (`relative_attention_bias`).
//// - **Conv1D-transposed weights**: PyTorch `nn.Linear` stores weights as
////   `[out, in]`; safetensors files dumped via `transformers` often keep that
////   layout. We assume `[in, out]` here to match viva_tensor's matmul
////   convention (`y = x @ W`). If your file uses `[out, in]`, transpose first
////   via `tensor.transpose` before calling these helpers.
////
//// Pure Gleam, no NIF. Errors are localized via the `HfLoadError` type so
//// callers can tell missing-weight from shape-mismatch from I/O failure.

import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/option.{Some}
import gleam/result
import gleam/string
import viva_tensor/core/error.{type TensorError} as core_error
import viva_tensor/io/safetensors
import viva_tensor/nn/attention.{type MultiHeadAttention, MultiHeadAttention}
import viva_tensor/nn/embedding.{type Embedding, Embedding}
import viva_tensor/nn/norm.{type LayerNorm, LayerNorm}
import viva_tensor/nn/transformer.{
  type Activation, type DecoderBlock, type EncoderBlock, type FeedForward,
  type Transformer, DecoderBlock, EncoderBlock, FeedForward, Transformer,
}
import viva_tensor/tensor.{type Tensor}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors raised by the HuggingFace SafeTensors loader.
///
/// - `WeightNotFound(name)` — expected key missing from the safetensors dict.
/// - `ShapeMismatch(name, expected, got)` — key found, but its shape does
///   not match the layer config (e.g. `embed_dim` mismatch).
/// - `IoError(reason)` — underlying `safetensors.read` failure, wrapped
///   into a string for easier propagation.
pub type HfLoadError {
  WeightNotFound(name: String)
  ShapeMismatch(name: String, expected: List(Int), got: List(Int))
  IoError(reason: String)
}

// ---------------------------------------------------------------------------
// TransformerConfig
// ---------------------------------------------------------------------------

/// Static dimensions and structural toggles needed to reconstruct a
/// `Transformer` from a flat weight dictionary.
///
/// - `num_encoder_layers` / `num_decoder_layers` — depth of each stack.
/// - `embed_dim` — model width (must equal `num_heads * head_dim`).
/// - `num_heads` — head count for MHA; `embed_dim % num_heads` must be `0`.
/// - `hidden_dim` — FFN inner width (`linear1` output / `linear2` input).
/// - `activation` — pointwise activation between `linear1` and `linear2`.
/// - `has_embedding` — when `True`, also load `embedding.weight`.
/// - `vocab_size` — only consulted when `has_embedding` is `True`.
pub type TransformerConfig {
  TransformerConfig(
    num_encoder_layers: Int,
    num_decoder_layers: Int,
    embed_dim: Int,
    num_heads: Int,
    hidden_dim: Int,
    activation: Activation,
    has_embedding: Bool,
    vocab_size: Int,
  )
}

// ---------------------------------------------------------------------------
// Top-level: safetensors -> weight dict
// ---------------------------------------------------------------------------

/// Thin wrapper around `safetensors.read/1` that converts the underlying
/// `TensorError` into our loader-local `IoError(reason)` so callers can
/// match on a single error type.
///
/// Expected file shape: any well-formed SafeTensors file with F32 or F64
/// payloads. See `viva_tensor/io/safetensors` for the wire format.
pub fn load_safetensors_dict(
  path: String,
) -> Result(Dict(String, Tensor), HfLoadError) {
  case safetensors.read(path) {
    Ok(d) -> Ok(d)
    Error(err) -> Error(IoError(tensor_error_to_string(err)))
  }
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

/// Load an `Embedding` from `prefix <> ".weight"`.
///
/// Expected HF weight names and shapes (for `prefix = "embedding"`):
///
/// ```
/// embedding.weight   [vocab_size, embedding_dim]
/// ```
///
/// Errors:
/// - `WeightNotFound(prefix <> ".weight")` if the key is missing.
/// - `ShapeMismatch(name, [vocab_size, embedding_dim], got)` on any
///   mismatch with the supplied dims.
pub fn load_embedding(
  weights: Dict(String, Tensor),
  prefix: String,
  vocab_size: Int,
  embedding_dim: Int,
) -> Result(Embedding, HfLoadError) {
  let name = prefix <> ".weight"
  use w <- result.try(get_weight(weights, name))
  use _ <- result.try(check_shape(name, w, [vocab_size, embedding_dim]))
  Ok(Embedding(
    num_embeddings: vocab_size,
    embedding_dim: embedding_dim,
    weight: w,
  ))
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

/// Load a `LayerNorm` from `prefix <> ".weight"` (scale) and
/// `prefix <> ".bias"`.
///
/// Expected HF weight names and shapes (for `prefix = "encoder.layers.0.norm1"`):
///
/// ```
/// encoder.layers.0.norm1.weight   [num_features]
/// encoder.layers.0.norm1.bias     [num_features]
/// ```
///
/// `eps` defaults to `1.0e-5`, matching `layer_norm_init` and PyTorch's
/// `nn.LayerNorm` default.
///
/// Errors:
/// - `WeightNotFound` when either key is missing.
/// - `ShapeMismatch(name, [num_features], got)` if either tensor does not
///   match `[num_features]`.
pub fn load_layer_norm(
  weights: Dict(String, Tensor),
  prefix: String,
  num_features: Int,
) -> Result(LayerNorm, HfLoadError) {
  let scale_name = prefix <> ".weight"
  let bias_name = prefix <> ".bias"
  use scale <- result.try(get_weight(weights, scale_name))
  use _ <- result.try(check_shape(scale_name, scale, [num_features]))
  use bias <- result.try(get_weight(weights, bias_name))
  use _ <- result.try(check_shape(bias_name, bias, [num_features]))
  Ok(LayerNorm(scale: scale, bias: bias, eps: 1.0e-5))
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

/// Load an MHA module from a four-projection HF layout (`q_proj`, `k_proj`,
/// `v_proj`, `out_proj`) with biases.
///
/// Expected HF weight names and shapes (for `prefix = "encoder.layers.0.self_attn"`):
///
/// ```
/// encoder.layers.0.self_attn.q_proj.weight     [embed_dim, embed_dim]
/// encoder.layers.0.self_attn.q_proj.bias       [embed_dim]
/// encoder.layers.0.self_attn.k_proj.weight     [embed_dim, embed_dim]
/// encoder.layers.0.self_attn.k_proj.bias       [embed_dim]
/// encoder.layers.0.self_attn.v_proj.weight     [embed_dim, embed_dim]
/// encoder.layers.0.self_attn.v_proj.bias       [embed_dim]
/// encoder.layers.0.self_attn.out_proj.weight   [embed_dim, embed_dim]
/// encoder.layers.0.self_attn.out_proj.bias     [embed_dim]
/// ```
///
/// Errors:
/// - `WeightNotFound` when any of the 8 keys is missing.
/// - `ShapeMismatch` if any weight is not `[embed_dim, embed_dim]` or any
///   bias is not `[embed_dim]`.
/// - `IoError("multi_head_attention: embed_dim not divisible by num_heads")`
///   when `embed_dim % num_heads != 0`.
pub fn load_multi_head_attention(
  weights: Dict(String, Tensor),
  prefix: String,
  num_heads: Int,
  embed_dim: Int,
) -> Result(MultiHeadAttention, HfLoadError) {
  case num_heads <= 0 || embed_dim <= 0 {
    True ->
      Error(IoError(
        "multi_head_attention: num_heads and embed_dim must be positive",
      ))
    False ->
      case embed_dim % num_heads == 0 {
        False ->
          Error(IoError(
            "multi_head_attention: embed_dim ("
            <> int.to_string(embed_dim)
            <> ") not divisible by num_heads ("
            <> int.to_string(num_heads)
            <> ")",
          ))
        True -> {
          let head_dim = embed_dim / num_heads
          let weight_shape = [embed_dim, embed_dim]
          let bias_shape = [embed_dim]

          use w_q <- result.try(get_and_check(
            weights,
            prefix <> ".q_proj.weight",
            weight_shape,
          ))
          use b_q <- result.try(get_and_check(
            weights,
            prefix <> ".q_proj.bias",
            bias_shape,
          ))
          use w_k <- result.try(get_and_check(
            weights,
            prefix <> ".k_proj.weight",
            weight_shape,
          ))
          use b_k <- result.try(get_and_check(
            weights,
            prefix <> ".k_proj.bias",
            bias_shape,
          ))
          use w_v <- result.try(get_and_check(
            weights,
            prefix <> ".v_proj.weight",
            weight_shape,
          ))
          use b_v <- result.try(get_and_check(
            weights,
            prefix <> ".v_proj.bias",
            bias_shape,
          ))
          use w_o <- result.try(get_and_check(
            weights,
            prefix <> ".out_proj.weight",
            weight_shape,
          ))
          use b_o <- result.try(get_and_check(
            weights,
            prefix <> ".out_proj.bias",
            bias_shape,
          ))

          Ok(MultiHeadAttention(
            num_heads: num_heads,
            embed_dim: embed_dim,
            head_dim: head_dim,
            w_q: w_q,
            w_k: w_k,
            w_v: w_v,
            w_o: w_o,
            b_q: Some(b_q),
            b_k: Some(b_k),
            b_v: Some(b_v),
            b_o: Some(b_o),
          ))
        }
      }
  }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// Load a position-wise FFN from two linear projections.
///
/// Expected HF weight names and shapes (for `prefix = "encoder.layers.0.ffn"`):
///
/// ```
/// encoder.layers.0.ffn.linear1.weight   [embed_dim, hidden_dim]
/// encoder.layers.0.ffn.linear1.bias     [hidden_dim]
/// encoder.layers.0.ffn.linear2.weight   [hidden_dim, embed_dim]
/// encoder.layers.0.ffn.linear2.bias     [embed_dim]
/// ```
///
/// Note: viva_tensor's matmul uses `y = x @ W`, so weights are stored
/// `[in, out]`. If your safetensors file came from PyTorch's `nn.Linear`
/// (which stores `[out, in]`), transpose before calling.
///
/// Errors:
/// - `WeightNotFound` if any of the 4 keys is missing.
/// - `ShapeMismatch` if a weight's shape does not match.
pub fn load_feed_forward(
  weights: Dict(String, Tensor),
  prefix: String,
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(FeedForward, HfLoadError) {
  use w1 <- result.try(
    get_and_check(weights, prefix <> ".linear1.weight", [embed_dim, hidden_dim]),
  )
  use b1 <- result.try(
    get_and_check(weights, prefix <> ".linear1.bias", [hidden_dim]),
  )
  use w2 <- result.try(
    get_and_check(weights, prefix <> ".linear2.weight", [hidden_dim, embed_dim]),
  )
  use b2 <- result.try(
    get_and_check(weights, prefix <> ".linear2.bias", [embed_dim]),
  )
  Ok(FeedForward(w1: w1, b1: b1, w2: w2, b2: b2, activation: activation))
}

// ---------------------------------------------------------------------------
// EncoderBlock
// ---------------------------------------------------------------------------

/// Compose `load_multi_head_attention`, `load_feed_forward`, and two
/// `load_layer_norm` calls under a shared block prefix.
///
/// Expected HF sub-prefixes (for `prefix = "encoder.layers.0"`):
///
/// ```
/// encoder.layers.0.self_attn.*   (see load_multi_head_attention)
/// encoder.layers.0.norm1.*       (LayerNorm, [embed_dim])
/// encoder.layers.0.norm2.*       (LayerNorm, [embed_dim])
/// encoder.layers.0.ffn.*         (FeedForward, see load_feed_forward)
/// ```
///
/// Errors: bubble up any `WeightNotFound` or `ShapeMismatch` from the
/// sublayer loaders.
pub fn load_encoder_block(
  weights: Dict(String, Tensor),
  prefix: String,
  num_heads: Int,
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(EncoderBlock, HfLoadError) {
  use mha <- result.try(load_multi_head_attention(
    weights,
    prefix <> ".self_attn",
    num_heads,
    embed_dim,
  ))
  use norm1 <- result.try(load_layer_norm(
    weights,
    prefix <> ".norm1",
    embed_dim,
  ))
  use norm2 <- result.try(load_layer_norm(
    weights,
    prefix <> ".norm2",
    embed_dim,
  ))
  use ffn <- result.try(load_feed_forward(
    weights,
    prefix <> ".ffn",
    embed_dim,
    hidden_dim,
    activation,
  ))
  Ok(EncoderBlock(attention: mha, ffn: ffn, norm1: norm1, norm2: norm2))
}

// ---------------------------------------------------------------------------
// DecoderBlock
// ---------------------------------------------------------------------------

/// Load a single decoder block (causal self-attention + cross-attention
/// + FFN, with three `LayerNorm`s).
///
/// Expected HF sub-prefixes (for `prefix = "decoder.layers.0"`):
///
/// ```
/// decoder.layers.0.self_attn.*    (see load_multi_head_attention)
/// decoder.layers.0.cross_attn.*   (see load_multi_head_attention)
/// decoder.layers.0.norm1.*        (LayerNorm, [embed_dim])
/// decoder.layers.0.norm2.*        (LayerNorm, [embed_dim])
/// decoder.layers.0.norm3.*        (LayerNorm, [embed_dim])
/// decoder.layers.0.ffn.*          (FeedForward, see load_feed_forward)
/// ```
fn load_decoder_block(
  weights: Dict(String, Tensor),
  prefix: String,
  num_heads: Int,
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(DecoderBlock, HfLoadError) {
  use self_mha <- result.try(load_multi_head_attention(
    weights,
    prefix <> ".self_attn",
    num_heads,
    embed_dim,
  ))
  use cross_mha <- result.try(load_multi_head_attention(
    weights,
    prefix <> ".cross_attn",
    num_heads,
    embed_dim,
  ))
  use norm1 <- result.try(load_layer_norm(
    weights,
    prefix <> ".norm1",
    embed_dim,
  ))
  use norm2 <- result.try(load_layer_norm(
    weights,
    prefix <> ".norm2",
    embed_dim,
  ))
  use norm3 <- result.try(load_layer_norm(
    weights,
    prefix <> ".norm3",
    embed_dim,
  ))
  use ffn <- result.try(load_feed_forward(
    weights,
    prefix <> ".ffn",
    embed_dim,
    hidden_dim,
    activation,
  ))
  Ok(DecoderBlock(
    self_attention: self_mha,
    cross_attention: cross_mha,
    ffn: ffn,
    norm1: norm1,
    norm2: norm2,
    norm3: norm3,
  ))
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

/// Load a full `Transformer` (encoder stack + decoder stack) from a flat
/// weight dictionary.
///
/// Expected HF prefixes:
///
/// ```
/// encoder.layers.{0..num_enc_layers-1}.*   (see load_encoder_block)
/// decoder.layers.{0..num_dec_layers-1}.*   (see load_decoder_block)
/// ```
///
/// Errors bubble up `WeightNotFound` / `ShapeMismatch` from any layer.
pub fn load_transformer(
  weights: Dict(String, Tensor),
  num_enc_layers: Int,
  num_dec_layers: Int,
  embed_dim: Int,
  num_heads: Int,
  hidden_dim: Int,
  activation: Activation,
) -> Result(Transformer, HfLoadError) {
  case num_enc_layers < 0 || num_dec_layers < 0 {
    True ->
      Error(IoError(
        "load_transformer: layer counts must be non-negative (got num_enc_layers="
        <> int.to_string(num_enc_layers)
        <> ", num_dec_layers="
        <> int.to_string(num_dec_layers)
        <> ")",
      ))
    False -> {
      let enc_indices = case num_enc_layers <= 0 {
        True -> []
        False -> range_int(0, num_enc_layers - 1)
      }
      let dec_indices = case num_dec_layers <= 0 {
        True -> []
        False -> range_int(0, num_dec_layers - 1)
      }
      use encoders <- result.try(
        list.try_map(enc_indices, fn(i) {
          load_encoder_block(
            weights,
            "encoder.layers." <> int.to_string(i),
            num_heads,
            embed_dim,
            hidden_dim,
            activation,
          )
        }),
      )
      use decoders <- result.try(
        list.try_map(dec_indices, fn(i) {
          load_decoder_block(
            weights,
            "decoder.layers." <> int.to_string(i),
            num_heads,
            embed_dim,
            hidden_dim,
            activation,
          )
        }),
      )
      Ok(Transformer(
        encoder_blocks: encoders,
        decoder_blocks: decoders,
        num_encoder_layers: num_enc_layers,
        num_decoder_layers: num_dec_layers,
      ))
    }
  }
}

// ---------------------------------------------------------------------------
// Convenience: file -> Transformer
// ---------------------------------------------------------------------------

/// Read a SafeTensors file and project the resulting weight dict into a
/// `Transformer` using `config`'s dimensions.
///
/// Equivalent to:
///
/// ```gleam
/// use weights <- result.try(load_safetensors_dict(path))
/// load_transformer(weights, ..., config.activation)
/// ```
///
/// Note: this convenience returns only the `Transformer`. To load the
/// optional `embedding.weight` table, call `load_embedding/4` separately on
/// the same weight dict — toggle via `config.has_embedding` and look up
/// `config.vocab_size`.
pub fn from_safetensors_file(
  path: String,
  config: TransformerConfig,
) -> Result(Transformer, HfLoadError) {
  use weights <- result.try(load_safetensors_dict(path))
  load_transformer(
    weights,
    config.num_encoder_layers,
    config.num_decoder_layers,
    config.embed_dim,
    config.num_heads,
    config.hidden_dim,
    config.activation,
  )
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn get_weight(
  weights: Dict(String, Tensor),
  name: String,
) -> Result(Tensor, HfLoadError) {
  case dict.get(weights, name) {
    Ok(t) -> Ok(t)
    Error(_) -> Error(WeightNotFound(name))
  }
}

fn check_shape(
  name: String,
  t: Tensor,
  expected: List(Int),
) -> Result(Nil, HfLoadError) {
  let got = tensor.shape(t)
  case got == expected {
    True -> Ok(Nil)
    False -> Error(ShapeMismatch(name: name, expected: expected, got: got))
  }
}

fn get_and_check(
  weights: Dict(String, Tensor),
  name: String,
  expected: List(Int),
) -> Result(Tensor, HfLoadError) {
  use t <- result.try(get_weight(weights, name))
  use _ <- result.try(check_shape(name, t, expected))
  Ok(t)
}

fn tensor_error_to_string(err: TensorError) -> String {
  case err {
    core_error.InvalidShape(reason) -> reason
    core_error.DtypeError(reason) -> reason
    core_error.ShapeMismatch(expected, got) ->
      "shape mismatch: expected "
      <> shape_to_string(expected)
      <> ", got "
      <> shape_to_string(got)
    core_error.DimensionError(reason) -> reason
    core_error.IndexOutOfBounds(idx, size) ->
      "index "
      <> int.to_string(idx)
      <> " out of bounds for size "
      <> int.to_string(size)
    other -> string.inspect(other)
  }
}

fn shape_to_string(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), ", ") <> "]"
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
