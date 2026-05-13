//// Transformer encoder/decoder building blocks (pre-norm style).
////
//// Pure-Gleam forward passes. No NIF, no autograd integration in this round.
//// Each block is a plain record carrying its sublayers; constructors return
//// zero-weight defaults so callers/tests can swap in real weights via
//// record-update syntax.
////
//// References:
//// - Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
////   https://arxiv.org/abs/1706.03762
//// - Xiong et al. (2020). "On Layer Normalization in the Transformer
////   Architecture." (Pre-LN vs Post-LN analysis.)
////   https://arxiv.org/abs/2002.04745
////
//// Style choice: we use **pre-norm** (LayerNorm before each sublayer, with
//// the residual stream untouched by norm). Pre-norm trains more stably
//// without warmup and matches GPT-style modern stacks.
////
//// Residual diagrams:
////
//// Encoder block (pre-norm):
////   x        ──────────────────────────────┐
////    │                                     │
////   norm1 ─▶ MHA(self, is_causal) ─▶ + ◀──┘  = r1
////   r1       ──────────────────────────────┐
////    │                                     │
////   norm2 ─▶ FFN ──────────────────▶ + ◀──┘   = output
////
//// Decoder block (pre-norm + causal self-attn + cross-attn):
////   x        ──────────────────────────────┐
////    │                                     │
////   norm1 ─▶ MHA(self, causal)    ─▶ + ◀──┘   = r1
////   r1       ──────────────────────────────┐
////    │                                     │
////   norm2 ─▶ MHA(cross, q=r1, kv=mem) ▶ + ◀┘   = r2
////   r2       ──────────────────────────────┐
////    │                                     │
////   norm3 ─▶ FFN ──────────────────▶ + ◀──┘   = output

import gleam/int
import gleam/list
import gleam/option.{None}
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape, ShapeMismatch}
import viva_tensor/nn/activations
import viva_tensor/nn/attention.{
  type MultiHeadAttention, multi_head_attention_forward,
  multi_head_attention_init, scaled_dot_product_attention,
}
import viva_tensor/nn/norm.{type LayerNorm, layer_norm_forward, layer_norm_init}
import viva_tensor/tensor.{type Tensor, Tensor}

// ---------------------------------------------------------------------------
// Activation enum
// ---------------------------------------------------------------------------

/// Pointwise activation used by the FFN sublayer.
///
/// `ReluAct` — `max(0, x)`. Cheap and the original choice in Vaswani et al.
/// `GeluAct` — `0.5 * x * (1 + erf(x / sqrt(2)))`. Smooth, default in
/// modern transformer stacks (BERT, GPT).
pub type Activation {
  ReluAct
  GeluAct
}

fn apply_activation(t: Tensor, act: Activation) -> Tensor {
  case act {
    ReluAct -> activations.relu(t)
    GeluAct -> activations.gelu(t)
  }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

/// Position-wise feed-forward sublayer.
///
/// Two affine projections with a pointwise activation in between:
/// `y = activation(x @ w1 + b1) @ w2 + b2`.
///
/// Shapes:
/// - `w1`: `[embed_dim, hidden_dim]`
/// - `b1`: `[hidden_dim]`
/// - `w2`: `[hidden_dim, embed_dim]`
/// - `b2`: `[embed_dim]`
pub type FeedForward {
  FeedForward(
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    activation: Activation,
  )
}

/// Build a `FeedForward` with zero-filled weight matrices and bias vectors.
///
/// Toy default: real training code should swap in Xavier/Kaiming weights via
/// record-update syntax. Tests typically inject `tensor.identity` or hand-
/// picked weights to verify shapes and routing.
pub fn feed_forward_init(
  embed_dim: Int,
  hidden_dim: Int,
  activation: Activation,
) -> FeedForward {
  FeedForward(
    w1: tensor.zeros([embed_dim, hidden_dim]),
    b1: tensor.zeros([hidden_dim]),
    w2: tensor.zeros([hidden_dim, embed_dim]),
    b2: tensor.zeros([embed_dim]),
    activation: activation,
  )
}

/// FeedForward forward pass:
///
/// ```
/// hidden = activation(input @ w1 + b1)
/// output = hidden @ w2 + b2
/// ```
///
/// Input is expected to be rank-2 `[seq_len, embed_dim]`. The bias vectors
/// are broadcast over the `seq_len` axis. Returns `ShapeMismatch` when the
/// matmul/bias shapes do not line up.
pub fn feed_forward_forward(
  ff: FeedForward,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  use h1 <- result.try(tensor.matmul(input, ff.w1))
  use h1_b <- result.try(add_bias_row(h1, ff.b1))
  let activated = apply_activation(h1_b, ff.activation)
  use h2 <- result.try(tensor.matmul(activated, ff.w2))
  add_bias_row(h2, ff.b2)
}

// Add a [out_features] bias to every row of a [seq, out_features] tensor.
fn add_bias_row(t: Tensor, bias: Tensor) -> Result(Tensor, TensorError) {
  case t.shape, bias.shape {
    [_seq, out], [bo] if bo == out -> {
      use t_data <- result.try(tensor.try_to_list(t))
      use b_data <- result.try(tensor.try_to_list(bias))
      let new_data =
        list.sized_chunk(t_data, out)
        |> list.flat_map(fn(row) { list.map2(row, b_data, fn(x, b) { x +. b }) })
      Ok(Tensor(data: new_data, shape: t.shape))
    }
    _, _ -> Error(ShapeMismatch(expected: t.shape, got: bias.shape))
  }
}

// ---------------------------------------------------------------------------
// EncoderBlock
// ---------------------------------------------------------------------------

/// Single Transformer encoder block (pre-norm).
///
/// Residual diagram:
/// ```
/// x1       = layer_norm1(input)
/// attn_out = multi_head_attention(x1, x1, x1, is_causal)
/// r1       = input + attn_out
/// x2       = layer_norm2(r1)
/// ffn_out  = feed_forward(x2)
/// output   = r1 + ffn_out
/// ```
pub type EncoderBlock {
  EncoderBlock(
    attention: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
  )
}

/// Build an `EncoderBlock` with zero-weight MHA and FFN sublayers and default
/// `LayerNorm`s (`scale=ones`, `bias=zeros`, `eps=1e-5`).
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads` (raised
///   by the underlying `multi_head_attention_init`).
pub fn encoder_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  activation: Activation,
) -> Result(EncoderBlock, TensorError) {
  use mha <- result.try(multi_head_attention_init(num_heads, embed_dim, False))
  let ffn = feed_forward_init(embed_dim, ffn_hidden_dim, activation)
  Ok(EncoderBlock(
    attention: mha,
    ffn: ffn,
    norm1: layer_norm_init(embed_dim),
    norm2: layer_norm_init(embed_dim),
  ))
}

/// Encoder block forward pass (pre-norm + residual).
///
/// ```
/// x1       = layer_norm1(input)
/// attn_out = multi_head_attention(x1, x1, x1, is_causal)
/// r1       = input + attn_out
/// x2       = layer_norm2(r1)
/// ffn_out  = feed_forward(x2)
/// output   = r1 + ffn_out
/// ```
///
/// Input shape: `[seq_len, embed_dim]`. Output shape: same.
pub fn encoder_block_forward(
  block: EncoderBlock,
  input: Tensor,
  is_causal: Bool,
) -> Result(Tensor, TensorError) {
  use x1 <- result.try(layer_norm_forward(block.norm1, input))
  use attn_out <- result.try(multi_head_attention_forward(
    block.attention,
    x1,
    x1,
    x1,
    is_causal,
  ))
  use r1 <- result.try(tensor.add(input, attn_out))
  use x2 <- result.try(layer_norm_forward(block.norm2, r1))
  use ffn_out <- result.try(feed_forward_forward(block.ffn, x2))
  tensor.add(r1, ffn_out)
}

// ---------------------------------------------------------------------------
// DecoderBlock
// ---------------------------------------------------------------------------

/// Single Transformer decoder block (pre-norm) with causal self-attention
/// followed by cross-attention over the encoder memory.
///
/// Residual diagram:
/// ```
/// x1       = layer_norm1(input)
/// self_out = multi_head_attention(x1, x1, x1, is_causal=True)
/// r1       = input + self_out
/// x2       = layer_norm2(r1)
/// cross    = multi_head_attention(x2, memory, memory, is_causal=False)
/// r2       = r1 + cross
/// x3       = layer_norm3(r2)
/// ffn_out  = feed_forward(x3)
/// output   = r2 + ffn_out
/// ```
pub type DecoderBlock {
  DecoderBlock(
    self_attention: MultiHeadAttention,
    cross_attention: MultiHeadAttention,
    ffn: FeedForward,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
  )
}

/// Build a `DecoderBlock` with zero-weight MHA (self + cross) and FFN
/// sublayers, and default `LayerNorm`s.
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads`.
pub fn decoder_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  activation: Activation,
) -> Result(DecoderBlock, TensorError) {
  use self_mha <- result.try(multi_head_attention_init(
    num_heads,
    embed_dim,
    False,
  ))
  use cross_mha <- result.try(multi_head_attention_init(
    num_heads,
    embed_dim,
    False,
  ))
  let ffn = feed_forward_init(embed_dim, ffn_hidden_dim, activation)
  Ok(DecoderBlock(
    self_attention: self_mha,
    cross_attention: cross_mha,
    ffn: ffn,
    norm1: layer_norm_init(embed_dim),
    norm2: layer_norm_init(embed_dim),
    norm3: layer_norm_init(embed_dim),
  ))
}

/// Decoder block forward pass (pre-norm + causal self-attn + cross-attn).
///
/// ```
/// x1       = layer_norm1(input)
/// self_out = multi_head_attention(x1, x1, x1, is_causal=True)
/// r1       = input + self_out
/// x2       = layer_norm2(r1)
/// cross    = multi_head_attention(x2, encoder_output, encoder_output,
///                                 is_causal=False)
/// r2       = r1 + cross
/// x3       = layer_norm3(r2)
/// ffn_out  = feed_forward(x3)
/// output   = r2 + ffn_out
/// ```
///
/// Input shape: `[tgt_seq_len, embed_dim]`.
/// Encoder output: `[src_seq_len, embed_dim]`.
/// Output shape: `[tgt_seq_len, embed_dim]`.
///
/// Note: when `tgt_seq_len == src_seq_len`, this delegates to the standard
/// `multi_head_attention_forward`. When sequence lengths differ (the typical
/// seq2seq case), we run cross-attention via a local per-head helper that
/// only assumes the embedding dim and head count line up, because the
/// shared MHA entrypoint currently requires `seq_q == seq_k == seq_v`.
pub fn decoder_block_forward(
  block: DecoderBlock,
  input: Tensor,
  encoder_output: Tensor,
) -> Result(Tensor, TensorError) {
  use x1 <- result.try(layer_norm_forward(block.norm1, input))
  use self_out <- result.try(multi_head_attention_forward(
    block.self_attention,
    x1,
    x1,
    x1,
    True,
  ))
  use r1 <- result.try(tensor.add(input, self_out))
  use x2 <- result.try(layer_norm_forward(block.norm2, r1))
  use cross_out <- result.try(cross_attention_forward(
    block.cross_attention,
    x2,
    encoder_output,
    encoder_output,
  ))
  use r2 <- result.try(tensor.add(r1, cross_out))
  use x3 <- result.try(layer_norm_forward(block.norm3, r2))
  use ffn_out <- result.try(feed_forward_forward(block.ffn, x3))
  tensor.add(r2, ffn_out)
}

// Cross-attention helper that handles `seq_q != seq_k`. Mirrors
// `multi_head_attention_forward` but splits Q from K/V sequence lengths.
//
// Steps:
// 1. Linear-project Q, K, V via the MHA weights (no bias in this iteration).
// 2. Reshape into per-head `[seq, head_dim]` slices.
// 3. Run scaled-dot-product attention per head (no causal mask).
// 4. Concat per-head outputs back to `[seq_q, embed_dim]` and apply w_o.
fn cross_attention_forward(
  mha: MultiHeadAttention,
  q: Tensor,
  k: Tensor,
  v: Tensor,
) -> Result(Tensor, TensorError) {
  use #(seq_q, _) <- result.try(rank2_dims(q, mha.embed_dim, "cross_attn q"))
  use #(seq_k, _) <- result.try(rank2_dims(k, mha.embed_dim, "cross_attn k"))
  use #(seq_v, _) <- result.try(rank2_dims(v, mha.embed_dim, "cross_attn v"))
  case seq_k == seq_v {
    False -> Error(ShapeMismatch(expected: k.shape, got: v.shape))
    True -> {
      // Linear projections (no bias in this round).
      use q_proj <- result.try(tensor.matmul(q, mha.w_q))
      use k_proj <- result.try(tensor.matmul(k, mha.w_k))
      use v_proj <- result.try(tensor.matmul(v, mha.w_v))

      use q_heads <- result.try(split_heads(
        q_proj,
        seq_q,
        mha.num_heads,
        mha.head_dim,
      ))
      use k_heads <- result.try(split_heads(
        k_proj,
        seq_k,
        mha.num_heads,
        mha.head_dim,
      ))
      use v_heads <- result.try(split_heads(
        v_proj,
        seq_v,
        mha.num_heads,
        mha.head_dim,
      ))

      let triples =
        list.zip(q_heads, list.zip(k_heads, v_heads))
        |> list.map(fn(triple) {
          let #(qh, rest) = triple
          let #(kh, vh) = rest
          #(qh, kh, vh)
        })

      use head_outputs <- result.try(
        list.try_map(triples, fn(t) {
          let #(qh, kh, vh) = t
          scaled_dot_product_attention(qh, kh, vh, None, False)
        }),
      )

      use concat <- result.try(concat_heads(
        head_outputs,
        seq_q,
        mha.num_heads,
        mha.head_dim,
      ))

      tensor.matmul(concat, mha.w_o)
    }
  }
}

fn rank2_dims(
  t: Tensor,
  embed_dim: Int,
  who: String,
) -> Result(#(Int, Int), TensorError) {
  case t.shape {
    [seq, e] if e == embed_dim -> Ok(#(seq, e))
    _ -> {
      let _ = who
      Error(ShapeMismatch(expected: [-1, embed_dim], got: t.shape))
    }
  }
}

// Reshape [seq, embed_dim] into a list of `num_heads` tensors of shape
// [seq, head_dim].
fn split_heads(
  x: Tensor,
  seq: Int,
  num_heads: Int,
  head_dim: Int,
) -> Result(List(Tensor), TensorError) {
  use data <- result.try(tensor.try_to_list(x))
  let rows = list.sized_chunk(data, num_heads * head_dim)
  let heads =
    list.range(0, num_heads - 1)
    |> list.map(fn(h) {
      let head_data =
        rows
        |> list.flat_map(fn(row) {
          row
          |> list.drop(h * head_dim)
          |> list.take(head_dim)
        })
      Tensor(data: head_data, shape: [seq, head_dim])
    })
  Ok(heads)
}

// Inverse of split_heads.
fn concat_heads(
  heads: List(Tensor),
  seq: Int,
  num_heads: Int,
  head_dim: Int,
) -> Result(Tensor, TensorError) {
  use head_rows <- result.try(
    list.try_map(heads, fn(h) {
      use d <- result.try(tensor.try_to_list(h))
      Ok(list.sized_chunk(d, head_dim))
    }),
  )
  let combined =
    list.range(0, seq - 1)
    |> list.flat_map(fn(s) {
      head_rows
      |> list.flat_map(fn(rows) {
        rows
        |> list.drop(s)
        |> list.first
        |> result.unwrap([])
      })
    })
  Ok(Tensor(data: combined, shape: [seq, num_heads * head_dim]))
}

// ---------------------------------------------------------------------------
// Transformer (encoder + decoder stack)
// ---------------------------------------------------------------------------

/// Full Transformer model: a stack of encoder blocks followed by a stack of
/// decoder blocks. Layer counts are kept explicit on the record so callers
/// can inspect the configured depth without walking the lists.
pub type Transformer {
  Transformer(
    encoder_blocks: List(EncoderBlock),
    decoder_blocks: List(DecoderBlock),
    num_encoder_layers: Int,
    num_decoder_layers: Int,
  )
}

/// Build a `Transformer` with `num_encoder_layers` encoder blocks and
/// `num_decoder_layers` decoder blocks. Each block is initialized via
/// `encoder_block_init` / `decoder_block_init` with the shared
/// `(embed_dim, num_heads, ffn_hidden_dim, activation)` config.
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads`, or when
///   any layer count is negative.
pub fn transformer_init(
  num_encoder_layers: Int,
  num_decoder_layers: Int,
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  activation: Activation,
) -> Result(Transformer, TensorError) {
  case num_encoder_layers < 0 || num_decoder_layers < 0 {
    True ->
      Error(InvalidShape(
        "transformer_init: layer counts must be non-negative (got num_encoder_layers="
        <> int.to_string(num_encoder_layers)
        <> ", num_decoder_layers="
        <> int.to_string(num_decoder_layers)
        <> ")",
      ))
    False -> {
      use encoders <- result.try(
        list.repeat(Nil, num_encoder_layers)
        |> list.try_map(fn(_) {
          encoder_block_init(embed_dim, num_heads, ffn_hidden_dim, activation)
        }),
      )
      use decoders <- result.try(
        list.repeat(Nil, num_decoder_layers)
        |> list.try_map(fn(_) {
          decoder_block_init(embed_dim, num_heads, ffn_hidden_dim, activation)
        }),
      )
      Ok(Transformer(
        encoder_blocks: encoders,
        decoder_blocks: decoders,
        num_encoder_layers: num_encoder_layers,
        num_decoder_layers: num_decoder_layers,
      ))
    }
  }
}

/// Run `src` through every encoder block in order (no causal masking).
///
/// Residual diagram (per block, pre-norm):
/// ```
/// x        = encoder_block_forward(prev, x, is_causal=False)
/// ```
pub fn transformer_encode(
  model: Transformer,
  src: Tensor,
) -> Result(Tensor, TensorError) {
  list.try_fold(model.encoder_blocks, src, fn(acc, block) {
    encoder_block_forward(block, acc, False)
  })
}

/// Run `tgt` through every decoder block in order, attending to `memory`
/// (typically the encoder output) on every cross-attention sublayer.
///
/// Residual diagram (per block, pre-norm + causal self-attn + cross-attn):
/// ```
/// x        = decoder_block_forward(prev, x, memory)
/// ```
pub fn transformer_decode(
  model: Transformer,
  tgt: Tensor,
  memory: Tensor,
) -> Result(Tensor, TensorError) {
  list.try_fold(model.decoder_blocks, tgt, fn(acc, block) {
    decoder_block_forward(block, acc, memory)
  })
}

/// End-to-end forward: encode `src` then decode `tgt` against the encoder
/// output.
///
/// ```
/// memory = transformer_encode(model, src)
/// output = transformer_decode(model, tgt, memory)
/// ```
pub fn transformer_forward(
  model: Transformer,
  src: Tensor,
  tgt: Tensor,
) -> Result(Tensor, TensorError) {
  use memory <- result.try(transformer_encode(model, src))
  transformer_decode(model, tgt, memory)
}
