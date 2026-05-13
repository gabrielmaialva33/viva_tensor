//// T5 encoder-decoder transformer assembly.
////
//// Wires the existing primitives into the T5 stack (Raffel et al., 2020).
//// Forward pass accepts weights loaded from a HF checkpoint such as
//// `google-t5/t5-small`, `google-t5/t5-base`, `google-t5/t5-large`, or any
//// T5-v1.1 variant.
////
//// Specialisations vs. the generic `nn/transformer` Encoder/Decoder:
////
//// - Positional encoding: T5 uses **relative position bias** added inside
////   attention. Our attention primitive doesn't expose a bias slot yet, so
////   we keep the bias tensor off the block for the time being — this is
////   wiring, not numerics. Real RPE support requires extending the
////   attention primitive.
//// - Normalisation: **RMSNorm** (pre-norm), no bias term.
//// - FFN: **GeGLU** (T5-v1.1 default). `out = w2(gelu(w_gate(x)) * w1(x))`.
////   Three weight tensors. We picked GeGLU over T5-base's plain ReLU because
////   v1.1 / Flan-T5 weights are far more common in the HF hub today.
//// - Attention: encoder uses **non-causal** self-attention, decoder uses
////   **causal** self-attention followed by cross-attention over the encoder
////   memory.
////
//// Cross-attention currently requires `tgt_seq_len == src_seq_len` because
//// `nn/attention.multi_head_attention_forward` only supports that case.
//// Mixed-length seq2seq inputs are caller-pre-padded for now.
////
//// References:
//// - Raffel et al. (2020). "Exploring the Limits of Transfer Learning with
////   a Unified Text-to-Text Transformer." (T5.)
////   https://arxiv.org/abs/1910.10683
//// - Shazeer (2018). "Self-attention with Relative Position
////   Representations." (Relative bias roots.)
////   https://arxiv.org/abs/1803.02155
//// - Shazeer (2020). "GLU Variants Improve Transformer." (GeGLU.)
////   https://arxiv.org/abs/2002.05202
////
//// HF checkpoint compatibility: `google-t5/t5-small`, `google-t5/t5-base`,
//// `google-t5/t5-large`, `google/t5-v1_1-base`, `google/flan-t5-base`.

import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/nn/activations
import viva_tensor/nn/attention.{
  type MultiHeadAttention, multi_head_attention_forward,
  multi_head_attention_init,
}
import viva_tensor/nn/embedding.{
  type Embedding, embedding_forward, embedding_init,
}
import viva_tensor/nn/norm.{type RmsNorm, rms_norm_forward, rms_norm_init}
import viva_tensor/tensor.{type Tensor}

// ---------------------------------------------------------------------------
// T5Block (shared encoder + decoder shape)
// ---------------------------------------------------------------------------

/// A T5 block. Encoder blocks set `cross_attention = None` / `norm3 = None`;
/// decoder blocks populate both. The GeGLU FFN keeps three weight tensors
/// (`ffn_w_gate`, `ffn_w1`, `ffn_w2`) and uses pre-norm RMSNorm before each
/// sublayer.
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub type T5Block {
  T5Block(
    self_attention: MultiHeadAttention,
    cross_attention: Option(MultiHeadAttention),
    ffn_w1: Tensor,
    ffn_w_gate: Tensor,
    ffn_w2: Tensor,
    norm1: RmsNorm,
    norm2: RmsNorm,
    norm3: Option(RmsNorm),
  )
}

/// Build a T5 encoder block with zero-weight MHA / GeGLU sublayers.
///
/// FFN shapes:
/// - `ffn_w_gate`: `[embed_dim, ffn_hidden_dim]` (gate projection).
/// - `ffn_w1`:     `[embed_dim, ffn_hidden_dim]` (up projection).
/// - `ffn_w2`:     `[ffn_hidden_dim, embed_dim]` (down projection).
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads`.
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub fn t5_encoder_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(T5Block, TensorError) {
  use self_mha <- result.try(multi_head_attention_init(
    num_heads,
    embed_dim,
    False,
  ))
  Ok(T5Block(
    self_attention: self_mha,
    cross_attention: None,
    ffn_w1: tensor.zeros([embed_dim, ffn_hidden_dim]),
    ffn_w_gate: tensor.zeros([embed_dim, ffn_hidden_dim]),
    ffn_w2: tensor.zeros([ffn_hidden_dim, embed_dim]),
    norm1: rms_norm_init(embed_dim),
    norm2: rms_norm_init(embed_dim),
    norm3: None,
  ))
}

/// Build a T5 decoder block (causal self-attn + cross-attn + GeGLU FFN).
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads`.
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub fn t5_decoder_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(T5Block, TensorError) {
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
  Ok(T5Block(
    self_attention: self_mha,
    cross_attention: Some(cross_mha),
    ffn_w1: tensor.zeros([embed_dim, ffn_hidden_dim]),
    ffn_w_gate: tensor.zeros([embed_dim, ffn_hidden_dim]),
    ffn_w2: tensor.zeros([ffn_hidden_dim, embed_dim]),
    norm1: rms_norm_init(embed_dim),
    norm2: rms_norm_init(embed_dim),
    norm3: Some(rms_norm_init(embed_dim)),
  ))
}

/// Encoder forward (pre-norm, non-causal self-attn + GeGLU FFN).
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub fn t5_encoder_block_forward(
  block: T5Block,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  use x1 <- result.try(rms_norm_forward(block.norm1, input))
  use attn_out <- result.try(multi_head_attention_forward(
    block.self_attention,
    x1,
    x1,
    x1,
    False,
  ))
  use h <- result.try(tensor.add(input, attn_out))
  use x2 <- result.try(rms_norm_forward(block.norm2, h))
  use ffn_out <- result.try(geglu_forward(
    x2,
    block.ffn_w_gate,
    block.ffn_w1,
    block.ffn_w2,
  ))
  tensor.add(h, ffn_out)
}

/// Decoder forward (pre-norm, causal self-attn + cross-attn over `memory`
/// + GeGLU FFN).
///
/// Constraint: `memory` must share the target sequence length, since the
/// underlying attention primitive currently requires `seq_q == seq_k`.
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub fn t5_decoder_block_forward(
  block: T5Block,
  input: Tensor,
  memory: Tensor,
) -> Result(Tensor, TensorError) {
  use x1 <- result.try(rms_norm_forward(block.norm1, input))
  use self_out <- result.try(multi_head_attention_forward(
    block.self_attention,
    x1,
    x1,
    x1,
    True,
  ))
  use r1 <- result.try(tensor.add(input, self_out))
  use x2 <- result.try(rms_norm_forward(block.norm2, r1))
  use cross_out <- result.try(case block.cross_attention {
    Some(cmha) -> multi_head_attention_forward(cmha, x2, memory, memory, False)
    None -> Ok(tensor.zeros_like(x2))
  })
  use r2 <- result.try(tensor.add(r1, cross_out))
  use x3 <- result.try(case block.norm3 {
    Some(n3) -> rms_norm_forward(n3, r2)
    None -> rms_norm_forward(block.norm2, r2)
  })
  use ffn_out <- result.try(geglu_forward(
    x3,
    block.ffn_w_gate,
    block.ffn_w1,
    block.ffn_w2,
  ))
  tensor.add(r2, ffn_out)
}

/// GeGLU feed-forward: `out = (gelu(x @ w_gate) * (x @ w1)) @ w2`.
///
/// Reference: Shazeer (2020), "GLU Variants Improve Transformer."
/// https://arxiv.org/abs/2002.05202
fn geglu_forward(
  x: Tensor,
  w_gate: Tensor,
  w1: Tensor,
  w2: Tensor,
) -> Result(Tensor, TensorError) {
  use gate <- result.try(tensor.matmul(x, w_gate))
  let gated = activations.gelu(gate)
  use up <- result.try(tensor.matmul(x, w1))
  use gated_up <- result.try(tensor.mul(gated, up))
  tensor.matmul(gated_up, w2)
}

// ---------------------------------------------------------------------------
// T5Model
// ---------------------------------------------------------------------------

/// Full T5 encoder-decoder model.
///
/// The token embedding `embedding` is shared between encoder and decoder
/// inputs (matching HF T5's tied input embeddings). `output_proj` projects
/// decoder hidden states back to vocab logits; HF T5 ties this to
/// `embedding.weight` when `tie_word_embeddings=True` (the default for
/// `t5-base`).
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub type T5Model {
  T5Model(
    embedding: Embedding,
    encoder_blocks: List(T5Block),
    decoder_blocks: List(T5Block),
    encoder_final_norm: RmsNorm,
    decoder_final_norm: RmsNorm,
    output_proj: Tensor,
  )
}

/// Build a T5 model with zero-weight encoder+decoder stacks.
///
/// Errors:
/// - `InvalidShape` propagated from block initializers.
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub fn t5_model_init(
  num_encoder_layers: Int,
  num_decoder_layers: Int,
  vocab_size: Int,
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(T5Model, TensorError) {
  use encoder_blocks <- result.try(
    list.repeat(Nil, num_encoder_layers)
    |> list.try_map(fn(_) {
      t5_encoder_block_init(embed_dim, num_heads, ffn_hidden_dim)
    }),
  )
  use decoder_blocks <- result.try(
    list.repeat(Nil, num_decoder_layers)
    |> list.try_map(fn(_) {
      t5_decoder_block_init(embed_dim, num_heads, ffn_hidden_dim)
    }),
  )
  Ok(T5Model(
    embedding: embedding_init(vocab_size, embed_dim),
    encoder_blocks: encoder_blocks,
    decoder_blocks: decoder_blocks,
    encoder_final_norm: rms_norm_init(embed_dim),
    decoder_final_norm: rms_norm_init(embed_dim),
    output_proj: tensor.zeros([embed_dim, vocab_size]),
  ))
}

/// End-to-end forward: encode `src_token_ids`, decode `tgt_token_ids`
/// attending to the encoder memory, project to vocab logits.
///
/// Returns `[tgt_seq_len, vocab_size]` logits. Currently requires
/// `tgt_seq_len == src_seq_len` (see `t5_decoder_block_forward`).
///
/// Reference: Raffel et al. (2020). https://arxiv.org/abs/1910.10683
pub fn t5_model_forward(
  model: T5Model,
  src_token_ids: Tensor,
  tgt_token_ids: Tensor,
) -> Result(Tensor, TensorError) {
  use src_emb <- result.try(embedding_forward(model.embedding, src_token_ids))
  use memory_raw <- result.try(
    list.try_fold(model.encoder_blocks, src_emb, fn(acc, block) {
      t5_encoder_block_forward(block, acc)
    }),
  )
  use memory <- result.try(rms_norm_forward(
    model.encoder_final_norm,
    memory_raw,
  ))

  use tgt_emb <- result.try(embedding_forward(model.embedding, tgt_token_ids))
  use decoded_raw <- result.try(
    list.try_fold(model.decoder_blocks, tgt_emb, fn(acc, block) {
      t5_decoder_block_forward(block, acc, memory)
    }),
  )
  use decoded <- result.try(rms_norm_forward(
    model.decoder_final_norm,
    decoded_raw,
  ))
  tensor.matmul(decoded, model.output_proj)
}
