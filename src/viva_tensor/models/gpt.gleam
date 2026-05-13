//// GPT-2 / GPT-3 style decoder-only transformer assembly.
////
//// Wires the existing primitives into the GPT-2/3 stack. Forward pass
//// accepts weights loaded from a HF checkpoint such as
//// `openai-community/gpt2`, `openai-community/gpt2-medium`, or any
//// `gpt2`-derived model.
////
//// Specialisations vs. the generic `nn/transformer` EncoderBlock:
////
//// - Positional encoding: **learned** absolute positions
////   (`LearnedPositionalEncoding`).
//// - Normalisation: **LayerNorm** (pre-norm — GPT-2 uses pre-LN, unlike the
////   original Vaswani-style post-LN).
//// - FFN activation: **GELU**.
//// - Attention: **causal** self-attention. We re-use `EncoderBlock` with
////   `is_causal=True`, which gives the same residual diagram as a GPT block
////   (self-attn + FFN, both pre-normed). No cross-attention here.
////
//// References:
//// - Radford et al. (2019). "Language Models are Unsupervised Multitask
////   Learners." (GPT-2.) https://openai.com/research/better-language-models
//// - Brown et al. (2020). "Language Models are Few-Shot Learners." (GPT-3.)
////   https://arxiv.org/abs/2005.14165
//// - Xiong et al. (2020). "On Layer Normalization in the Transformer
////   Architecture." (Pre-LN.) https://arxiv.org/abs/2002.04745
////
//// HF checkpoint compatibility: `openai-community/gpt2`,
//// `openai-community/gpt2-medium`, `openai-community/gpt2-large`,
//// `openai-community/gpt2-xl`.

import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/nn/embedding.{
  type Embedding, type LearnedPositionalEncoding, embedding_forward,
  embedding_init, learned_positional_forward, learned_positional_init,
}
import viva_tensor/nn/norm.{type LayerNorm, layer_norm_forward, layer_norm_init}
import viva_tensor/nn/transformer.{
  type EncoderBlock, GeluAct, encoder_block_forward, encoder_block_init,
}
import viva_tensor/tensor.{type Tensor}

// ---------------------------------------------------------------------------
// GptBlock
// ---------------------------------------------------------------------------

/// One GPT-2 / GPT-3 block.
///
/// Re-uses `nn/transformer.EncoderBlock` with `is_causal=True` and GELU
/// activation in the FFN sublayer. Identical residual diagram to the generic
/// pre-norm encoder block.
///
/// Reference: Radford et al. (2019), "Language Models are Unsupervised
/// Multitask Learners." (GPT-2.)
pub type GptBlock {
  GptBlock(decoder_sublayer: EncoderBlock)
}

/// Build a `GptBlock` with zero-weight sublayers (GELU FFN, causal MHA).
///
/// Reference: Radford et al. (2019).
pub fn gpt_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(GptBlock, TensorError) {
  use enc <- result.try(encoder_block_init(
    embed_dim,
    num_heads,
    ffn_hidden_dim,
    GeluAct,
  ))
  Ok(GptBlock(decoder_sublayer: enc))
}

/// Forward pass on `[seq_len, embed_dim]` input. Causal self-attention.
///
/// Reference: Radford et al. (2019).
pub fn gpt_block_forward(
  block: GptBlock,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  encoder_block_forward(block.decoder_sublayer, input, True)
}

// ---------------------------------------------------------------------------
// GptModel
// ---------------------------------------------------------------------------

/// Full GPT model: token + position embeddings -> N GptBlocks -> LayerNorm
/// -> lm_head.
///
/// `output_proj` is the unembedding projection to vocab logits; HF GPT-2
/// ties it to `token_embeddings.weight`. Callers wanting tied weights should
/// record-update both fields with the same loaded tensor.
///
/// Reference: Radford et al. (2019).
pub type GptModel {
  GptModel(
    token_embeddings: Embedding,
    position_embeddings: LearnedPositionalEncoding,
    blocks: List(GptBlock),
    final_norm: LayerNorm,
    output_proj: Tensor,
  )
}

/// Build a `GptModel` with `num_layers` zero-weight blocks.
///
/// Errors:
/// - `InvalidShape` propagated from `gpt_block_init`.
///
/// Reference: Radford et al. (2019).
pub fn gpt_model_init(
  num_layers: Int,
  vocab_size: Int,
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  max_position: Int,
) -> Result(GptModel, TensorError) {
  use blocks <- result.try(
    list.repeat(Nil, num_layers)
    |> list.try_map(fn(_) {
      gpt_block_init(embed_dim, num_heads, ffn_hidden_dim)
    }),
  )
  Ok(GptModel(
    token_embeddings: embedding_init(vocab_size, embed_dim),
    position_embeddings: learned_positional_init(max_position, embed_dim),
    blocks: blocks,
    final_norm: layer_norm_init(embed_dim),
    output_proj: tensor.zeros([embed_dim, vocab_size]),
  ))
}

/// End-to-end forward: 1D `token_ids` `[seq_len]` -> logits
/// `[seq_len, vocab_size]`.
///
/// Reference: Radford et al. (2019).
pub fn gpt_model_forward(
  model: GptModel,
  token_ids: Tensor,
) -> Result(Tensor, TensorError) {
  use word <- result.try(embedding_forward(model.token_embeddings, token_ids))
  let seq_len = case word.shape {
    [s, _] -> s
    _ -> 0
  }
  use pos <- result.try(learned_positional_forward(
    model.position_embeddings,
    seq_len,
  ))
  use h0 <- result.try(tensor.add(word, pos))
  use hidden <- result.try(
    list.try_fold(model.blocks, h0, fn(acc, block) {
      gpt_block_forward(block, acc)
    }),
  )
  use normed <- result.try(layer_norm_forward(model.final_norm, hidden))
  tensor.matmul(normed, model.output_proj)
}
