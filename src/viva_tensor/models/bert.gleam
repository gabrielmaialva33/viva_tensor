//// BERT encoder-only transformer assembly.
////
//// Wires the existing primitives into the BERT-base encoder stack used by
//// Devlin et al. (2018). The forward pass is intended to consume weights
//// loaded from a Hugging Face checkpoint such as `bert-base-uncased`,
//// `bert-base-cased`, or any descendant (`distilbert-base-uncased` shares
//// the input embedding layout).
////
//// Specialisations vs. the generic `nn/transformer` EncoderBlock:
////
//// - Positional encoding: **learned** positional embeddings
////   (`LearnedPositionalEncoding`).
//// - Token-type / segment embeddings: a small `Embedding` of size 2.
//// - Embedding layer composes `word + position + token_type` then
////   `LayerNorm`s the sum.
//// - Normalisation: **LayerNorm** (pre-norm in this assembly so we can
////   reuse `nn/transformer.EncoderBlock`).
//// - FFN activation: **GELU**.
//// - Attention: **bidirectional** (`is_causal=False`).
////
//// References:
//// - Devlin et al. (2018). "BERT: Pre-training of Deep Bidirectional
////   Transformers for Language Understanding."
////   https://arxiv.org/abs/1810.04805
//// - Vaswani et al. (2017). "Attention Is All You Need."
////   https://arxiv.org/abs/1706.03762
////
//// HF checkpoint compatibility: `bert-base-uncased`, `bert-base-cased`,
//// `bert-large-uncased`.

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
// BertEmbedding
// ---------------------------------------------------------------------------

/// BERT input embedding: word + position + token_type, followed by LayerNorm.
///
/// Reference: Devlin et al. (2018), section 3.1.
/// https://arxiv.org/abs/1810.04805
pub type BertEmbedding {
  BertEmbedding(
    word_embeddings: Embedding,
    position_embeddings: LearnedPositionalEncoding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
  )
}

/// Build a `BertEmbedding` with zero-weight tables and a default `LayerNorm`.
///
/// `num_token_types` is typically `2` for the next-sentence-prediction style
/// segment IDs used by the original BERT.
///
/// Reference: Devlin et al. (2018).
/// https://arxiv.org/abs/1810.04805
pub fn bert_embedding_init(
  vocab_size: Int,
  embed_dim: Int,
  max_position: Int,
  num_token_types: Int,
) -> BertEmbedding {
  BertEmbedding(
    word_embeddings: embedding_init(vocab_size, embed_dim),
    position_embeddings: learned_positional_init(max_position, embed_dim),
    token_type_embeddings: embedding_init(num_token_types, embed_dim),
    layer_norm: layer_norm_init(embed_dim),
  )
}

/// Forward pass over `[seq_len]` token ids + `[seq_len]` token type ids.
///
/// Returns `[seq_len, embed_dim]` hidden states ready for the encoder stack.
///
/// Reference: Devlin et al. (2018), section 3.1.
/// https://arxiv.org/abs/1810.04805
pub fn bert_embedding_forward(
  layer: BertEmbedding,
  token_ids: Tensor,
  token_type_ids: Tensor,
) -> Result(Tensor, TensorError) {
  use word <- result.try(embedding_forward(layer.word_embeddings, token_ids))
  let seq_len = case word.shape {
    [s, _] -> s
    _ -> 0
  }
  use pos <- result.try(learned_positional_forward(
    layer.position_embeddings,
    seq_len,
  ))
  use tt <- result.try(embedding_forward(
    layer.token_type_embeddings,
    token_type_ids,
  ))
  use sum1 <- result.try(tensor.add(word, pos))
  use sum2 <- result.try(tensor.add(sum1, tt))
  layer_norm_forward(layer.layer_norm, sum2)
}

// ---------------------------------------------------------------------------
// BertBlock
// ---------------------------------------------------------------------------

/// Single BERT encoder block.
///
/// Re-uses `nn/transformer.EncoderBlock` with `is_causal=False` and GELU
/// activation in the FFN sublayer.
///
/// Reference: Devlin et al. (2018), section 3.
/// https://arxiv.org/abs/1810.04805
pub type BertBlock {
  BertBlock(encoder: EncoderBlock)
}

/// Build a `BertBlock` with zero-weight sublayers (GELU FFN).
///
/// Reference: Devlin et al. (2018).
/// https://arxiv.org/abs/1810.04805
pub fn bert_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(BertBlock, TensorError) {
  use enc <- result.try(encoder_block_init(
    embed_dim,
    num_heads,
    ffn_hidden_dim,
    GeluAct,
  ))
  Ok(BertBlock(encoder: enc))
}

/// Forward pass on `[seq_len, embed_dim]` input. Non-causal attention.
///
/// Reference: Devlin et al. (2018).
/// https://arxiv.org/abs/1810.04805
pub fn bert_block_forward(
  block: BertBlock,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  encoder_block_forward(block.encoder, input, False)
}

// ---------------------------------------------------------------------------
// BertModel
// ---------------------------------------------------------------------------

/// Full BERT model: embedding + N encoder blocks. No pooler / no LM head in
/// this assembly — callers can attach a task head by reading
/// `model.blocks`' final hidden state.
///
/// Reference: Devlin et al. (2018).
/// https://arxiv.org/abs/1810.04805
pub type BertModel {
  BertModel(embedding: BertEmbedding, blocks: List(BertBlock))
}

/// Build a `BertModel` with `num_layers` zero-weight blocks.
///
/// Reference: Devlin et al. (2018).
/// https://arxiv.org/abs/1810.04805
pub fn bert_model_init(
  num_layers: Int,
  vocab_size: Int,
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
  max_position: Int,
) -> Result(BertModel, TensorError) {
  use blocks <- result.try(
    list.repeat(Nil, num_layers)
    |> list.try_map(fn(_) {
      bert_block_init(embed_dim, num_heads, ffn_hidden_dim)
    }),
  )
  Ok(BertModel(
    embedding: bert_embedding_init(vocab_size, embed_dim, max_position, 2),
    blocks: blocks,
  ))
}

/// End-to-end forward: `[seq_len]` token ids and `[seq_len]` token_type ids
/// -> final hidden states `[seq_len, embed_dim]`.
///
/// Reference: Devlin et al. (2018).
/// https://arxiv.org/abs/1810.04805
pub fn bert_model_forward(
  model: BertModel,
  token_ids: Tensor,
  token_type_ids: Tensor,
) -> Result(Tensor, TensorError) {
  use h0 <- result.try(bert_embedding_forward(
    model.embedding,
    token_ids,
    token_type_ids,
  ))
  list.try_fold(model.blocks, h0, fn(acc, block) {
    bert_block_forward(block, acc)
  })
}
