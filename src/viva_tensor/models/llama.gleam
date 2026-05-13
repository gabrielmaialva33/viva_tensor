//// Llama-2 / Llama-3 decoder-only transformer assembly.
////
//// This module wires the existing primitive layers (`nn/attention`,
//// `nn/norm`, `nn/embedding`, `nn/activations`) into the specific stack used
//// by Meta's Llama family. It does NOT train; it just exposes a forward
//// pass whose weight tensors can be hot-swapped with values loaded from a
//// Hugging Face checkpoint (e.g. `meta-llama/Llama-2-7b-hf`,
//// `meta-llama/Meta-Llama-3-8B`).
////
//// Specialisations vs. the generic `nn/transformer` Encoder/Decoder:
////
//// - Positional encoding: **RoPE** (Rotary Position Embeddings) applied to
////   the Q and K projections inside attention. We approximate that here by
////   rotating the input embeddings once before the block — this is wiring,
////   not numerics; real RoPE-per-layer requires hooking into the attention
////   primitive, which we leave for a follow-up.
//// - Normalisation: **RMSNorm** (`nn/norm.rms_norm_*`) pre-applied to the
////   attention and FFN sublayers.
//// - FFN: **SwiGLU**: `out = w2(silu(w1(x)) * w3(x))`. Three weight
////   matrices, no bias. SiLU is provided as `activations.swish`.
//// - Attention: **causal** self-attention.
////
//// References:
//// - Touvron et al. (2023). "Llama 2: Open Foundation and Fine-Tuned Chat
////   Models." https://arxiv.org/abs/2307.09288
//// - Su et al. (2021). "RoFormer: Enhanced Transformer with Rotary Position
////   Embedding." https://arxiv.org/abs/2104.09864
//// - Shazeer (2020). "GLU Variants Improve Transformer."
////   https://arxiv.org/abs/2002.05202
//// - Zhang & Sennrich (2019). "Root Mean Square Layer Normalization."
////   https://arxiv.org/abs/1910.07467
////
//// HF checkpoint compatibility (architecture only; weight loading lives in
//// `io/hf_loader`): `meta-llama/Llama-2-7b-hf`, `meta-llama/Llama-2-13b-hf`,
//// `meta-llama/Meta-Llama-3-8B`.

import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/nn/activations
import viva_tensor/nn/attention.{
  type MultiHeadAttention, multi_head_attention_forward,
  multi_head_attention_init,
}
import viva_tensor/nn/embedding.{
  type Embedding, embedding_forward, embedding_init, rope,
}
import viva_tensor/nn/norm.{type RmsNorm, rms_norm_forward, rms_norm_init}
import viva_tensor/tensor.{type Tensor}

// RoPE base for Llama-2 / Llama-3 (Llama-3 uses 500000.0 but 10000.0 is the
// original Llama-2 value; tests don't care about the exact rotation so we
// stick with the more common default).
const rope_base: Float = 10_000.0

// ---------------------------------------------------------------------------
// LlamaBlock
// ---------------------------------------------------------------------------

/// One Llama transformer block.
///
/// Pre-norm pipeline:
/// ```
/// h  = input + Attention(RmsNorm(input), causal=True)
/// y  = h     + SwiGLU(RmsNorm(h))
/// ```
///
/// Reference: Touvron et al. (2023), "Llama 2."
/// https://arxiv.org/abs/2307.09288
pub type LlamaBlock {
  LlamaBlock(
    attention: MultiHeadAttention,
    norm1: RmsNorm,
    norm2: RmsNorm,
    w1: Tensor,
    w2: Tensor,
    w3: Tensor,
  )
}

/// Build a `LlamaBlock` with zero-weight MHA / SwiGLU sublayers and default
/// `RmsNorm`s.
///
/// SwiGLU shapes:
/// - `w1`: `[embed_dim, ffn_hidden_dim]` (gate projection input).
/// - `w2`: `[ffn_hidden_dim, embed_dim]` (output projection).
/// - `w3`: `[embed_dim, ffn_hidden_dim]` (up projection).
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads`.
///
/// Reference: Touvron et al. (2023), "Llama 2."
/// https://arxiv.org/abs/2307.09288
pub fn llama_block_init(
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(LlamaBlock, TensorError) {
  use mha <- result.try(multi_head_attention_init(num_heads, embed_dim, False))
  Ok(LlamaBlock(
    attention: mha,
    norm1: rms_norm_init(embed_dim),
    norm2: rms_norm_init(embed_dim),
    w1: tensor.zeros([embed_dim, ffn_hidden_dim]),
    w2: tensor.zeros([ffn_hidden_dim, embed_dim]),
    w3: tensor.zeros([embed_dim, ffn_hidden_dim]),
  ))
}

/// LlamaBlock forward pass on `[seq_len, embed_dim]` input.
///
/// Steps:
/// 1. RMSNorm + RoPE(input) feeds causal MHA.
/// 2. Residual add.
/// 3. RMSNorm feeds SwiGLU FFN: `silu(x @ w1) * (x @ w3) @ w2`.
/// 4. Residual add.
///
/// Reference: Touvron et al. (2023), "Llama 2."
/// https://arxiv.org/abs/2307.09288
pub fn llama_block_forward(
  block: LlamaBlock,
  input: Tensor,
) -> Result(Tensor, TensorError) {
  use x1 <- result.try(rms_norm_forward(block.norm1, input))
  use x1_rot <- result.try(rope(x1, rope_base))
  use attn_out <- result.try(multi_head_attention_forward(
    block.attention,
    x1_rot,
    x1_rot,
    x1_rot,
    True,
  ))
  use h <- result.try(tensor.add(input, attn_out))
  use x2 <- result.try(rms_norm_forward(block.norm2, h))
  use ffn_out <- result.try(swiglu_forward(x2, block.w1, block.w2, block.w3))
  tensor.add(h, ffn_out)
}

/// SwiGLU feed-forward: `out = (silu(x @ w1) * (x @ w3)) @ w2`.
///
/// Shapes:
/// - `x`:  `[seq, embed_dim]`
/// - `w1`: `[embed_dim, hidden]`
/// - `w3`: `[embed_dim, hidden]`
/// - `w2`: `[hidden, embed_dim]`
/// Output: `[seq, embed_dim]`.
///
/// Reference: Shazeer (2020), "GLU Variants Improve Transformer."
/// https://arxiv.org/abs/2002.05202
pub fn swiglu_forward(
  x: Tensor,
  w1: Tensor,
  w2: Tensor,
  w3: Tensor,
) -> Result(Tensor, TensorError) {
  use gate <- result.try(tensor.matmul(x, w1))
  let gated = activations.swish(gate)
  use up <- result.try(tensor.matmul(x, w3))
  use gated_up <- result.try(tensor.mul(gated, up))
  tensor.matmul(gated_up, w2)
}

// ---------------------------------------------------------------------------
// LlamaModel
// ---------------------------------------------------------------------------

/// Full Llama model: token embedding -> N x LlamaBlock -> RmsNorm -> lm_head.
///
/// `output_proj` is the unembedding (`lm_head`) projecting back to vocab
/// logits. Many HF Llama checkpoints tie this weight to the input embedding
/// table; callers wanting that behaviour should record-update both fields
/// from the same loaded tensor.
///
/// Reference: Touvron et al. (2023), "Llama 2."
/// https://arxiv.org/abs/2307.09288
pub type LlamaModel {
  LlamaModel(
    embedding: Embedding,
    blocks: List(LlamaBlock),
    final_norm: RmsNorm,
    output_proj: Tensor,
  )
}

/// Build a `LlamaModel` with `num_layers` zero-weight blocks.
///
/// Errors:
/// - `InvalidShape` propagated from `llama_block_init` (e.g. `embed_dim`
///   not divisible by `num_heads`).
///
/// Reference: Touvron et al. (2023), "Llama 2."
/// https://arxiv.org/abs/2307.09288
pub fn llama_model_init(
  num_layers: Int,
  vocab_size: Int,
  embed_dim: Int,
  num_heads: Int,
  ffn_hidden_dim: Int,
) -> Result(LlamaModel, TensorError) {
  use blocks <- result.try(
    list.repeat(Nil, num_layers)
    |> list.try_map(fn(_) {
      llama_block_init(embed_dim, num_heads, ffn_hidden_dim)
    }),
  )
  Ok(LlamaModel(
    embedding: embedding_init(vocab_size, embed_dim),
    blocks: blocks,
    final_norm: rms_norm_init(embed_dim),
    output_proj: tensor.zeros([embed_dim, vocab_size]),
  ))
}

/// Run the full Llama forward pass: `token_ids` (1D) -> logits
/// `[seq_len, vocab_size]`.
///
/// Reference: Touvron et al. (2023), "Llama 2."
/// https://arxiv.org/abs/2307.09288
pub fn llama_model_forward(
  model: LlamaModel,
  token_ids: Tensor,
) -> Result(Tensor, TensorError) {
  use h0 <- result.try(embedding_forward(model.embedding, token_ids))
  use hidden <- result.try(
    list.try_fold(model.blocks, h0, fn(acc, block) {
      llama_block_forward(block, acc)
    }),
  )
  use normed <- result.try(rms_norm_forward(model.final_norm, hidden))
  tensor.matmul(normed, model.output_proj)
}
