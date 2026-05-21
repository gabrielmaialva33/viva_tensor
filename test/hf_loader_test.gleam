//// HuggingFace SafeTensors loader unit tests.
////
//// Builds synthetic in-memory weight dicts that mimic the flat
//// `name -> tensor` structure produced by `transformers`-style exports,
//// then verifies the loader projects them into the right viva_tensor
//// records with the right shapes/values.

import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/option.{type Option, Some}
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/io/hf_loader.{ShapeMismatch, WeightNotFound}
import viva_tensor/nn/attention.{MultiHeadAttention}
import viva_tensor/nn/norm.{LayerNorm}
import viva_tensor/nn/transformer.{
  DecoderBlock, EncoderBlock, FeedForward, ReluAct, Transformer,
} as nn_transformer
import viva_tensor/tensor.{type Tensor, Tensor}

// ---------------------------------------------------------------------------
// Synthetic-data helpers
// ---------------------------------------------------------------------------

const rtol: Float = 1.0e-9

const atol: Float = 1.0e-9

fn ramp(n: Int, start: Float) -> List(Float) {
  case n <= 0 {
    True -> []
    False ->
      range_int(0, n - 1)
      |> list.map(fn(i) { start +. int.to_float(i) })
  }
}

fn make_tensor(shape: List(Int), seed: Float) -> Tensor {
  let n = list.fold(shape, 1, fn(acc, d) { acc * d })
  Tensor(data: ramp(n, seed), shape: shape)
}

fn put(
  d: Dict(String, Tensor),
  name: String,
  shape: List(Int),
  seed: Float,
) -> Dict(String, Tensor) {
  dict.insert(d, name, make_tensor(shape, seed))
}

fn mha_weights(
  base: Dict(String, Tensor),
  prefix: String,
  embed_dim: Int,
  seed_offset: Float,
) -> Dict(String, Tensor) {
  base
  |> put(prefix <> ".q_proj.weight", [embed_dim, embed_dim], seed_offset +. 0.0)
  |> put(prefix <> ".q_proj.bias", [embed_dim], seed_offset +. 1.0)
  |> put(prefix <> ".k_proj.weight", [embed_dim, embed_dim], seed_offset +. 2.0)
  |> put(prefix <> ".k_proj.bias", [embed_dim], seed_offset +. 3.0)
  |> put(prefix <> ".v_proj.weight", [embed_dim, embed_dim], seed_offset +. 4.0)
  |> put(prefix <> ".v_proj.bias", [embed_dim], seed_offset +. 5.0)
  |> put(
    prefix <> ".out_proj.weight",
    [embed_dim, embed_dim],
    seed_offset +. 6.0,
  )
  |> put(prefix <> ".out_proj.bias", [embed_dim], seed_offset +. 7.0)
}

fn ffn_weights(
  base: Dict(String, Tensor),
  prefix: String,
  embed_dim: Int,
  hidden_dim: Int,
  seed_offset: Float,
) -> Dict(String, Tensor) {
  base
  |> put(
    prefix <> ".linear1.weight",
    [embed_dim, hidden_dim],
    seed_offset +. 0.0,
  )
  |> put(prefix <> ".linear1.bias", [hidden_dim], seed_offset +. 1.0)
  |> put(
    prefix <> ".linear2.weight",
    [hidden_dim, embed_dim],
    seed_offset +. 2.0,
  )
  |> put(prefix <> ".linear2.bias", [embed_dim], seed_offset +. 3.0)
}

fn norm_weights(
  base: Dict(String, Tensor),
  prefix: String,
  num_features: Int,
  seed_offset: Float,
) -> Dict(String, Tensor) {
  base
  |> put(prefix <> ".weight", [num_features], seed_offset)
  |> put(prefix <> ".bias", [num_features], seed_offset +. 1.0)
}

fn encoder_block_weights(
  base: Dict(String, Tensor),
  prefix: String,
  embed_dim: Int,
  hidden_dim: Int,
  seed_offset: Float,
) -> Dict(String, Tensor) {
  base
  |> mha_weights(prefix <> ".self_attn", embed_dim, seed_offset)
  |> norm_weights(prefix <> ".norm1", embed_dim, seed_offset +. 10.0)
  |> norm_weights(prefix <> ".norm2", embed_dim, seed_offset +. 20.0)
  |> ffn_weights(prefix <> ".ffn", embed_dim, hidden_dim, seed_offset +. 30.0)
}

fn decoder_block_weights(
  base: Dict(String, Tensor),
  prefix: String,
  embed_dim: Int,
  hidden_dim: Int,
  seed_offset: Float,
) -> Dict(String, Tensor) {
  base
  |> mha_weights(prefix <> ".self_attn", embed_dim, seed_offset)
  |> mha_weights(prefix <> ".cross_attn", embed_dim, seed_offset +. 8.0)
  |> norm_weights(prefix <> ".norm1", embed_dim, seed_offset +. 16.0)
  |> norm_weights(prefix <> ".norm2", embed_dim, seed_offset +. 26.0)
  |> norm_weights(prefix <> ".norm3", embed_dim, seed_offset +. 36.0)
  |> ffn_weights(prefix <> ".ffn", embed_dim, hidden_dim, seed_offset +. 46.0)
}

fn shape_equal(tn: Tensor, expected: List(Int)) -> Nil {
  t.shape(tn) |> should.equal(expected)
}

fn data_close(tn: Tensor, expected: List(Float)) -> Nil {
  numerics.lists_close(t.to_list(tn), expected, rtol, atol)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// Embedding
// ---------------------------------------------------------------------------

pub fn load_embedding_test() {
  let weights = dict.new() |> put("embed.weight", [10, 4], 0.0)
  let assert Ok(emb) = t.load_embedding(weights, "embed", 10, 4)
  emb.num_embeddings |> should.equal(10)
  emb.embedding_dim |> should.equal(4)
  shape_equal(emb.weight, [10, 4])
  data_close(emb.weight, ramp(40, 0.0))
}

pub fn load_embedding_missing_test() {
  let weights: Dict(String, Tensor) = dict.new()
  case t.load_embedding(weights, "embed", 10, 4) {
    Error(WeightNotFound(name)) -> name |> should.equal("embed.weight")
    other -> {
      let _ = other
      should.fail()
    }
  }
}

pub fn load_embedding_shape_test() {
  // Provide [9, 4] but ask for [10, 4].
  let weights = dict.new() |> put("embed.weight", [9, 4], 0.0)
  case t.load_embedding(weights, "embed", 10, 4) {
    Error(ShapeMismatch(name, expected, got)) -> {
      name |> should.equal("embed.weight")
      expected |> should.equal([10, 4])
      got |> should.equal([9, 4])
    }
    other -> {
      let _ = other
      should.fail()
    }
  }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

pub fn load_layer_norm_test() {
  let weights =
    dict.new()
    |> put("ln.weight", [4], 1.0)
    |> put("ln.bias", [4], 5.0)
  let assert Ok(ln) = t.load_layer_norm(weights, "ln", 4)
  shape_equal(ln.scale, [4])
  shape_equal(ln.bias, [4])
  data_close(ln.scale, [1.0, 2.0, 3.0, 4.0])
  data_close(ln.bias, [5.0, 6.0, 7.0, 8.0])
}

pub fn load_layer_norm_missing_bias_test() {
  // weight present, bias absent
  let weights = dict.new() |> put("ln.weight", [4], 1.0)
  case t.load_layer_norm(weights, "ln", 4) {
    Error(WeightNotFound(name)) -> name |> should.equal("ln.bias")
    other -> {
      let _ = other
      should.fail()
    }
  }
}

// ---------------------------------------------------------------------------
// MultiHeadAttention
// ---------------------------------------------------------------------------

pub fn load_mha_test() {
  let embed_dim = 4
  let weights = mha_weights(dict.new(), "self_attn", embed_dim, 0.0)
  let assert Ok(mha) =
    t.load_multi_head_attention(weights, "self_attn", 2, embed_dim)
  mha.num_heads |> should.equal(2)
  mha.embed_dim |> should.equal(4)
  mha.head_dim |> should.equal(2)
  shape_equal(mha.w_q, [4, 4])
  shape_equal(mha.w_k, [4, 4])
  shape_equal(mha.w_v, [4, 4])
  shape_equal(mha.w_o, [4, 4])
  // Biases should all be Some([4]) — verify length and one shape.
  case mha.b_q, mha.b_k, mha.b_v, mha.b_o {
    Some(bq), Some(bk), Some(bv), Some(bo) -> {
      shape_equal(bq, [4])
      shape_equal(bk, [4])
      shape_equal(bv, [4])
      shape_equal(bo, [4])
    }
    _, _, _, _ -> should.fail()
  }
  // Spot-check the q_proj weight values (16 ramp from 0.0).
  data_close(mha.w_q, ramp(16, 0.0))
}

pub fn load_mha_missing_test() {
  let embed_dim = 4
  // Build the MHA dict then drop one key.
  let full = mha_weights(dict.new(), "self_attn", embed_dim, 0.0)
  let partial = dict.delete(full, "self_attn.v_proj.weight")
  case t.load_multi_head_attention(partial, "self_attn", 2, embed_dim) {
    Error(WeightNotFound(name)) ->
      name |> should.equal("self_attn.v_proj.weight")
    other -> {
      let _ = other
      should.fail()
    }
  }
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

pub fn load_feed_forward_test() {
  let weights = ffn_weights(dict.new(), "ffn", 4, 8, 0.0)
  let assert Ok(ff) = t.load_feed_forward(weights, "ffn", 4, 8, ReluAct)
  shape_equal(ff.w1, [4, 8])
  shape_equal(ff.b1, [8])
  shape_equal(ff.w2, [8, 4])
  shape_equal(ff.b2, [4])
  ff.activation |> should.equal(ReluAct)
}

// ---------------------------------------------------------------------------
// EncoderBlock
// ---------------------------------------------------------------------------

pub fn load_encoder_block_test() {
  let embed_dim = 4
  let hidden_dim = 8
  let weights =
    encoder_block_weights(
      dict.new(),
      "encoder.layers.0",
      embed_dim,
      hidden_dim,
      0.0,
    )
  let assert Ok(block) =
    t.load_encoder_block(
      weights,
      "encoder.layers.0",
      2,
      embed_dim,
      hidden_dim,
      ReluAct,
    )
  shape_equal(block.attention.w_q, [embed_dim, embed_dim])
  shape_equal(block.norm1.scale, [embed_dim])
  shape_equal(block.norm2.bias, [embed_dim])
  shape_equal(block.ffn.w1, [embed_dim, hidden_dim])
  shape_equal(block.ffn.w2, [hidden_dim, embed_dim])
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

pub fn load_transformer_test() {
  let embed_dim = 4
  let hidden_dim = 8
  let weights =
    dict.new()
    |> encoder_block_weights("encoder.layers.0", embed_dim, hidden_dim, 0.0)
    |> decoder_block_weights("decoder.layers.0", embed_dim, hidden_dim, 100.0)
  let assert Ok(model) =
    t.load_transformer(weights, 1, 1, embed_dim, 2, hidden_dim, ReluAct)
  model.num_encoder_layers |> should.equal(1)
  model.num_decoder_layers |> should.equal(1)
  list.length(model.encoder_blocks) |> should.equal(1)
  list.length(model.decoder_blocks) |> should.equal(1)
  let assert [enc, ..] = model.encoder_blocks
  let assert [dec, ..] = model.decoder_blocks
  shape_equal(enc.attention.w_q, [embed_dim, embed_dim])
  shape_equal(enc.norm1.scale, [embed_dim])
  shape_equal(enc.ffn.w1, [embed_dim, hidden_dim])
  shape_equal(dec.self_attention.w_q, [embed_dim, embed_dim])
  shape_equal(dec.cross_attention.w_q, [embed_dim, embed_dim])
  shape_equal(dec.norm3.scale, [embed_dim])
  shape_equal(dec.ffn.w2, [hidden_dim, embed_dim])
}

// ---------------------------------------------------------------------------
// Roundtrip: build a Transformer, materialize all its weights into a flat
// dict, then load it back and verify the loaded model has identical tensors.
// ---------------------------------------------------------------------------

fn flatten_norm(
  base: Dict(String, Tensor),
  prefix: String,
  ln: t.LayerNorm,
) -> Dict(String, Tensor) {
  base
  |> dict.insert(prefix <> ".weight", ln.scale)
  |> dict.insert(prefix <> ".bias", ln.bias)
}

fn unwrap_bias(b: Option(Tensor)) -> Tensor {
  case b {
    Some(t) -> t
    _ -> Tensor(data: [], shape: [])
  }
}

fn flatten_mha(
  base: Dict(String, Tensor),
  prefix: String,
  mha: t.MultiHeadAttention,
) -> Dict(String, Tensor) {
  base
  |> dict.insert(prefix <> ".q_proj.weight", mha.w_q)
  |> dict.insert(prefix <> ".q_proj.bias", unwrap_bias(mha.b_q))
  |> dict.insert(prefix <> ".k_proj.weight", mha.w_k)
  |> dict.insert(prefix <> ".k_proj.bias", unwrap_bias(mha.b_k))
  |> dict.insert(prefix <> ".v_proj.weight", mha.w_v)
  |> dict.insert(prefix <> ".v_proj.bias", unwrap_bias(mha.b_v))
  |> dict.insert(prefix <> ".out_proj.weight", mha.w_o)
  |> dict.insert(prefix <> ".out_proj.bias", unwrap_bias(mha.b_o))
}

fn flatten_ffn(
  base: Dict(String, Tensor),
  prefix: String,
  ff: t.FeedForward,
) -> Dict(String, Tensor) {
  base
  |> dict.insert(prefix <> ".linear1.weight", ff.w1)
  |> dict.insert(prefix <> ".linear1.bias", ff.b1)
  |> dict.insert(prefix <> ".linear2.weight", ff.w2)
  |> dict.insert(prefix <> ".linear2.bias", ff.b2)
}

fn flatten_encoder(
  base: Dict(String, Tensor),
  prefix: String,
  block: t.EncoderBlock,
) -> Dict(String, Tensor) {
  base
  |> flatten_mha(prefix <> ".self_attn", block.attention)
  |> flatten_norm(prefix <> ".norm1", block.norm1)
  |> flatten_norm(prefix <> ".norm2", block.norm2)
  |> flatten_ffn(prefix <> ".ffn", block.ffn)
}

fn flatten_decoder(
  base: Dict(String, Tensor),
  prefix: String,
  block: t.DecoderBlock,
) -> Dict(String, Tensor) {
  base
  |> flatten_mha(prefix <> ".self_attn", block.self_attention)
  |> flatten_mha(prefix <> ".cross_attn", block.cross_attention)
  |> flatten_norm(prefix <> ".norm1", block.norm1)
  |> flatten_norm(prefix <> ".norm2", block.norm2)
  |> flatten_norm(prefix <> ".norm3", block.norm3)
  |> flatten_ffn(prefix <> ".ffn", block.ffn)
}

// Build a Transformer with deterministic non-zero weights (record-update
// from the zero-init defaults) so the roundtrip actually exercises non-
// trivial tensors.
fn make_filled_transformer(embed_dim: Int, hidden_dim: Int) -> t.Transformer {
  let assert Ok(base) =
    nn_transformer.transformer_init(1, 1, embed_dim, 2, hidden_dim, ReluAct)

  // Replace every weight tensor with a deterministic ramp so equality
  // comparisons are meaningful.
  let assert [enc, ..] = base.encoder_blocks
  let assert [dec, ..] = base.decoder_blocks

  let enc_mha =
    EncoderBlock(
      attention: enc.attention
        |> with_mha_ramp(embed_dim, 0.0),
      norm1: enc.norm1
        |> with_norm_ramp(embed_dim, 100.0),
      norm2: enc.norm2
        |> with_norm_ramp(embed_dim, 200.0),
      ffn: enc.ffn
        |> with_ffn_ramp(embed_dim, hidden_dim, 300.0),
    )

  let dec_filled =
    DecoderBlock(
      self_attention: dec.self_attention
        |> with_mha_ramp(embed_dim, 400.0),
      cross_attention: dec.cross_attention
        |> with_mha_ramp(embed_dim, 500.0),
      norm1: dec.norm1
        |> with_norm_ramp(embed_dim, 600.0),
      norm2: dec.norm2
        |> with_norm_ramp(embed_dim, 700.0),
      norm3: dec.norm3
        |> with_norm_ramp(embed_dim, 800.0),
      ffn: dec.ffn
        |> with_ffn_ramp(embed_dim, hidden_dim, 900.0),
    )

  Transformer(..base, encoder_blocks: [enc_mha], decoder_blocks: [dec_filled])
}

fn with_mha_ramp(
  mha: t.MultiHeadAttention,
  embed_dim: Int,
  base: Float,
) -> t.MultiHeadAttention {
  MultiHeadAttention(
    ..mha,
    w_q: make_tensor([embed_dim, embed_dim], base),
    w_k: make_tensor([embed_dim, embed_dim], base +. 16.0),
    w_v: make_tensor([embed_dim, embed_dim], base +. 32.0),
    w_o: make_tensor([embed_dim, embed_dim], base +. 48.0),
    b_q: Some(make_tensor([embed_dim], base +. 64.0)),
    b_k: Some(make_tensor([embed_dim], base +. 65.0)),
    b_v: Some(make_tensor([embed_dim], base +. 66.0)),
    b_o: Some(make_tensor([embed_dim], base +. 67.0)),
  )
}

fn with_norm_ramp(
  ln: t.LayerNorm,
  num_features: Int,
  base: Float,
) -> t.LayerNorm {
  LayerNorm(
    ..ln,
    scale: make_tensor([num_features], base),
    bias: make_tensor([num_features], base +. 1.0),
  )
}

fn with_ffn_ramp(
  ff: t.FeedForward,
  embed_dim: Int,
  hidden_dim: Int,
  base: Float,
) -> t.FeedForward {
  FeedForward(
    ..ff,
    w1: make_tensor([embed_dim, hidden_dim], base),
    b1: make_tensor([hidden_dim], base +. 1.0),
    w2: make_tensor([hidden_dim, embed_dim], base +. 2.0),
    b2: make_tensor([embed_dim], base +. 3.0),
  )
}

pub fn transformer_roundtrip_test() {
  let embed_dim = 4
  let hidden_dim = 8
  let model = make_filled_transformer(embed_dim, hidden_dim)

  // Materialize every tensor under HF-style keys.
  let assert [enc, ..] = model.encoder_blocks
  let assert [dec, ..] = model.decoder_blocks
  let weights =
    dict.new()
    |> flatten_encoder("encoder.layers.0", enc)
    |> flatten_decoder("decoder.layers.0", dec)

  // Reload and compare element-by-element.
  let assert Ok(loaded) =
    t.load_transformer(weights, 1, 1, embed_dim, 2, hidden_dim, ReluAct)
  let assert [loaded_enc, ..] = loaded.encoder_blocks
  let assert [loaded_dec, ..] = loaded.decoder_blocks

  // Encoder MHA q_proj weight.
  numerics.lists_close(
    t.to_list(loaded_enc.attention.w_q),
    t.to_list(enc.attention.w_q),
    rtol,
    atol,
  )
  |> should.be_true

  // Encoder norm1 scale.
  numerics.lists_close(
    t.to_list(loaded_enc.norm1.scale),
    t.to_list(enc.norm1.scale),
    rtol,
    atol,
  )
  |> should.be_true

  // Encoder FFN linear2 bias.
  numerics.lists_close(
    t.to_list(loaded_enc.ffn.b2),
    t.to_list(enc.ffn.b2),
    rtol,
    atol,
  )
  |> should.be_true

  // Decoder cross-attention v_proj bias.
  let assert Some(loaded_bv) = loaded_dec.cross_attention.b_v
  let assert Some(src_bv) = dec.cross_attention.b_v
  numerics.lists_close(t.to_list(loaded_bv), t.to_list(src_bv), rtol, atol)
  |> should.be_true

  // Decoder norm3 bias.
  numerics.lists_close(
    t.to_list(loaded_dec.norm3.bias),
    t.to_list(dec.norm3.bias),
    rtol,
    atol,
  )
  |> should.be_true
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
