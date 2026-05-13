import gleam/list
import gleeunit
import gleeunit/should
import support/numerics.{lists_close}
import viva_tensor/nn/transformer.{
  type EncoderBlock, type Transformer, GeluAct, ReluAct,
}
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-6

fn t2d(rows: List(List(Float))) -> tensor.Tensor {
  let assert Ok(t) = tensor.from_list2d(rows)
  t
}

fn zeros_2d(rows: Int, cols: Int) -> tensor.Tensor {
  tensor.zeros([rows, cols])
}

// ---------------------------------------------------------------------------
// FeedForward
// ---------------------------------------------------------------------------

pub fn feed_forward_init_test() {
  let ff = transformer.feed_forward_init(4, 8, GeluAct)
  ff.w1.shape |> should.equal([4, 8])
  ff.b1.shape |> should.equal([8])
  ff.w2.shape |> should.equal([8, 4])
  ff.b2.shape |> should.equal([4])
  ff.activation |> should.equal(GeluAct)
}

pub fn feed_forward_forward_zero_test() {
  // Zero weights and zero biases — output must be all zeros regardless of
  // input.
  let ff = transformer.feed_forward_init(4, 8, ReluAct)
  let input = t2d([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
  let assert Ok(out) = transformer.feed_forward_forward(ff, input)
  out.shape |> should.equal([2, 4])
  lists_close(tensor.to_list(out), list.repeat(0.0, 8), rtol, atol)
  |> should.be_true
}

// ---------------------------------------------------------------------------
// EncoderBlock
// ---------------------------------------------------------------------------

pub fn encoder_block_init_test() {
  let assert Ok(block) = transformer.encoder_block_init(8, 2, 16, GeluAct)
  let _: EncoderBlock = block
  block.attention.num_heads |> should.equal(2)
  block.attention.embed_dim |> should.equal(8)
  block.ffn.w1.shape |> should.equal([8, 16])
  block.ffn.w2.shape |> should.equal([16, 8])
  block.norm1.scale.shape |> should.equal([8])
  block.norm2.scale.shape |> should.equal([8])
}

pub fn encoder_block_forward_shape_test() {
  let assert Ok(block) = transformer.encoder_block_init(8, 2, 16, GeluAct)
  let input = zeros_2d(4, 8)
  let assert Ok(out) = transformer.encoder_block_forward(block, input, False)
  out.shape |> should.equal([4, 8])
}

// ---------------------------------------------------------------------------
// DecoderBlock
// ---------------------------------------------------------------------------

pub fn decoder_block_forward_shape_test() {
  let assert Ok(block) = transformer.decoder_block_init(8, 2, 16, GeluAct)
  let input = zeros_2d(4, 8)
  let memory = zeros_2d(5, 8)
  let assert Ok(out) = transformer.decoder_block_forward(block, input, memory)
  out.shape |> should.equal([4, 8])
}

// ---------------------------------------------------------------------------
// Transformer
// ---------------------------------------------------------------------------

pub fn transformer_init_test() {
  let assert Ok(model) = transformer.transformer_init(3, 2, 8, 2, 16, GeluAct)
  let _: Transformer = model
  model.num_encoder_layers |> should.equal(3)
  model.num_decoder_layers |> should.equal(2)
  list.length(model.encoder_blocks) |> should.equal(3)
  list.length(model.decoder_blocks) |> should.equal(2)
}

pub fn transformer_forward_shape_test() {
  let assert Ok(model) = transformer.transformer_init(2, 2, 8, 2, 16, GeluAct)
  let src = zeros_2d(3, 8)
  let tgt = zeros_2d(2, 8)
  let assert Ok(out) = transformer.transformer_forward(model, src, tgt)
  out.shape |> should.equal([2, 8])
}

pub fn transformer_init_invalid_dim_test() {
  // embed_dim=10, num_heads=3 → 10 % 3 != 0 → InvalidShape.
  let result = transformer.transformer_init(2, 2, 10, 3, 16, ReluAct)
  case result {
    Error(_) -> True
    Ok(_) -> False
  }
  |> should.be_true
}
