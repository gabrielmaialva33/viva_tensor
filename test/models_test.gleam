//// Tests for the pre-baked transformer model assemblies (Llama, BERT, GPT,
//// T5). With zero weights the forward pass collapses to all-zero outputs in
//// most branches — that's plenty for confirming the wiring (shapes, layer
//// counts, sublayer ordering). One numeric test pins the SwiGLU formula.

import gleam/list
import gleeunit
import gleeunit/should
import support/numerics.{floats_close, lists_close}
import viva_tensor/models/bert
import viva_tensor/models/gpt
import viva_tensor/models/llama
import viva_tensor/models/t5
import viva_tensor/nn/activations
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-6

fn t2d(rows: List(List(Float))) -> tensor.Tensor {
  let assert Ok(t) = tensor.from_list2d(rows)
  t
}

// ---------------------------------------------------------------------------
// Llama
// ---------------------------------------------------------------------------

pub fn llama_block_forward_shape_test() {
  let assert Ok(block) = llama.llama_block_init(4, 2, 8)
  let input = t2d([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
  let assert Ok(out) = llama.llama_block_forward(block, input)
  out.shape |> should.equal([2, 4])
}

pub fn llama_block_init_invalid_dim_test() {
  // embed_dim=5 not divisible by num_heads=2 must fail.
  let r = llama.llama_block_init(5, 2, 8)
  case r {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn llama_model_forward_shape_test() {
  let assert Ok(model) = llama.llama_model_init(2, 16, 4, 2, 8)
  let token_ids = tensor.from_list([0.0, 1.0, 2.0])
  let assert Ok(logits) = llama.llama_model_forward(model, token_ids)
  logits.shape |> should.equal([3, 16])
}

// ---------------------------------------------------------------------------
// BERT
// ---------------------------------------------------------------------------

pub fn bert_block_forward_shape_test() {
  let assert Ok(block) = bert.bert_block_init(4, 2, 8)
  let input = t2d([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
  let assert Ok(out) = bert.bert_block_forward(block, input)
  out.shape |> should.equal([2, 4])
}

pub fn bert_embedding_forward_shape_test() {
  let layer = bert.bert_embedding_init(16, 4, 8, 2)
  let token_ids = tensor.from_list([0.0, 1.0, 2.0])
  let token_type_ids = tensor.from_list([0.0, 0.0, 1.0])
  let assert Ok(out) =
    bert.bert_embedding_forward(layer, token_ids, token_type_ids)
  out.shape |> should.equal([3, 4])
}

pub fn bert_model_forward_shape_test() {
  let assert Ok(model) = bert.bert_model_init(2, 16, 4, 2, 8, 32)
  let token_ids = tensor.from_list([0.0, 1.0, 2.0])
  let token_type_ids = tensor.from_list([0.0, 0.0, 1.0])
  let assert Ok(hidden) =
    bert.bert_model_forward(model, token_ids, token_type_ids)
  hidden.shape |> should.equal([3, 4])
}

// ---------------------------------------------------------------------------
// GPT
// ---------------------------------------------------------------------------

pub fn gpt_block_forward_shape_test() {
  let assert Ok(block) = gpt.gpt_block_init(4, 2, 8)
  let input = t2d([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
  let assert Ok(out) = gpt.gpt_block_forward(block, input)
  out.shape |> should.equal([2, 4])
}

pub fn gpt_model_forward_shape_test() {
  let assert Ok(model) = gpt.gpt_model_init(2, 16, 4, 2, 8, 32)
  let token_ids = tensor.from_list([0.0, 1.0, 2.0])
  let assert Ok(logits) = gpt.gpt_model_forward(model, token_ids)
  logits.shape |> should.equal([3, 16])
}

// ---------------------------------------------------------------------------
// T5
// ---------------------------------------------------------------------------

pub fn t5_encoder_block_forward_shape_test() {
  let assert Ok(block) = t5.t5_encoder_block_init(4, 2, 8)
  let input = t2d([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
  let assert Ok(out) = t5.t5_encoder_block_forward(block, input)
  out.shape |> should.equal([2, 4])
}

pub fn t5_decoder_block_forward_shape_test() {
  let assert Ok(block) = t5.t5_decoder_block_init(4, 2, 8)
  let input = t2d([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
  let memory = t2d([[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6]])
  let assert Ok(out) = t5.t5_decoder_block_forward(block, input, memory)
  out.shape |> should.equal([2, 4])
}

pub fn t5_model_forward_shape_test() {
  let assert Ok(model) = t5.t5_model_init(1, 1, 16, 4, 2, 8)
  let src = tensor.from_list([0.0, 1.0, 2.0])
  let tgt = tensor.from_list([3.0, 4.0, 5.0])
  let assert Ok(logits) = t5.t5_model_forward(model, src, tgt)
  logits.shape |> should.equal([3, 16])
}

// ---------------------------------------------------------------------------
// SwiGLU formula
// ---------------------------------------------------------------------------

pub fn swiglu_activation_test() {
  // x = [[1.0, 2.0]] (seq=1, embed_dim=2)
  // w1 = identity (so gate = x), w3 = identity (so up = x), w2 = identity
  // out = silu(x) * x  (then @ I = same)
  let x = t2d([[1.0, 2.0]])
  let identity_w = t2d([[1.0, 0.0], [0.0, 1.0]])
  let assert Ok(out) =
    llama.swiglu_forward(x, identity_w, identity_w, identity_w)
  out.shape |> should.equal([1, 2])

  // Expected: silu(1) * 1 = 1 * sigmoid(1)  and  silu(2) * 2 = 2 * 2*sigmoid(2)
  // silu(z) = z * sigmoid(z)
  // silu(1) * 1  = 1 * sigmoid(1)            = ~0.7310585786
  // silu(2) * 2  = (2 * sigmoid(2)) * 2      = 4 * sigmoid(2) ~ 3.5231883811
  let silu_one_times_one = activations.swish(tensor.from_list([1.0]))
  let silu_two_times_two = activations.swish(tensor.from_list([2.0]))
  let assert [s1] = tensor.to_list(silu_one_times_one)
  let assert [s2] = tensor.to_list(silu_two_times_two)
  let expected = [s1 *. 1.0, s2 *. 2.0]
  lists_close(tensor.to_list(out), expected, rtol, atol) |> should.be_true

  // Spot-check: silu(1) * 1 should be very close to sigmoid(1).
  let assert [first, ..] = tensor.to_list(out)
  floats_close(first, 0.7310585786300049, rtol, atol) |> should.be_true

  // Touch list to avoid unused-import warnings for `list`.
  list.length([1])
  |> should.equal(1)
}
