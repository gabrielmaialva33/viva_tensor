//// Regression tests for the LLM BPE / SentencePiece tokenizer FFI
//// (`viva_tensor_tokenizer_ffi`). Goldens are validated byte-for-byte against
//// HuggingFace `tokenizers`. Fixtures live under tmp/ (gitignored), so each
//// test skips cleanly when the model checkpoint isn't present.

import gleam/dynamic.{type Dynamic}
import gleam/io
import gleeunit/should

const llama3_tok = "tmp/llama32_1b/tokenizer.json"

const tinyllama_tok = "tmp/tinyllama/tokenizer.json"

// Byte-level BPE (Llama-3): encode must match HF tokenizers exactly.
pub fn llama3_byte_level_encode_test() {
  case path_exists(llama3_tok) {
    False -> skip(llama3_tok)
    True -> {
      let assert Ok(state) = tok_load(llama3_tok)
      tok_encode(state, "The capital of France is")
      |> should.equal([791, 6864, 315, 9822, 374])
    }
  }
}

// Special tokens embedded by a chat template must be matched atomically,
// not shattered into bytes by BPE.
pub fn llama3_chat_template_special_tokens_test() {
  case path_exists(llama3_tok) {
    False -> skip(llama3_tok)
    True -> {
      let assert Ok(state) = tok_load(llama3_tok)
      let tpl =
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
      tok_encode(state, tpl)
      |> should.equal([
        128_000, 128_006, 882, 128_007, 271, 3923, 374, 279, 6864, 315, 9822, 30,
        128_009, 128_006, 78_191, 128_007, 271,
      ])
    }
  }
}

// BOS/EOS must resolve to the Llama-3 ids, not the legacy <s>/</s> defaults.
pub fn llama3_bos_eos_resolution_test() {
  case path_exists(llama3_tok) {
    False -> skip(llama3_tok)
    True -> {
      let assert Ok(state) = tok_load(llama3_tok)
      tok_bos(state) |> should.equal(128_000)
      tok_eos(state) |> should.equal(128_009)
    }
  }
}

pub fn llama3_encode_decode_roundtrip_test() {
  case path_exists(llama3_tok) {
    False -> skip(llama3_tok)
    True -> {
      let assert Ok(state) = tok_load(llama3_tok)
      let text = "Once upon a time, in a kingdom of code,"
      tok_decode(state, tok_encode(state, text)) |> should.equal(text)
    }
  }
}

// The SentencePiece (TinyLlama / Llama-2) path must stay byte-identical
// after the byte-level additions.
pub fn tinyllama_sentencepiece_regression_test() {
  case path_exists(tinyllama_tok) {
    False -> skip(tinyllama_tok)
    True -> {
      let assert Ok(state) = tok_load(tinyllama_tok)
      tok_encode(state, "The capital of France is")
      |> should.equal([450, 7483, 310, 3444, 338])
      tok_bos(state) |> should.equal(1)
      tok_eos(state) |> should.equal(2)
    }
  }
}

fn skip(path: String) -> Nil {
  io.println(path <> " fixture not found; skipping tokenizer FFI test")
}

@external(erlang, "viva_tensor_llm", "path_exists")
fn path_exists(path: String) -> Bool

@external(erlang, "viva_tensor_tokenizer_ffi", "load")
fn tok_load(path: String) -> Result(Dynamic, Dynamic)

@external(erlang, "viva_tensor_tokenizer_ffi", "encode")
fn tok_encode(state: Dynamic, text: String) -> List(Int)

@external(erlang, "viva_tensor_tokenizer_ffi", "decode")
fn tok_decode(state: Dynamic, ids: List(Int)) -> String

@external(erlang, "viva_tensor_tokenizer_ffi", "bos_id")
fn tok_bos(state: Dynamic) -> Int

@external(erlang, "viva_tensor_tokenizer_ffi", "eos_id")
fn tok_eos(state: Dynamic) -> Int
