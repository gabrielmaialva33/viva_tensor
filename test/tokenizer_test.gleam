//// Tests for `viva_tensor/text/tokenizer`.

import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// Whitespace tokenizer
// =============================================================================

pub fn whitespace_encode_decode_roundtrip_test() {
  let vocab = ["[PAD]", "[UNK]", "hello", "world", "foo"]
  let tok = t.whitespace_tokenizer_from_vocab(vocab, "[UNK]", "[PAD]")

  let ids = t.whitespace_encode(tok, "Hello world")
  ids |> should.equal([2, 3])
  t.whitespace_decode(tok, ids) |> should.equal("hello world")
}

pub fn whitespace_unk_test() {
  let vocab = ["[PAD]", "[UNK]", "hello"]
  let tok = t.whitespace_tokenizer_from_vocab(vocab, "[UNK]", "[PAD]")

  let ids = t.whitespace_encode(tok, "hello mystery")
  ids |> should.equal([2, 1])
}

// =============================================================================
// Character tokenizer
// =============================================================================

pub fn char_encode_test() {
  let tok = t.char_tokenizer_from_alphabet(["?", "a", "b", "c"], "?")
  t.char_encode(tok, "abc") |> should.equal([1, 2, 3])
  t.char_encode(tok, "az") |> should.equal([1, 0])
}

pub fn char_decode_test() {
  let tok = t.char_tokenizer_from_alphabet(["?", "a", "b", "c"], "?")
  t.char_decode(tok, [1, 2, 3]) |> should.equal("abc")
}

// =============================================================================
// WordPiece tokenizer
// =============================================================================

pub fn word_piece_basic_test() {
  let vocab = ["hello", "##world", "[CLS]", "[SEP]", "[UNK]", "[PAD]"]
  let tok =
    t.word_piece_tokenizer_from_vocab(vocab, "[UNK]", "[CLS]", "[SEP]", "[PAD]")

  // [CLS]=2, hello=0, ##world=1, [SEP]=3
  t.word_piece_encode(tok, "helloworld") |> should.equal([2, 0, 1, 3])

  t.word_piece_decode(tok, [2, 0, 1, 3]) |> should.equal("helloworld")
}

pub fn word_piece_unk_test() {
  let vocab = ["hello", "##world", "[CLS]", "[SEP]", "[UNK]", "[PAD]"]
  let tok =
    t.word_piece_tokenizer_from_vocab(vocab, "[UNK]", "[CLS]", "[SEP]", "[PAD]")

  // "zzz" has no prefix in vocab -> falls back to [UNK]=4
  t.word_piece_encode(tok, "zzz") |> should.equal([2, 4, 3])
}

// =============================================================================
// BPE tokenizer
// =============================================================================

pub fn bpe_simple_merges_test() {
  // ids: ?=0, l=1, o=2, lo=3, lol=4
  let vocab = ["?", "l", "o", "lo", "lol"]
  let merges = [#("l", "o")]
  let tok = t.bpe_tokenizer_from_vocab_and_merges(vocab, merges, "?")

  // "lol" -> ["l","o","l"] -> merge l+o -> ["lo","l"] -> [3, 1]
  t.bpe_encode(tok, "lol") |> should.equal([3, 1])
}

pub fn bpe_no_merge_test() {
  let vocab = ["?", "a", "b", "c"]
  let merges = [#("x", "y")]
  let tok = t.bpe_tokenizer_from_vocab_and_merges(vocab, merges, "?")

  // No merges apply -> falls through to character pieces.
  t.bpe_encode(tok, "abc") |> should.equal([1, 2, 3])
}

// =============================================================================
// Tensor / id helpers
// =============================================================================

pub fn ids_to_tensor_test() {
  let ids_tensor = t.ids_to_tensor([5, 10, 15])
  tensor.shape(ids_tensor) |> should.equal([3])
  tensor.to_list(ids_tensor) |> should.equal([5.0, 10.0, 15.0])
  t.tensor_to_ids(ids_tensor) |> should.equal([5, 10, 15])
}

pub fn pad_or_truncate_pad_test() {
  t.pad_or_truncate([1, 2], 5, 0) |> should.equal([1, 2, 0, 0, 0])
}

pub fn pad_or_truncate_truncate_test() {
  t.pad_or_truncate([1, 2, 3, 4, 5, 6], 3, 0) |> should.equal([1, 2, 3])
}
