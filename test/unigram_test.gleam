//// Tests for `viva_tensor/text/unigram` (Unigram LM + SentencePiece wrapper).
////
//// Pieces use the SentencePiece `▁` (U+2581) "lower one eighth block" prefix
//// to mark word boundaries, so the test vocabularies include both `▁`-
//// prefixed and bare variants where relevant.

import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor as t

pub fn main() -> Nil {
  gleeunit.main()
}

const unk: String = "<unk>"

const bos: String = "<s>"

const eos: String = "</s>"

// Helpers --------------------------------------------------------------------

fn special_pieces() -> List(#(String, Float)) {
  [#(unk, -100.0), #(bos, -100.0), #(eos, -100.0)]
}

// =============================================================================
// Unigram
// =============================================================================

pub fn unigram_simple_roundtrip_test() {
  let pieces =
    list.append(special_pieces(), [
      #("▁hello", -1.0),
      #("▁world", -1.0),
    ])
  let tok = t.unigram_tokenizer_from_pieces(pieces, unk, bos, eos)

  let ids = t.unigram_encode(tok, "hello world")
  // First and last must be bos/eos.
  list.first(ids) |> should.equal(Ok(1))
  // Decode strips bos/eos and the leading `▁`.
  t.unigram_decode(tok, ids) |> should.equal("hello world")
}

pub fn unigram_prefers_long_pieces_test() {
  // Viterbi must pick the single piece `▁hello` (score -1.0) over the
  // `▁he`+`llo` split (score -4.0); a left-to-right greedy with the bare
  // `he`/`llo` would split.
  let pieces =
    list.append(special_pieces(), [
      #("▁hello", -1.0),
      #("▁he", -2.0),
      #("llo", -2.0),
    ])
  let tok = t.unigram_tokenizer_from_pieces(pieces, unk, bos, eos)

  let ids = t.unigram_encode(tok, "hello")
  // bos + 1 piece + eos == 3 ids.
  list.length(ids) |> should.equal(3)
}

pub fn unigram_unk_test() {
  // Vocab covers nothing in the input: every grapheme falls back to unk.
  let pieces = list.append(special_pieces(), [#("▁hello", -1.0)])
  let tok = t.unigram_tokenizer_from_pieces(pieces, unk, bos, eos)

  let ids = t.unigram_encode(tok, "zzz")
  // bos at the head, eos at the tail; everything in the middle should be
  // unk_id (== 0, the position of `<unk>` in the pieces list).
  let unk_id = 0
  let bos_id = 1
  let eos_id = 2
  list.first(ids) |> should.equal(Ok(bos_id))
  list.last(ids) |> should.equal(Ok(eos_id))
  let middle = case ids {
    [_, ..rest] -> list.take(rest, list.length(rest) - 1)
    [] -> []
  }
  list.all(middle, fn(id) { id == unk_id }) |> should.equal(True)
}

pub fn unigram_bos_eos_test() {
  let pieces = list.append(special_pieces(), [#("▁hi", -1.0)])
  let tok = t.unigram_tokenizer_from_pieces(pieces, unk, bos, eos)

  let ids = t.unigram_encode(tok, "hi")
  list.first(ids) |> should.equal(Ok(1))
  // 1 == bos_id (position of `<s>`); 2 == eos_id (position of `</s>`).
  list.last(ids) |> should.equal(Ok(2))
}

pub fn unigram_underscore_prefix_test() {
  // After normalization, "hello world" becomes "▁hello▁world", so the second
  // word's piece must start with the underscore marker.
  let pieces =
    list.append(special_pieces(), [
      #("▁hello", -1.0),
      #("▁world", -1.0),
    ])
  let tok = t.unigram_tokenizer_from_pieces(pieces, unk, bos, eos)

  let ids = t.unigram_encode(tok, "hello world")
  // Expected: [bos, ▁hello, ▁world, eos] -> 4 ids.
  list.length(ids) |> should.equal(4)
  // Decoded form re-inserts the space between words.
  t.unigram_decode(tok, ids) |> should.equal("hello world")
}

// =============================================================================
// SentencePiece wrapper
// =============================================================================

pub fn sentence_piece_unigram_wrapper_test() {
  let pieces = list.append(special_pieces(), [#("▁hi", -1.0)])
  let inner = t.unigram_tokenizer_from_pieces(pieces, unk, bos, eos)
  let wrapper = t.sentence_piece_unigram(inner)

  // Wrapper output equals the inner unigram's output.
  let direct = t.unigram_encode(inner, "hi")
  let wrapped = t.sentence_piece_encode(wrapper, "hi")
  wrapped |> should.equal(direct)

  let decoded = t.sentence_piece_decode(wrapper, wrapped)
  decoded |> should.equal("hi")
}

pub fn sentence_piece_bpe_wrapper_test() {
  // Vocab: ?=0, ▁=1, h=2, i=3, ▁h=4, ▁hi=5.
  let vocab = ["?", "▁", "h", "i", "▁h", "▁hi"]
  let merges = [#("▁", "h"), #("▁h", "i")]
  let inner = t.bpe_tokenizer_from_vocab_and_merges(vocab, merges, "?")
  let wrapper = t.sentence_piece_bpe(inner)

  let ids = t.sentence_piece_encode(wrapper, "hi")
  // Pipeline: "hi" -> "▁hi" -> graphemes [▁,h,i] -> merge ▁+h -> [▁h,i]
  //          -> merge ▁h+i -> [▁hi] -> [5].
  ids |> should.equal([5])

  t.sentence_piece_decode(wrapper, ids) |> should.equal("hi")
}
