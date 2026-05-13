//// Basic text tokenizers for inference-time encoding with pre-trained
//// HuggingFace-style vocabularies.
////
//// This module is **encoding-only**: it does **not** train vocabularies or
//// BPE merge tables. The expected workflow is to load an existing tokenizer
//// produced by another tool (HuggingFace `tokenizers`, sentencepiece, etc.)
//// and use these functions to encode/decode text into integer id sequences
//// suitable for feeding into a tensor model.
////
//// Four tokenizer flavors are provided:
////
//// - `WhitespaceTokenizer` — split on whitespace and lookup.
//// - `CharTokenizer` — split into Unicode graphemes.
//// - `WordPieceTokenizer` — BERT-style greedy longest-match-first subwords
////   with `##` continuation prefix.
//// - `BpeTokenizer` — apply an ordered merge table to characters until no
////   more merges apply.
////
//// The companion helpers `ids_to_tensor`, `tensor_to_ids` and
//// `pad_or_truncate` bridge id sequences and `viva_tensor` tensors.

import gleam/dict.{type Dict}
import gleam/list
import gleam/string
import viva_tensor/tensor.{type Tensor, Tensor}

// =============================================================================
// Whitespace tokenizer
// =============================================================================

/// Token sequence produced by splitting on whitespace and lowercasing.
///
/// Stores both the forward (token -> id) and inverse (id -> token) maps to
/// keep encode/decode O(1) per lookup.
pub type WhitespaceTokenizer {
  WhitespaceTokenizer(
    vocab: Dict(String, Int),
    inverse_vocab: Dict(Int, String),
    unk_id: Int,
    pad_id: Int,
  )
}

/// Build a `WhitespaceTokenizer` from an ordered vocabulary list. The id of
/// each token is its position in the list. `unk_token` and `pad_token` must
/// already appear in `vocab`; otherwise their ids default to `0`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.whitespace_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "hello", "world"],
///   "[UNK]",
///   "[PAD]",
/// )
/// let _ = tok.whitespace_encode(t, "hello world")
/// ```
pub fn whitespace_tokenizer_from_vocab(
  vocab: List(String),
  unk_token: String,
  pad_token: String,
) -> WhitespaceTokenizer {
  let #(forward, inverse) = build_vocab(vocab)
  let unk_id = lookup_or_zero(forward, unk_token)
  let pad_id = lookup_or_zero(forward, pad_token)
  WhitespaceTokenizer(
    vocab: forward,
    inverse_vocab: inverse,
    unk_id: unk_id,
    pad_id: pad_id,
  )
}

/// Encode text by lowercasing, splitting on whitespace, and looking each
/// token up in the vocabulary. Unknown tokens map to `unk_id`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.whitespace_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "hello", "world"],
///   "[UNK]",
///   "[PAD]",
/// )
/// let _ = tok.whitespace_encode(t, "hello world")
/// ```
pub fn whitespace_encode(
  tokenizer: WhitespaceTokenizer,
  text: String,
) -> List(Int) {
  text
  |> string.lowercase
  |> split_whitespace
  |> list.map(fn(token) {
    case dict.get(tokenizer.vocab, token) {
      Ok(id) -> id
      Error(_) -> tokenizer.unk_id
    }
  })
}

/// Decode a sequence of ids back into a space-joined string. Missing ids are
/// rendered as the empty string.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.whitespace_tokenizer_from_vocab(
///   ["[PAD]", "[UNK]", "hello", "world"],
///   "[UNK]",
///   "[PAD]",
/// )
/// let _ = tok.whitespace_decode(t, [2, 3])
/// ```
pub fn whitespace_decode(
  tokenizer: WhitespaceTokenizer,
  ids: List(Int),
) -> String {
  ids
  |> list.map(fn(id) {
    case dict.get(tokenizer.inverse_vocab, id) {
      Ok(token) -> token
      Error(_) -> ""
    }
  })
  |> string.join(" ")
}

// =============================================================================
// Character-level tokenizer
// =============================================================================

/// Maps individual Unicode graphemes to ids.
pub type CharTokenizer {
  CharTokenizer(
    vocab: Dict(String, Int),
    inverse_vocab: Dict(Int, String),
    unk_id: Int,
  )
}

/// Build a character tokenizer from an alphabet. The id of each grapheme is
/// its position in the input list.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.char_tokenizer_from_alphabet(["?", "a", "b", "c"], "?")
/// let _ = tok.char_encode(t, "abc")
/// ```
pub fn char_tokenizer_from_alphabet(
  alphabet: List(String),
  unk_token: String,
) -> CharTokenizer {
  let #(forward, inverse) = build_vocab(alphabet)
  let unk_id = lookup_or_zero(forward, unk_token)
  CharTokenizer(vocab: forward, inverse_vocab: inverse, unk_id: unk_id)
}

/// Encode text by splitting it into Unicode graphemes and looking each up.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.char_tokenizer_from_alphabet(["?", "a", "b", "c"], "?")
/// let _ = tok.char_encode(t, "abc")
/// ```
pub fn char_encode(tokenizer: CharTokenizer, text: String) -> List(Int) {
  text
  |> string.to_graphemes
  |> list.map(fn(grapheme) {
    case dict.get(tokenizer.vocab, grapheme) {
      Ok(id) -> id
      Error(_) -> tokenizer.unk_id
    }
  })
}

/// Decode ids back into a single string by concatenating their graphemes.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.char_tokenizer_from_alphabet(["?", "a", "b", "c"], "?")
/// let _ = tok.char_decode(t, [1, 2, 3])
/// ```
pub fn char_decode(tokenizer: CharTokenizer, ids: List(Int)) -> String {
  ids
  |> list.map(fn(id) {
    case dict.get(tokenizer.inverse_vocab, id) {
      Ok(token) -> token
      Error(_) -> ""
    }
  })
  |> string.concat
}

// =============================================================================
// WordPiece tokenizer (BERT-style, encoding-only)
// =============================================================================

/// Simplified BERT-style WordPiece tokenizer. Continuation pieces use the
/// `##` prefix convention. Words longer than `max_input_chars_per_word`
/// graphemes are mapped to `unk_id`.
pub type WordPieceTokenizer {
  WordPieceTokenizer(
    vocab: Dict(String, Int),
    inverse_vocab: Dict(Int, String),
    unk_id: Int,
    cls_id: Int,
    sep_id: Int,
    pad_id: Int,
    max_input_chars_per_word: Int,
  )
}

/// Build a WordPiece tokenizer from an ordered vocabulary. `unk_token`,
/// `cls_token`, `sep_token` and `pad_token` must appear in `vocab`. Their
/// ids default to `0` when missing.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##world"]
/// let t = tok.word_piece_tokenizer_from_vocab(
///   vocab,
///   "[UNK]",
///   "[CLS]",
///   "[SEP]",
///   "[PAD]",
/// )
/// let _ = tok.word_piece_encode(t, "helloworld")
/// ```
pub fn word_piece_tokenizer_from_vocab(
  vocab: List(String),
  unk_token: String,
  cls_token: String,
  sep_token: String,
  pad_token: String,
) -> WordPieceTokenizer {
  let #(forward, inverse) = build_vocab(vocab)
  WordPieceTokenizer(
    vocab: forward,
    inverse_vocab: inverse,
    unk_id: lookup_or_zero(forward, unk_token),
    cls_id: lookup_or_zero(forward, cls_token),
    sep_id: lookup_or_zero(forward, sep_token),
    pad_id: lookup_or_zero(forward, pad_token),
    max_input_chars_per_word: 100,
  )
}

/// Encode text via greedy longest-match-first WordPiece. The output starts
/// with `cls_id`, ends with `sep_id`, and contains one id per subword piece.
/// Continuation pieces are expected to live in the vocabulary with a `##`
/// prefix (e.g. `"##ing"`).
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##world"]
/// let t = tok.word_piece_tokenizer_from_vocab(
///   vocab,
///   "[UNK]",
///   "[CLS]",
///   "[SEP]",
///   "[PAD]",
/// )
/// let _ = tok.word_piece_encode(t, "helloworld")
/// ```
pub fn word_piece_encode(
  tokenizer: WordPieceTokenizer,
  text: String,
) -> List(Int) {
  let words =
    text
    |> string.lowercase
    |> split_whitespace

  let body =
    list.flat_map(words, fn(word) { word_piece_encode_word(tokenizer, word) })

  list.flatten([[tokenizer.cls_id], body, [tokenizer.sep_id]])
}

/// Decode ids by stripping `##` prefixes and concatenating contiguous
/// continuation pieces back into words. Words are separated with a single
/// space.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "hello", "##world"]
/// let t = tok.word_piece_tokenizer_from_vocab(
///   vocab,
///   "[UNK]",
///   "[CLS]",
///   "[SEP]",
///   "[PAD]",
/// )
/// let _ = tok.word_piece_decode(t, [2, 4, 5, 3])
/// ```
pub fn word_piece_decode(
  tokenizer: WordPieceTokenizer,
  ids: List(Int),
) -> String {
  let tokens =
    ids
    |> list.map(fn(id) {
      case dict.get(tokenizer.inverse_vocab, id) {
        Ok(token) -> token
        Error(_) -> ""
      }
    })

  let special = [
    lookup_or_empty(tokenizer.inverse_vocab, tokenizer.cls_id),
    lookup_or_empty(tokenizer.inverse_vocab, tokenizer.sep_id),
    lookup_or_empty(tokenizer.inverse_vocab, tokenizer.pad_id),
  ]

  tokens
  |> list.filter(fn(token) { token != "" && !list.contains(special, token) })
  |> stitch_word_pieces
  |> list.reverse
  |> string.join(" ")
}

fn word_piece_encode_word(
  tokenizer: WordPieceTokenizer,
  word: String,
) -> List(Int) {
  case string.length(word) {
    0 -> []
    n if n > tokenizer.max_input_chars_per_word -> [tokenizer.unk_id]
    _ ->
      case word_piece_match(tokenizer, word, True, []) {
        Ok(ids) -> list.reverse(ids)
        Error(_) -> [tokenizer.unk_id]
      }
  }
}

fn word_piece_match(
  tokenizer: WordPieceTokenizer,
  remaining: String,
  is_start: Bool,
  acc: List(Int),
) -> Result(List(Int), Nil) {
  case remaining {
    "" -> Ok(acc)
    _ ->
      case longest_prefix_in_vocab(tokenizer.vocab, remaining, is_start) {
        Ok(#(id, rest)) -> word_piece_match(tokenizer, rest, False, [id, ..acc])
        Error(_) -> Error(Nil)
      }
  }
}

fn longest_prefix_in_vocab(
  vocab: Dict(String, Int),
  word: String,
  is_start: Bool,
) -> Result(#(Int, String), Nil) {
  let len = string.length(word)
  try_prefix(vocab, word, is_start, len)
}

fn try_prefix(
  vocab: Dict(String, Int),
  word: String,
  is_start: Bool,
  size: Int,
) -> Result(#(Int, String), Nil) {
  case size {
    0 -> Error(Nil)
    _ -> {
      let prefix = string.slice(word, 0, size)
      let candidate = case is_start {
        True -> prefix
        False -> "##" <> prefix
      }
      case dict.get(vocab, candidate) {
        Ok(id) -> {
          let rest = string.slice(word, size, string.length(word) - size)
          Ok(#(id, rest))
        }
        Error(_) -> try_prefix(vocab, word, is_start, size - 1)
      }
    }
  }
}

fn stitch_word_pieces(tokens: List(String)) -> List(String) {
  list.fold(tokens, [], fn(acc, token) {
    case string.starts_with(token, "##") {
      True -> {
        let suffix = string.slice(token, 2, string.length(token) - 2)
        case acc {
          [head, ..rest] -> [head <> suffix, ..rest]
          [] -> [suffix]
        }
      }
      False -> [token, ..acc]
    }
  })
}

// =============================================================================
// BPE tokenizer (encoding-only)
// =============================================================================

/// Byte-pair encoding tokenizer driven by an ordered merge table. Encoding
/// is performed against an already-trained vocabulary and merge list:
/// training is **not** supported.
pub type BpeTokenizer {
  BpeTokenizer(
    vocab: Dict(String, Int),
    inverse_vocab: Dict(Int, String),
    merges: List(#(String, String)),
    unk_id: Int,
  )
}

/// Build a BPE tokenizer from an ordered vocabulary and an ordered list of
/// merges. The first merge in the list has the highest priority.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "l", "o", "lo"],
///   [#("l", "o")],
///   "?",
/// )
/// let _ = tok.bpe_encode(t, "lo")
/// ```
pub fn bpe_tokenizer_from_vocab_and_merges(
  vocab: List(String),
  merges: List(#(String, String)),
  unk_token: String,
) -> BpeTokenizer {
  let #(forward, inverse) = build_vocab(vocab)
  BpeTokenizer(
    vocab: forward,
    inverse_vocab: inverse,
    merges: merges,
    unk_id: lookup_or_zero(forward, unk_token),
  )
}

/// Encode text by splitting it into characters and repeatedly applying the
/// highest-priority applicable merge from the merge table until no more
/// merges fire. Each surviving piece is then looked up in the vocab; missing
/// pieces map to `unk_id`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "l", "o", "lo"],
///   [#("l", "o")],
///   "?",
/// )
/// let _ = tok.bpe_encode(t, "lol")
/// ```
pub fn bpe_encode(tokenizer: BpeTokenizer, text: String) -> List(Int) {
  let pieces =
    text
    |> string.to_graphemes
    |> apply_bpe_merges(tokenizer.merges)

  list.map(pieces, fn(piece) {
    case dict.get(tokenizer.vocab, piece) {
      Ok(id) -> id
      Error(_) -> tokenizer.unk_id
    }
  })
}

/// Decode ids by reversing each lookup and concatenating the resulting
/// pieces.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let t = tok.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "l", "o", "lo"],
///   [#("l", "o")],
///   "?",
/// )
/// let _ = tok.bpe_decode(t, [3, 1])
/// ```
pub fn bpe_decode(tokenizer: BpeTokenizer, ids: List(Int)) -> String {
  ids
  |> list.map(fn(id) {
    case dict.get(tokenizer.inverse_vocab, id) {
      Ok(token) -> token
      Error(_) -> ""
    }
  })
  |> string.concat
}

fn apply_bpe_merges(
  pieces: List(String),
  merges: List(#(String, String)),
) -> List(String) {
  case first_applicable_merge(pieces, merges) {
    Error(_) -> pieces
    Ok(#(left, right)) -> {
      let merged = merge_pair(pieces, left, right)
      apply_bpe_merges(merged, merges)
    }
  }
}

fn first_applicable_merge(
  pieces: List(String),
  merges: List(#(String, String)),
) -> Result(#(String, String), Nil) {
  case merges {
    [] -> Error(Nil)
    [#(left, right), ..rest] ->
      case has_adjacent_pair(pieces, left, right) {
        True -> Ok(#(left, right))
        False -> first_applicable_merge(pieces, rest)
      }
  }
}

fn has_adjacent_pair(
  pieces: List(String),
  left: String,
  right: String,
) -> Bool {
  case pieces {
    [] -> False
    [_] -> False
    [a, b, ..rest] ->
      case a == left && b == right {
        True -> True
        False -> has_adjacent_pair([b, ..rest], left, right)
      }
  }
}

fn merge_pair(
  pieces: List(String),
  left: String,
  right: String,
) -> List(String) {
  case pieces {
    [] -> []
    [a, b, ..rest] ->
      case a == left && b == right {
        True -> [a <> b, ..merge_pair(rest, left, right)]
        False -> [a, ..merge_pair([b, ..rest], left, right)]
      }
    [single] -> [single]
  }
}

// =============================================================================
// Tensor / id helpers
// =============================================================================

/// Convert a list of integer ids into a `[seq_len]` shaped tensor. Each id
/// is stored as a float so it can flow through the dense tensor pipeline.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let _ = tok.ids_to_tensor([1, 2, 3])
/// ```
pub fn ids_to_tensor(ids: List(Int)) -> Tensor {
  let data = list.map(ids, int_to_float)
  Tensor(data: data, shape: [list.length(data)])
}

/// Convert a `[seq_len]` integer-valued tensor back into a `List(Int)`.
/// Non-integer floats are truncated toward zero.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let _ = tok.tensor_to_ids(tok.ids_to_tensor([1, 2, 3]))
/// ```
pub fn tensor_to_ids(tensor: Tensor) -> List(Int) {
  tensor
  |> tensor.to_list
  |> list.map(float_to_int)
}

/// Pad or truncate a list of ids to exactly `max_length`. Shorter lists are
/// right-padded with `pad_id`; longer lists are truncated to the first
/// `max_length` elements.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/tokenizer as tok
/// let _ = tok.pad_or_truncate([1, 2], 4, 0)
/// ```
pub fn pad_or_truncate(
  ids: List(Int),
  max_length: Int,
  pad_id: Int,
) -> List(Int) {
  let len = list.length(ids)
  case len >= max_length {
    True -> list.take(ids, max_length)
    False -> list.append(ids, list.repeat(pad_id, max_length - len))
  }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn build_vocab(
  tokens: List(String),
) -> #(Dict(String, Int), Dict(Int, String)) {
  let indexed = list.index_map(tokens, fn(token, index) { #(token, index) })
  let forward = dict.from_list(indexed)
  let inverse =
    indexed
    |> list.map(fn(pair) {
      let #(token, index) = pair
      #(index, token)
    })
    |> dict.from_list
  #(forward, inverse)
}

fn lookup_or_zero(vocab: Dict(String, Int), token: String) -> Int {
  case dict.get(vocab, token) {
    Ok(id) -> id
    Error(_) -> 0
  }
}

fn lookup_or_empty(inverse: Dict(Int, String), id: Int) -> String {
  case dict.get(inverse, id) {
    Ok(token) -> token
    Error(_) -> ""
  }
}

fn split_whitespace(text: String) -> List(String) {
  text
  |> string.replace("\t", " ")
  |> string.replace("\n", " ")
  |> string.replace("\r", " ")
  |> string.split(" ")
  |> list.filter(fn(piece) { piece != "" })
}

@external(erlang, "erlang", "float")
fn int_to_float(value: Int) -> Float

@external(erlang, "erlang", "trunc")
fn float_to_int(value: Float) -> Int
