//// Unigram language-model tokenizer (SentencePiece / T5 style) and a thin
//// SentencePiece wrapper for both Unigram and BPE flavors.
////
//// This module is **inference-only**. The SentencePiece training loop (an
//// EM iteration that greedily prunes pieces from a seed lattice) is **out of
//// scope** — the expected workflow is to load a vocabulary produced by the
//// upstream `sentencepiece` binary (or HuggingFace `tokenizers`) and to call
//// the encode/decode helpers here at runtime.
////
//// Encoding distinction vs `viva_tensor/text/tokenizer`:
////
//// - `WordPieceTokenizer.encode` is **greedy** longest-match-first. It cannot
////   recover from a wrong early choice.
//// - `unigram_encode` uses **Viterbi** dynamic programming over the lattice
////   of all in-vocab subword pieces, maximizing the sum of per-piece
////   log-probabilities. Given the pieces `[("hello", -1.0), ("he", -2.0),
////   ("llo", -2.0)]`, Viterbi picks the single `hello` (score -1.0) over the
////   `he`+`llo` split (score -4.0); a greedy encoder splitting left-to-right
////   might pick the latter depending on tie-breaking.
////
//// The cost of Viterbi here is `O(N * max_piece_length)` dict lookups, where
//// `N` is the grapheme length of the input and `max_piece_length` is the
//// longest piece in the vocabulary. A prefix-trie implementation would have
//// the same asymptotic cost but with smaller constants and an early-exit on
//// missing prefixes; we keep the dict-based version for simplicity since
//// gleam_stdlib's `Dict` is already used by the surrounding tokenizers.

import gleam/dict.{type Dict}
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/string
import viva_tensor/text/tokenizer as text_tokenizer

/// SentencePiece "lower one eighth block" (U+2581) prefix used to mark word
/// boundaries — a leading whitespace in the input is rewritten to this
/// character, and any internal whitespace is replaced with it too. The exact
/// character is hard-coded here so the constant survives source re-encoding.
const sp_underscore: String = "▁"

// =============================================================================
// Unigram tokenizer
// =============================================================================

/// Unigram LM tokenizer (SentencePiece/T5 style).
///
/// Holds:
/// - `vocab`        — token to id.
/// - `inverse_vocab`— id to token (for decode).
/// - `scores`       — token to log-probability. Pieces in `vocab` but absent
///                    from `scores` are ignored by Viterbi.
/// - `unk_id`       — id used when no in-vocab piece covers a grapheme.
/// - `bos_id` / `eos_id` — sentinel ids prepended/appended by encode.
pub type UnigramTokenizer {
  UnigramTokenizer(
    /// Token to id.
    vocab: Dict(String, Int),
    /// id to token (for decoding).
    inverse_vocab: Dict(Int, String),
    /// Token to log-probability (higher = more likely; used for Viterbi).
    /// Pieces in `vocab` not present in `scores` are ignored.
    scores: Dict(String, Float),
    unk_id: Int,
    bos_id: Int,
    eos_id: Int,
  )
}

/// Build a `UnigramTokenizer` from a flat list of `(token, log_prob)` pairs.
///
/// The id of each piece is its position in the list. `unk_token`, `bos_token`
/// and `eos_token` must already appear in `pieces`; their ids default to `0`
/// when missing.
///
/// The score of every piece (including `unk_token`/`bos_token`/`eos_token`)
/// is recorded; in practice you should give the specials a very negative
/// score so Viterbi never picks them on its own.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/unigram
/// let _ = unigram.unigram_tokenizer_from_pieces(
///   [#("<unk>", -100.0), #("<s>", -100.0), #("</s>", -100.0),
///    #("hello", -1.0)],
///   "<unk>", "<s>", "</s>",
/// )
/// ```
pub fn unigram_tokenizer_from_pieces(
  pieces: List(#(String, Float)),
  unk_token: String,
  bos_token: String,
  eos_token: String,
) -> UnigramTokenizer {
  let tokens = list.map(pieces, fn(pair) { pair.0 })
  let #(forward, inverse) = build_vocab(tokens)
  let scores = dict.from_list(pieces)
  UnigramTokenizer(
    vocab: forward,
    inverse_vocab: inverse,
    scores: scores,
    unk_id: lookup_or_zero(forward, unk_token),
    bos_id: lookup_or_zero(forward, bos_token),
    eos_id: lookup_or_zero(forward, eos_token),
  )
}

/// Encode `text` into a list of piece ids using **Viterbi** dynamic
/// programming (max-sum of `log P(piece)`).
///
/// Pipeline:
/// 1. Replace ASCII space with the SentencePiece `▁` marker (U+2581) and
///    prefix a single `▁` to mark the start of the first word.
/// 2. Build `dp[i] = max over (piece, dp[i - len(piece)] + score(piece))`
///    for `i` from 0 to grapheme-count.
/// 3. If no piece covers a position, fall back to a single-grapheme `unk_id`
///    with a heavily-penalized score.
/// 4. Backtrack from `dp[N]` to recover the segmentation, map each piece to
///    its id, then wrap the result with `bos_id` and `eos_id`.
///
/// Distinct from a greedy left-to-right encoder: Viterbi can recover from a
/// locally-optimal but globally-bad first choice.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/unigram
/// let tok = unigram.unigram_tokenizer_from_pieces(
///   [#("<unk>", -100.0), #("<s>", -100.0), #("</s>", -100.0),
///    #("▁hello", -1.0), #("hello", -1.0)],
///   "<unk>", "<s>", "</s>",
/// )
/// let _ = unigram.unigram_encode(tok, "hello")
/// ```
pub fn unigram_encode(tokenizer: UnigramTokenizer, text: String) -> List(Int) {
  let normalized = normalize_sp(text)
  let graphemes = string.to_graphemes(normalized)
  let pieces = viterbi_segment(tokenizer, graphemes)
  let ids =
    list.map(pieces, fn(piece) {
      case dict.get(tokenizer.vocab, piece) {
        Ok(id) -> id
        Error(_) -> tokenizer.unk_id
      }
    })
  [tokenizer.bos_id, ..ids] |> list.append([tokenizer.eos_id])
}

/// Decode a list of ids back into a string.
///
/// Looks up each id in `inverse_vocab`, drops the `bos_id` / `eos_id`
/// markers, concatenates the surviving pieces, replaces `▁` with a regular
/// ASCII space, and strips a single leading space.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/unigram
/// let tok = unigram.unigram_tokenizer_from_pieces(
///   [#("<unk>", -100.0), #("<s>", -100.0), #("</s>", -100.0),
///    #("▁hello", -1.0)],
///   "<unk>", "<s>", "</s>",
/// )
/// let _ = unigram.unigram_decode(tok, [1, 3, 2])
/// ```
pub fn unigram_decode(tokenizer: UnigramTokenizer, ids: List(Int)) -> String {
  let pieces =
    ids
    |> list.filter(fn(id) { id != tokenizer.bos_id && id != tokenizer.eos_id })
    |> list.map(fn(id) {
      case dict.get(tokenizer.inverse_vocab, id) {
        Ok(token) -> token
        Error(_) -> ""
      }
    })

  let joined = string.concat(pieces)
  let with_spaces = string.replace(joined, sp_underscore, " ")
  case string.starts_with(with_spaces, " ") {
    True -> string.slice(with_spaces, 1, string.length(with_spaces) - 1)
    False -> with_spaces
  }
}

// =============================================================================
// SentencePiece wrapper (Unigram or BPE)
// =============================================================================

/// Which underlying segmentation a `SentencePieceTokenizer` uses.
pub type SentencePieceMode {
  SpUnigram
  SpBpe
}

/// Thin wrapper that lets a single value carry either the Unigram or the BPE
/// segmentation, preserving the SentencePiece `▁`-prefix convention.
///
/// Unlike BERT's WordPiece which uses `##` to mark continuation pieces and
/// strips it on the way out, SentencePiece marks the **start** of a token
/// with `▁` and treats everything else as raw bytes/graphemes — that's why
/// the BPE flavor here renormalizes input the same way the Unigram flavor
/// does (space to `▁`, leading `▁`).
pub type SentencePieceTokenizer {
  SentencePieceTokenizer(
    mode: SentencePieceMode,
    unigram: Option(UnigramTokenizer),
    bpe: Option(text_tokenizer.BpeTokenizer),
  )
}

/// Build a SentencePiece wrapper from an underlying Unigram tokenizer.
/// Encoding/decoding go through Viterbi.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/unigram
/// let inner = unigram.unigram_tokenizer_from_pieces(
///   [#("<unk>", -100.0), #("<s>", -100.0), #("</s>", -100.0),
///    #("▁a", -1.0)],
///   "<unk>", "<s>", "</s>",
/// )
/// let _ = unigram.sentence_piece_unigram(inner)
/// ```
pub fn sentence_piece_unigram(
  unigram: UnigramTokenizer,
) -> SentencePieceTokenizer {
  SentencePieceTokenizer(mode: SpUnigram, unigram: Some(unigram), bpe: None)
}

/// Build a SentencePiece wrapper from a BPE tokenizer. Encoding is the same
/// greedy merge loop as `bpe_encode` but with input renormalized to the
/// SentencePiece `▁`-prefix convention so vocabularies trained by
/// `sentencepiece --model_type=bpe` look up correctly.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor as t
/// import viva_tensor/text/unigram
/// let inner = t.bpe_tokenizer_from_vocab_and_merges(
///   ["?", "▁", "h", "i"], [], "?",
/// )
/// let _ = unigram.sentence_piece_bpe(inner)
/// ```
pub fn sentence_piece_bpe(
  bpe: text_tokenizer.BpeTokenizer,
) -> SentencePieceTokenizer {
  SentencePieceTokenizer(mode: SpBpe, unigram: None, bpe: Some(bpe))
}

/// Encode `text` according to the wrapper's `mode`. Falls back to an empty
/// list if the wrapper was constructed with a mismatched flavor (should
/// never happen via the public constructors).
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/unigram
/// let inner = unigram.unigram_tokenizer_from_pieces(
///   [#("<unk>", -100.0), #("<s>", -100.0), #("</s>", -100.0),
///    #("▁hi", -1.0)],
///   "<unk>", "<s>", "</s>",
/// )
/// let _ = unigram.sentence_piece_encode(
///   unigram.sentence_piece_unigram(inner),
///   "hi",
/// )
/// ```
pub fn sentence_piece_encode(
  tokenizer: SentencePieceTokenizer,
  text: String,
) -> List(Int) {
  case tokenizer.mode, tokenizer.unigram, tokenizer.bpe {
    SpUnigram, Some(inner), _ -> unigram_encode(inner, text)
    SpBpe, _, Some(inner) ->
      text_tokenizer.bpe_encode(inner, normalize_sp(text))
    _, _, _ -> []
  }
}

/// Decode ids back to a string. Reuses the underlying flavor's decoder; for
/// BPE the SentencePiece `▁` markers in the surviving pieces are mapped back
/// to ASCII spaces, mirroring `unigram_decode`.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/text/unigram
/// let inner = unigram.unigram_tokenizer_from_pieces(
///   [#("<unk>", -100.0), #("<s>", -100.0), #("</s>", -100.0),
///    #("▁hi", -1.0)],
///   "<unk>", "<s>", "</s>",
/// )
/// let _ = unigram.sentence_piece_decode(
///   unigram.sentence_piece_unigram(inner),
///   [1, 3, 2],
/// )
/// ```
pub fn sentence_piece_decode(
  tokenizer: SentencePieceTokenizer,
  ids: List(Int),
) -> String {
  case tokenizer.mode, tokenizer.unigram, tokenizer.bpe {
    SpUnigram, Some(inner), _ -> unigram_decode(inner, ids)
    SpBpe, _, Some(inner) -> {
      let raw = text_tokenizer.bpe_decode(inner, ids)
      let with_spaces = string.replace(raw, sp_underscore, " ")
      case string.starts_with(with_spaces, " ") {
        True -> string.slice(with_spaces, 1, string.length(with_spaces) - 1)
        False -> with_spaces
      }
    }
    _, _, _ -> ""
  }
}

// =============================================================================
// Internal helpers
// =============================================================================

/// Penalty applied to a single-grapheme fallback when no in-vocab piece
/// covers a position. Large negative so a piece-based segmentation always
/// wins when available.
const unk_score: Float = -1.0e6

fn normalize_sp(text: String) -> String {
  let with_marks = string.replace(text, " ", sp_underscore)
  case string.starts_with(with_marks, sp_underscore) {
    True -> with_marks
    False -> sp_underscore <> with_marks
  }
}

/// Viterbi over the grapheme list. Returns the most likely piece sequence.
fn viterbi_segment(
  tokenizer: UnigramTokenizer,
  graphemes: List(String),
) -> List(String) {
  let n = list.length(graphemes)
  case n {
    0 -> []
    _ -> {
      let positions = build_positions(graphemes)
      // dp: position i -> (best_score, best_piece_start, best_piece)
      let init_dp =
        dict.new()
        |> dict.insert(0, #(0.0, -1, ""))
      let dp = fill_dp(tokenizer, positions, n, 1, init_dp)
      backtrack(dp, n, [])
    }
  }
}

/// Pre-index graphemes into a Dict(Int, String) for O(1) random access.
fn build_positions(graphemes: List(String)) -> Dict(Int, String) {
  graphemes
  |> list.index_map(fn(g, i) { #(i, g) })
  |> dict.from_list
}

fn fill_dp(
  tokenizer: UnigramTokenizer,
  positions: Dict(Int, String),
  n: Int,
  i: Int,
  dp: Dict(Int, #(Float, Int, String)),
) -> Dict(Int, #(Float, Int, String)) {
  case i > n {
    True -> dp
    False -> {
      let best = best_predecessor(tokenizer, positions, dp, i, i - 1, None)
      let dp2 = dict.insert(dp, i, best)
      fill_dp(tokenizer, positions, n, i + 1, dp2)
    }
  }
}

/// Try every start `j` in [0, i-1] and keep the best `dp[j] + score(slice)`.
fn best_predecessor(
  tokenizer: UnigramTokenizer,
  positions: Dict(Int, String),
  dp: Dict(Int, #(Float, Int, String)),
  i: Int,
  j: Int,
  best: Option(#(Float, Int, String)),
) -> #(Float, Int, String) {
  case j < 0 {
    True ->
      case best {
        Some(b) -> b
        None -> #(unk_score, i - 1, "")
      }
    False -> {
      let candidate = case dict.get(dp, j) {
        Error(_) -> None
        Ok(#(prev_score, _, _)) -> {
          let piece = slice_graphemes(positions, j, i)
          case score_for_piece(tokenizer, piece) {
            Some(s) -> Some(#(prev_score +. s, j, piece))
            None -> None
          }
        }
      }
      let new_best = case candidate, best {
        None, _ -> best
        Some(c), None -> Some(c)
        Some(c), Some(b) ->
          case c.0 >. b.0 {
            True -> Some(c)
            False -> Some(b)
          }
      }
      best_predecessor(tokenizer, positions, dp, i, j - 1, new_best)
    }
  }
}

/// Score a single piece. Pieces present in `scores` use their log-prob;
/// single-grapheme pieces fall back to `unk_score` to allow UNK coverage;
/// everything else returns `None` (this segmentation is impossible).
fn score_for_piece(
  tokenizer: UnigramTokenizer,
  piece: String,
) -> Option(Float) {
  case dict.get(tokenizer.scores, piece) {
    Ok(s) -> Some(s)
    Error(_) ->
      case string.length(piece) {
        1 -> Some(unk_score)
        _ -> None
      }
  }
}

fn slice_graphemes(
  positions: Dict(Int, String),
  start: Int,
  end: Int,
) -> String {
  collect_slice(positions, start, end, "")
}

fn collect_slice(
  positions: Dict(Int, String),
  i: Int,
  end: Int,
  acc: String,
) -> String {
  case i >= end {
    True -> acc
    False ->
      case dict.get(positions, i) {
        Ok(g) -> collect_slice(positions, i + 1, end, acc <> g)
        Error(_) -> acc
      }
  }
}

fn backtrack(
  dp: Dict(Int, #(Float, Int, String)),
  i: Int,
  acc: List(String),
) -> List(String) {
  case i <= 0 {
    True -> acc
    False ->
      case dict.get(dp, i) {
        Ok(#(_, prev, piece)) ->
          case prev < 0 {
            True -> acc
            False -> backtrack(dp, prev, [piece, ..acc])
          }
        Error(_) -> acc
      }
  }
}

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
