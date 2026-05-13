//// Speculative decoding and token sampling helpers.
////
//// This module implements two families of utilities:
////
//// 1. **Sampling** — temperature scaling, top-k filtering, top-p (nucleus)
////    filtering, and a `sample`/`greedy_sample` pair to pick a token id from
////    a 1-D `[vocab_size]` logits tensor.
//// 2. **Speculative decoding** — the Chen et al. 2023 / Leviathan et al.
////    2023 algorithm, which uses a small *draft* model to propose `K`
////    tokens that a larger *target* model verifies in a single forward
////    pass. Accepted prefixes are committed; the first rejected token is
////    resampled from the residual `max(0, p_target - p_draft)` distribution.
////
//// The "model" is supplied as a callback returning the next-token logits
//// given a prefix — this module does not own any inference engine.
////
//// All randomness is sourced from `int.random` (BEAM `:rand`). Tests can
//// reach deterministic outputs either by using a `temperature=0.0` shortcut
//// (degenerate to greedy) or by carefully seeding `:rand` from the caller.

import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/order
import gleam/result
import viva_tensor/core/error.{DimensionError, InvalidShape}
import viva_tensor/core/ffi
import viva_tensor/tensor.{type Tensor, type TensorError, Tensor}

// =============================================================================
// Sampling configuration
// =============================================================================

/// Sampling hyperparameters consumed by `sample`.
///
/// - `temperature`: divides logits before softmax. `1.0` disables scaling.
///   `0.0` is treated as the greedy limit (argmax).
/// - `top_k`: keep only the top `k` logits, set the rest to `-inf` so
///   softmax zeroes them. `0` disables top-k filtering.
/// - `top_p` (nucleus): keep the smallest set of tokens whose cumulative
///   probability exceeds `p`, mask the rest. `1.0` (or any value `>= 1.0`)
///   disables top-p filtering.
pub type SamplingConfig {
  SamplingConfig(temperature: Float, top_k: Int, top_p: Float)
}

// =============================================================================
// Speculative decoding configuration
// =============================================================================

/// Configuration for `speculative_decode`.
///
/// - `draft_tokens`: how many tokens the draft model proposes per loop
///   iteration (`K` in the literature). Common values are 4-8.
/// - `sampling`: sampling config used by the draft and the bonus token.
/// - `max_new_tokens`: total cap on appended tokens — the loop stops once
///   the number of appended tokens reaches this threshold.
pub type SpeculativeConfig {
  SpeculativeConfig(
    draft_tokens: Int,
    sampling: SamplingConfig,
    max_new_tokens: Int,
  )
}

// =============================================================================
// Constants
// =============================================================================

/// Upper bound passed to `int.random` to build a `[0.0, 1.0)` uniform float.
/// Matches the convention used by `viva_tensor/nn/init`. `2^31`.
const random_int_bound: Int = 2_147_483_648

/// Same value as a `Float`, kept separate to avoid `int.to_float` per sample.
const random_int_bound_f: Float = 2_147_483_648.0

/// Sentinel mask value used for "very small" logits. We do not use a literal
/// `-inf` because Float `-inf` paths through `ffi.exp` are platform-defined;
/// `-1.0e30` is small enough that `exp(x - max)` underflows to 0 for any
/// non-mask competitor, which is the behavior we want from filtering.
const neg_inf_mask: Float = -1.0e30

// =============================================================================
// Greedy sampling
// =============================================================================

/// Returns `argmax(logits)`.
///
/// Errors if `logits` is empty. Equivalent to `tensor.try_argmax` but
/// re-exported here so callers don't need a second import for the common
/// "greedy decode" path.
pub fn greedy_sample(logits: Tensor) -> Result(Int, TensorError) {
  tensor.try_argmax(logits)
}

// =============================================================================
// Temperature / top-k / top-p
// =============================================================================

/// Divide every logit by `temperature`.
///
/// Formula: `logits' = logits / temperature`.
///
/// - `temperature > 1.0` flattens the distribution (more entropy).
/// - `temperature < 1.0` sharpens it (closer to argmax).
/// - `temperature == 0.0` is treated by `sample` as the greedy limit and
///   short-circuits before this helper is called. Calling `apply_temperature`
///   with `temperature == 0.0` would divide by zero, so we return the input
///   unchanged in that defensive branch.
pub fn apply_temperature(logits: Tensor, temperature: Float) -> Tensor {
  case temperature == 0.0 || temperature == 1.0 {
    True -> logits
    False -> {
      let data = tensor.to_list(logits)
      let scaled = list.map(data, fn(x) { x /. temperature })
      Tensor(data: scaled, shape: tensor.shape(logits))
    }
  }
}

/// Set every logit outside the top-`k` to `-inf` (a large negative sentinel).
///
/// After this, `softmax` puts effectively zero mass on the filtered tokens.
///
/// - `k <= 0` is a no-op (returns the input unchanged).
/// - `k >= vocab_size` is also a no-op.
/// - Requires a 1-D tensor — returns `DimensionError` otherwise.
pub fn top_k_filter(logits: Tensor, k: Int) -> Result(Tensor, TensorError) {
  use data <- result.try(ensure_1d(logits))
  let n = list.length(data)
  case k <= 0 || k >= n {
    True -> Ok(logits)
    False -> {
      // Sort descending and grab the k-th largest value as the threshold.
      let sorted_desc = list.sort(data, fn(a, b) { compare_desc(a, b) })
      let threshold =
        sorted_desc
        |> list.drop(k - 1)
        |> list.first
        |> result.unwrap(0.0)

      // Mask values strictly below the threshold. Ties at the threshold are
      // kept; this can leave more than `k` entries unmasked but never fewer.
      let masked =
        list.map(data, fn(x) {
          case x <. threshold {
            True -> neg_inf_mask
            False -> x
          }
        })
      Ok(Tensor(data: masked, shape: tensor.shape(logits)))
    }
  }
}

/// Nucleus filter: mask the smallest-probability tokens whose cumulative
/// softmax mass would push the kept set above `p`.
///
/// Algorithm:
///   1. `probs = softmax(logits)`
///   2. Sort `(index, prob)` pairs by `prob` descending.
///   3. Walk the sorted list and accumulate probability. Include tokens
///      until adding the next one would have made the running sum `> p`.
///      In practice for a peaky distribution this can keep only the top-1
///      (e.g. when the top probability is already `> p`).
///   4. Apply the mask back to the original `logits`.
///
/// - `p >= 1.0` is a no-op (no filtering).
/// - `p <= 0.0` keeps only the argmax (the smallest valid nucleus).
/// - Requires a 1-D tensor.
pub fn top_p_filter(logits: Tensor, p: Float) -> Result(Tensor, TensorError) {
  use data <- result.try(ensure_1d(logits))
  case p >=. 1.0 {
    True -> Ok(logits)
    False -> {
      let probs = softmax_data(data)
      let indexed = list.index_map(probs, fn(prob, i) { #(i, prob) })
      let sorted =
        list.sort(indexed, fn(a, b) {
          let #(_, pa) = a
          let #(_, pb) = b
          compare_desc(pa, pb)
        })

      let keep_indices = nucleus_indices(sorted, p, 0.0, [])
      let masked =
        list.index_map(data, fn(x, i) {
          case list.contains(keep_indices, i) {
            True -> x
            False -> neg_inf_mask
          }
        })
      Ok(Tensor(data: masked, shape: tensor.shape(logits)))
    }
  }
}

/// Temperature + top-k + top-p sampling from a 1-D `[vocab_size]` logit
/// tensor.
///
/// Pipeline:
///   logits → (÷ temperature) → top-k mask → top-p mask → softmax →
///   inverse-CDF sample.
///
/// `temperature == 0.0` short-circuits to greedy (argmax) so callers can
/// use a single config knob to toggle determinism.
pub fn sample(
  logits: Tensor,
  config: SamplingConfig,
) -> Result(Int, TensorError) {
  case config.temperature == 0.0 {
    True -> greedy_sample(logits)
    False -> {
      let scaled = apply_temperature(logits, config.temperature)
      use after_k <- result.try(top_k_filter(scaled, config.top_k))
      use after_p <- result.try(top_p_filter(after_k, config.top_p))
      use data <- result.try(ensure_1d(after_p))
      case data {
        [] -> Error(InvalidShape("Empty logits tensor"))
        _ -> Ok(sample_from_probs(softmax_data(data)))
      }
    }
  }
}

// =============================================================================
// Speculative decoding
// =============================================================================

/// Speculative decoding (Chen et al. 2023, Leviathan et al. 2023).
///
/// Algorithm per outer iteration:
///
/// 1. Run the draft model autoregressively for `K = draft_tokens` steps,
///    capturing the draft logits at each position and a candidate token
///    sampled with `config.sampling`.
/// 2. Run the verifier on the prefix + drafted tokens. Conceptually this is
///    one parallel forward pass over `K+1` positions. We model that here by
///    calling `verify_fn` once per position (the user-supplied callback can
///    cache as it pleases — we don't assume a specific shape).
/// 3. For each drafted token `t` at position `i`:
///       `r ~ Uniform(0, 1)`
///       accept iff `r <= min(1, p_target(t) / p_draft(t))`
/// 4. On the first rejection, replace `t` with a sample from the residual
///    distribution `proportional to max(0, p_target - p_draft)`. Stop the
///    inner loop — everything after the rejection point is discarded.
/// 5. If every drafted token was accepted, sample one extra "bonus" token
///    from the verifier's distribution at the last draft position (the
///    position the verifier predicts "for free" thanks to the parallel
///    pass).
///
/// Numerical hazard: if `p_draft(t) == 0.0` for a drafted token, the ratio
/// `p_target / 0` blows up. We treat `p_draft <= 0.0` as "always accept"
/// (the ratio is mathematically `+inf`, clipped to `1.0` by the
/// `min(1, ratio)` rule). `p_draft` is essentially never zero for a token
/// the draft just *sampled*, but the guard is cheap and keeps downstream
/// math safe.
pub fn speculative_decode(
  config: SpeculativeConfig,
  initial_tokens: List(Int),
  draft_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  verify_fn: fn(List(Int)) -> Result(Tensor, TensorError),
) -> Result(List(Int), TensorError) {
  speculative_loop(config, initial_tokens, draft_fn, verify_fn, 0)
}

fn speculative_loop(
  config: SpeculativeConfig,
  tokens: List(Int),
  draft_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  verify_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  produced: Int,
) -> Result(List(Int), TensorError) {
  case produced >= config.max_new_tokens {
    True -> Ok(tokens)
    False -> {
      use #(drafts, draft_logits_list) <- result.try(
        draft_phase(config, tokens, draft_fn, config.draft_tokens, [], []),
      )
      use verify_logits_list <- result.try(
        verify_phase(tokens, drafts, verify_fn, []),
      )

      let #(accepted_tokens, rejected_at, residual_token) =
        acceptance_phase(
          drafts,
          draft_logits_list,
          verify_logits_list,
          config.sampling,
        )

      let new_tokens = list.append(tokens, accepted_tokens)
      let extra_token = case rejected_at {
        Some(_) -> [residual_token]
        None -> {
          case list.last(verify_logits_list) {
            Ok(last_logits) -> [bonus_sample(last_logits, config.sampling)]
            Error(_) -> []
          }
        }
      }

      let combined = list.append(new_tokens, extra_token)
      let new_produced = produced + list.length(accepted_tokens)
      let final_produced = new_produced + list.length(extra_token)

      case final_produced >= config.max_new_tokens {
        True -> Ok(combined)
        False ->
          speculative_loop(
            config,
            combined,
            draft_fn,
            verify_fn,
            final_produced,
          )
      }
    }
  }
}

fn draft_phase(
  config: SpeculativeConfig,
  prefix: List(Int),
  draft_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  remaining: Int,
  acc_tokens: List(Int),
  acc_logits: List(Tensor),
) -> Result(#(List(Int), List(Tensor)), TensorError) {
  case remaining <= 0 {
    True -> Ok(#(list.reverse(acc_tokens), list.reverse(acc_logits)))
    False -> {
      use logits <- result.try(draft_fn(prefix))
      use token <- result.try(sample(logits, config.sampling))
      draft_phase(
        config,
        list.append(prefix, [token]),
        draft_fn,
        remaining - 1,
        [token, ..acc_tokens],
        [logits, ..acc_logits],
      )
    }
  }
}

fn verify_phase(
  prefix: List(Int),
  drafts: List(Int),
  verify_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  acc: List(Tensor),
) -> Result(List(Tensor), TensorError) {
  case drafts {
    [] -> Ok(list.reverse(acc))
    [d, ..rest] -> {
      use logits <- result.try(verify_fn(prefix))
      verify_phase(list.append(prefix, [d]), rest, verify_fn, [logits, ..acc])
    }
  }
}

fn acceptance_phase(
  drafts: List(Int),
  draft_logits: List(Tensor),
  verify_logits: List(Tensor),
  sampling: SamplingConfig,
) -> #(List(Int), Option(Int), Int) {
  acceptance_loop(drafts, draft_logits, verify_logits, sampling, 0, [])
}

fn acceptance_loop(
  drafts: List(Int),
  draft_logits: List(Tensor),
  verify_logits: List(Tensor),
  sampling: SamplingConfig,
  index: Int,
  accepted: List(Int),
) -> #(List(Int), Option(Int), Int) {
  case drafts, draft_logits, verify_logits {
    [], _, _ -> #(list.reverse(accepted), None, 0)
    _, [], _ -> #(list.reverse(accepted), None, 0)
    _, _, [] -> #(list.reverse(accepted), None, 0)
    [tok, ..d_rest], [d_logits, ..dl_rest], [v_logits, ..vl_rest] -> {
      let d_data = tensor.to_list(apply_filters(d_logits, sampling))
      let v_data = tensor.to_list(apply_filters(v_logits, sampling))
      let d_probs = softmax_data(d_data)
      let v_probs = softmax_data(v_data)

      let p_draft = list_at(d_probs, tok)
      let p_target = list_at(v_probs, tok)

      case accept_reject(p_draft, p_target) {
        True ->
          acceptance_loop(d_rest, dl_rest, vl_rest, sampling, index + 1, [
            tok,
            ..accepted
          ])
        False -> {
          let resampled = resample_from_residual(d_probs, v_probs)
          #(list.reverse(accepted), Some(index), resampled)
        }
      }
    }
  }
}

/// Speculative sampling acceptance test:
///   accept with probability `min(1, p_target / p_draft)`.
///
/// Numerical hazard handling: if `p_draft <= 0.0`, we treat the ratio as
/// `+inf` and accept unconditionally (`min(1, +inf) = 1`). This protects
/// against the division-by-zero blowup described in the module docstring.
pub fn accept_reject(p_draft: Float, p_target: Float) -> Bool {
  let ratio = case p_draft <=. 0.0 {
    True -> 1.0
    False -> {
      let r = p_target /. p_draft
      case r >. 1.0 {
        True -> 1.0
        False -> r
      }
    }
  }
  let u = sample_unit()
  u <=. ratio
}

/// Resample from the residual distribution `max(0, p_target - p_draft)`,
/// renormalized. Used when a drafted token is rejected. If the residual
/// is degenerate (all zeros, which can happen when target == draft and we
/// somehow rejected anyway) the function falls back to argmax of `p_target`.
pub fn resample_from_residual(
  draft_probs: List(Float),
  target_probs: List(Float),
) -> Int {
  let residual =
    list.zip(target_probs, draft_probs)
    |> list.map(fn(pair) {
      let #(t, d) = pair
      case t -. d >. 0.0 {
        True -> t -. d
        False -> 0.0
      }
    })
  let total = list.fold(residual, 0.0, fn(acc, x) { acc +. x })
  case total <=. 0.0 {
    True -> argmax_of(target_probs)
    False -> {
      let normalized = list.map(residual, fn(x) { x /. total })
      sample_from_probs(normalized)
    }
  }
}

// =============================================================================
// Greedy streaming generation
// =============================================================================

/// Autoregressive greedy generation. Calls `model_fn(prefix)` repeatedly,
/// appends `argmax(logits)`, and stops on `stop_token` (if provided) or when
/// `max_new_tokens` new tokens have been produced.
///
/// Returns the full output sequence (initial prefix + generated tokens).
pub fn greedy_generate(
  initial_tokens: List(Int),
  max_new_tokens: Int,
  model_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  stop_token: Option(Int),
) -> Result(List(Int), TensorError) {
  greedy_generate_loop(initial_tokens, max_new_tokens, model_fn, stop_token, 0)
}

fn greedy_generate_loop(
  tokens: List(Int),
  max_new_tokens: Int,
  model_fn: fn(List(Int)) -> Result(Tensor, TensorError),
  stop_token: Option(Int),
  produced: Int,
) -> Result(List(Int), TensorError) {
  case produced >= max_new_tokens {
    True -> Ok(tokens)
    False -> {
      use logits <- result.try(model_fn(tokens))
      use next <- result.try(greedy_sample(logits))
      let new_tokens = list.append(tokens, [next])
      case stop_token {
        Some(stop) if stop == next -> Ok(new_tokens)
        _ ->
          greedy_generate_loop(
            new_tokens,
            max_new_tokens,
            model_fn,
            stop_token,
            produced + 1,
          )
      }
    }
  }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn apply_filters(logits: Tensor, sampling: SamplingConfig) -> Tensor {
  let scaled = apply_temperature(logits, sampling.temperature)
  let after_k =
    top_k_filter(scaled, sampling.top_k)
    |> result.unwrap(scaled)
  top_p_filter(after_k, sampling.top_p)
  |> result.unwrap(after_k)
}

fn bonus_sample(logits: Tensor, sampling: SamplingConfig) -> Int {
  case sample(logits, sampling) {
    Ok(t) -> t
    Error(_) -> 0
  }
}

fn ensure_1d(t: Tensor) -> Result(List(Float), TensorError) {
  case tensor.shape(t) {
    [_] -> Ok(tensor.to_list(t))
    _ ->
      Error(DimensionError("Sampling helpers require a 1-D [vocab_size] tensor"))
  }
}

/// Numerically stable softmax of a flat float list.
fn softmax_data(data: List(Float)) -> List(Float) {
  case data {
    [] -> []
    [first, ..rest] -> {
      let max_val =
        list.fold(rest, first, fn(acc, x) {
          case x >. acc {
            True -> x
            False -> acc
          }
        })
      let shifted = list.map(data, fn(x) { ffi.exp(x -. max_val) })
      let sum_exp = list.fold(shifted, 0.0, fn(acc, x) { acc +. x })
      case sum_exp <=. 0.0 {
        True -> list.map(data, fn(_) { 0.0 })
        False -> list.map(shifted, fn(x) { x /. sum_exp })
      }
    }
  }
}

/// Inverse-CDF sampler over a normalized probability vector.
fn sample_from_probs(probs: List(Float)) -> Int {
  let u = sample_unit()
  inverse_cdf(probs, u, 0.0, 0)
}

fn inverse_cdf(probs: List(Float), u: Float, cum: Float, index: Int) -> Int {
  case probs {
    [] -> index - 1
    [p, ..rest] -> {
      let new_cum = cum +. p
      case u <=. new_cum {
        True -> index
        False -> inverse_cdf(rest, u, new_cum, index + 1)
      }
    }
  }
}

fn sample_unit() -> Float {
  int.to_float(int.random(random_int_bound)) /. random_int_bound_f
}

fn list_at(data: List(Float), index: Int) -> Float {
  case data, index {
    [], _ -> 0.0
    [x, ..], 0 -> x
    [_, ..rest], i -> list_at(rest, i - 1)
  }
}

fn argmax_of(probs: List(Float)) -> Int {
  argmax_loop(probs, 0, 0, 0.0 -. 1.0e30)
}

fn argmax_loop(probs: List(Float), i: Int, best_i: Int, best_v: Float) -> Int {
  case probs {
    [] -> best_i
    [p, ..rest] ->
      case p >. best_v {
        True -> argmax_loop(rest, i + 1, i, p)
        False -> argmax_loop(rest, i + 1, best_i, best_v)
      }
  }
}

fn nucleus_indices(
  sorted: List(#(Int, Float)),
  p: Float,
  cum: Float,
  acc: List(Int),
) -> List(Int) {
  case sorted {
    [] -> acc
    [#(idx, prob), ..rest] -> {
      case cum >=. p {
        // Already crossed `p` — stop without including the current token.
        True -> acc
        False -> nucleus_indices(rest, p, cum +. prob, [idx, ..acc])
      }
    }
  }
}

fn compare_desc(a: Float, b: Float) -> order.Order {
  case a <. b {
    True -> order.Gt
    False ->
      case a >. b {
        True -> order.Lt
        False -> order.Eq
      }
  }
}
