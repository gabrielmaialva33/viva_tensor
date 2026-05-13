//// Tests for `viva_tensor/generate/speculative`.
////
//// Sampling tests use deterministic paths (greedy / temperature=0.0) where
//// possible. The `speculative_*` tests build draft/verify callbacks that
//// return the same logits regardless of prefix length, which gives us
//// reproducible output lengths even with random acceptance.

import gleam/list
import gleam/option
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/generate/speculative.{SamplingConfig, SpeculativeConfig}
import viva_tensor/tensor.{Tensor}

pub fn main() -> Nil {
  gleeunit.main()
}

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-7

// --- greedy_sample ----------------------------------------------------------

pub fn greedy_sample_test() {
  let t = Tensor([1.0, 5.0, 2.0, 4.0], [4])
  let assert Ok(idx) = speculative.greedy_sample(t)
  idx |> should.equal(1)
}

// --- apply_temperature ------------------------------------------------------

pub fn apply_temperature_test() {
  // temperature=0.5 → logits / 0.5 → 2x. Gap between consecutive logits
  // should double.
  let t = Tensor([1.0, 2.0, 3.0], [3])
  let scaled = speculative.apply_temperature(t, 0.5)
  case tensor.to_list(scaled) {
    [a, b, c] -> {
      numerics.floats_close(a, 2.0, rtol, atol) |> should.be_true()
      numerics.floats_close(b, 4.0, rtol, atol) |> should.be_true()
      numerics.floats_close(c, 6.0, rtol, atol) |> should.be_true()
      // The "gap" between consecutive logits doubles vs. the original [1,2,3].
      numerics.floats_close(b -. a, 2.0, rtol, atol) |> should.be_true()
      numerics.floats_close(c -. b, 2.0, rtol, atol) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- top_k_filter -----------------------------------------------------------

pub fn top_k_filter_test() {
  // k=2 on [1, 3, 2, 4]: keep 3 and 4, mask the rest.
  let t = Tensor([1.0, 3.0, 2.0, 4.0], [4])
  let assert Ok(filtered) = speculative.top_k_filter(t, 2)
  case tensor.to_list(filtered) {
    [a, b, c, d] -> {
      // 1.0 and 2.0 are below the top-2 threshold (3.0) → masked.
      { a <. -1.0e20 } |> should.be_true()
      { c <. -1.0e20 } |> should.be_true()
      // 3.0 and 4.0 are kept unchanged.
      numerics.floats_close(b, 3.0, rtol, atol) |> should.be_true()
      numerics.floats_close(d, 4.0, rtol, atol) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- top_p_filter -----------------------------------------------------------

pub fn top_p_filter_test() {
  // Highly peaky distribution: token 1 dominates.
  // softmax([0.0, 10.0, 0.0, 0.0]) ~ [~0, ~1, ~0, ~0].
  // p=0.5 — top-1 alone already crosses 0.5, so only token 1 survives.
  let t = Tensor([0.0, 10.0, 0.0, 0.0], [4])
  let assert Ok(filtered) = speculative.top_p_filter(t, 0.5)
  case tensor.to_list(filtered) {
    [a, b, c, d] -> {
      // Token 1 (logit 10.0) kept.
      numerics.floats_close(b, 10.0, rtol, atol) |> should.be_true()
      // Others masked.
      { a <. -1.0e20 } |> should.be_true()
      { c <. -1.0e20 } |> should.be_true()
      { d <. -1.0e20 } |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- sample (temperature → 0 == greedy) ------------------------------------

pub fn sample_with_temperature_zero_test() {
  let t = Tensor([0.1, 0.2, 9.0, 0.0, 0.5], [5])
  let cfg = SamplingConfig(temperature: 0.0, top_k: 0, top_p: 1.0)
  let assert Ok(idx) = speculative.sample(t, cfg)
  // Greedy picks the argmax (index 2 — logit 9.0).
  idx |> should.equal(2)
}

// --- speculative_decode: all accepted ---------------------------------------
//
// When draft_fn and verify_fn return identical logits, every accept_reject
// ratio is `p_target / p_draft = 1.0`, so every drafted token is accepted.
// The bonus token at the end brings the total to `K + 1`.

pub fn speculative_all_accepted_test() {
  let prefix = [42]
  let draft_tokens = 3
  let cfg =
    SpeculativeConfig(
      draft_tokens: draft_tokens,
      sampling: SamplingConfig(temperature: 0.0, top_k: 0, top_p: 1.0),
      max_new_tokens: 4,
    )

  // Same logits every call → draft == verify → every token accepted.
  let logits_fn = fn(_prefix) { Ok(Tensor([0.1, 5.0, 0.2, 0.3], [4])) }
  let assert Ok(out) =
    speculative.speculative_decode(cfg, prefix, logits_fn, logits_fn)
  // prefix length + draft_tokens + 1 bonus
  list.length(out) |> should.equal(1 + draft_tokens + 1)
}

// --- speculative_decode: all rejected ---------------------------------------
//
// Draft picks index 0 (huge logit at 0). Verifier puts ~all mass on a
// different index. With temperature=0 the draft is deterministic and the
// verifier prob at the drafted token is ~0, so the acceptance probability
// `min(1, ~0 / ~1)` is ~0 and the first draft is rejected. The loop
// stops at that point and emits exactly one residual token. Output length
// is `prefix + 1`.

pub fn speculative_all_rejected_test() {
  let prefix = [0]
  let cfg =
    SpeculativeConfig(
      draft_tokens: 4,
      sampling: SamplingConfig(temperature: 0.0, top_k: 0, top_p: 1.0),
      max_new_tokens: 1,
    )
  let draft_fn = fn(_prefix) {
    // Argmax at index 0.
    Ok(Tensor([10.0, -10.0, -10.0, -10.0], [4]))
  }
  let verify_fn = fn(_prefix) {
    // Argmax at index 3, peaky enough that softmax[0] ~ 0.
    Ok(Tensor([-10.0, -10.0, -10.0, 10.0], [4]))
  }
  let assert Ok(out) =
    speculative.speculative_decode(cfg, prefix, draft_fn, verify_fn)
  list.length(out) |> should.equal(list.length(prefix) + 1)
}

// --- greedy_generate --------------------------------------------------------

pub fn greedy_generate_test() {
  // Model that always argmaxes at index 7.
  let model_fn = fn(_prefix) {
    Ok(Tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 9.0], [8]))
  }
  // Without stop, output should be prefix + 5 tokens, all 7s.
  let assert Ok(out) =
    speculative.greedy_generate([1, 2], 5, model_fn, option.None)
  list.length(out) |> should.equal(7)
  out
  |> list.drop(2)
  |> list.all(fn(t) { t == 7 })
  |> should.be_true()

  // With stop_token = 7, output halts after first generation.
  let assert Ok(out2) =
    speculative.greedy_generate([1, 2], 5, model_fn, option.Some(7))
  list.length(out2) |> should.equal(3)
  case list.last(out2) {
    Ok(last) -> last |> should.equal(7)
    Error(_) -> should.fail()
  }
}

pub fn greedy_generate_max_tokens_test() {
  let model_fn = fn(_prefix) { Ok(Tensor([0.0, 1.0, 0.0], [3])) }
  let assert Ok(out) = speculative.greedy_generate([], 3, model_fn, option.None)
  // 0 prefix + 3 generated tokens.
  list.length(out) |> should.equal(3)
  // All generated tokens are index 1.
  out |> list.all(fn(t) { t == 1 }) |> should.be_true()
}
