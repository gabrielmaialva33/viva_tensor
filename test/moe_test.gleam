//// Tests for the Mixture-of-Experts module (`viva_tensor/nn/moe`).
////
//// Strategy: hand-crafted router gates so top-k selection is deterministic
//// regardless of float rounding. The load-balancing loss is exercised on
//// both the "perfectly uniform" sweet spot (≈ 1.0) and the "all tokens to
//// one expert" pathological case (a much higher value).

import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import support/numerics.{floats_close, lists_close}
import viva_tensor/nn/moe.{type MoeBlock, type Router, Router}
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

// Build a router with a hand-crafted gate. The gate is column-i biased so
// token i selects expert i (deterministic top-1).
fn diag_router(embed_dim: Int, num_experts: Int, top_k: Int) -> Router {
  let router = moe.router_init(embed_dim, num_experts, top_k)
  let gate_data =
    range_int(0, embed_dim - 1)
    |> list.flat_map(fn(r) {
      range_int(0, num_experts - 1)
      |> list.map(fn(c) {
        case r == c {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  let gate = tensor.Tensor(data: gate_data, shape: [embed_dim, num_experts])
  Router(..router, gate: gate)
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

pub fn router_init_test() {
  let router = moe.router_init(8, 4, 2)
  router.gate.shape |> should.equal([8, 4])
  router.num_experts |> should.equal(4)
  router.top_k |> should.equal(2)
  router.noise_std |> should.equal(0.0)
}

pub fn router_route_shape_test() {
  let router = diag_router(4, 4, 2)
  // 3 tokens, embed_dim = 4.
  let tokens =
    t2d([
      [1.0, 0.5, 0.2, 0.1],
      [0.0, 1.0, 0.3, 0.7],
      [0.1, 0.2, 1.0, 0.0],
    ])
  let assert Ok(#(ids, weights, aux)) = moe.router_route(router, tokens)
  ids.shape |> should.equal([3, 2])
  weights.shape |> should.equal([3, 2])
  aux.shape |> should.equal([])
}

pub fn router_top_1_test() {
  let router = diag_router(4, 4, 1)
  let tokens =
    t2d([
      [2.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 3.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
    ])
  let assert Ok(#(ids, weights, _aux)) = moe.router_route(router, tokens)
  // Each row's diagonal-projection logit dominates → expert == column index of
  // the largest input value.
  tensor.to_list(ids) |> should.equal([0.0, 2.0, 1.0])
  // Softmax over a single value is 1.0.
  lists_close(tensor.to_list(weights), [1.0, 1.0, 1.0], rtol, atol)
  |> should.be_true
}

pub fn router_softmax_normalizes_test() {
  let router = diag_router(4, 4, 3)
  let tokens =
    t2d([
      [3.0, 1.0, 2.0, 0.5],
      [0.1, 2.0, 3.0, 1.5],
    ])
  let assert Ok(#(_ids, weights, _aux)) = moe.router_route(router, tokens)
  // Each row of top-k weights must sum to ~1.0.
  let rows = list.sized_chunk(tensor.to_list(weights), 3)
  list.all(rows, fn(row) {
    let s = list.fold(row, 0.0, fn(a, v) { a +. v })
    floats_close(s, 1.0, rtol, atol)
  })
  |> should.be_true
}

// ---------------------------------------------------------------------------
// MoeBlock
// ---------------------------------------------------------------------------

pub fn moe_block_forward_shape_test() {
  // 4 tokens × embed_dim = 8, 4 experts, top_k = 2.
  let assert Ok(block) = moe.moe_block_init(8, 16, 4, 2)
  // All weights zero → output zero, but shape and aux-loss tensor must be
  // well-formed.
  let tokens = tensor.zeros([4, 8])
  let assert Ok(#(out, aux)) = moe.moe_block_forward(block, tokens)
  out.shape |> should.equal([4, 8])
  aux.shape |> should.equal([])
  // Zero-weight experts always emit zero.
  lists_close(tensor.to_list(out), list.repeat(0.0, 32), rtol, atol)
  |> should.be_true
  let _: MoeBlock = block
}

pub fn moe_block_init_invalid_top_k_test() {
  // top_k > num_experts must error out.
  let result = moe.moe_block_init(8, 16, 4, 5)
  case result {
    Error(_) -> Nil
    Ok(_) -> should.equal("expected Error", "")
  }
}

// ---------------------------------------------------------------------------
// Load-balancing helpers
// ---------------------------------------------------------------------------

pub fn load_balance_loss_uniform_test() {
  // 4 experts, 4 tokens, top_k = 1.
  // Each row of router_probs is uniform (0.25, 0.25, 0.25, 0.25). Each token
  // assigned to a different expert (perfect balance). Expected loss ≈ 1.0.
  let probs =
    t2d([
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
      [0.25, 0.25, 0.25, 0.25],
    ])
  let assignments =
    t2d([
      [0.0],
      [1.0],
      [2.0],
      [3.0],
    ])
  let assert Ok(loss) = moe.compute_load_balance_loss(probs, assignments, 4)
  loss.shape |> should.equal([])
  let val = case tensor.to_list(loss) {
    [v, ..] -> v
    [] -> 0.0
  }
  floats_close(val, 1.0, rtol, atol) |> should.be_true
}

pub fn load_balance_loss_unbalanced_test() {
  // 4 experts. Router probabilities are also collapsed onto expert 0, and
  // every token is routed to expert 0. Expected loss = num_experts * (f_0 *
  // p_0) = 4 * (1.0 * 1.0) = 4.0 — much higher than the uniform 1.0.
  let probs =
    t2d([
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
      [1.0, 0.0, 0.0, 0.0],
    ])
  let assignments =
    t2d([
      [0.0],
      [0.0],
      [0.0],
      [0.0],
    ])
  let assert Ok(loss) = moe.compute_load_balance_loss(probs, assignments, 4)
  let val = case tensor.to_list(loss) {
    [v, ..] -> v
    [] -> 0.0
  }
  // Sanity: must be strictly greater than the uniform baseline by a wide
  // margin (we expect ≈ 4.0).
  { val >. 2.0 } |> should.be_true
  floats_close(val, 4.0, rtol, atol) |> should.be_true
}

pub fn expert_distribution_test() {
  // Probs sum to 1 per row → importance must sum to num_tokens.
  let probs =
    t2d([
      [0.5, 0.3, 0.2],
      [0.2, 0.5, 0.3],
      [0.4, 0.4, 0.2],
    ])
  let assignments =
    t2d([
      [0.0, 1.0],
      [1.0, 2.0],
      [0.0, 0.0],
    ])
  let assert Ok(#(importance, load)) =
    moe.expert_distribution(probs, assignments, 3)
  importance.shape |> should.equal([3])
  load.shape |> should.equal([3])

  // Importance: column sums of probs.
  // expert 0: 0.5 + 0.2 + 0.4 = 1.1
  // expert 1: 0.3 + 0.5 + 0.4 = 1.2
  // expert 2: 0.2 + 0.3 + 0.2 = 0.7
  lists_close(tensor.to_list(importance), [1.1, 1.2, 0.7], rtol, atol)
  |> should.be_true

  // Load counts:
  // expert 0: appears in rows [0,1] (1x) and [2,_] (2x) → 3
  // expert 1: rows [0,_] (1x) and [1,_] (1x) → 2
  // expert 2: row [1,_] (1x) → 1
  lists_close(tensor.to_list(load), [3.0, 2.0, 1.0], rtol, atol)
  |> should.be_true

  // Sums sanity:
  let importance_sum =
    list.fold(tensor.to_list(importance), 0.0, fn(a, v) { a +. v })
  floats_close(importance_sum, 3.0, rtol, atol) |> should.be_true
  let load_sum = list.fold(tensor.to_list(load), 0.0, fn(a, v) { a +. v })
  floats_close(load_sum, 6.0, rtol, atol) |> should.be_true

  // Avoid unused-import warning for `float`.
  let _ = float.absolute_value(1.0)
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
