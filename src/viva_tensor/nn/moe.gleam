//// Mixture of Experts (MoE) building blocks.
////
//// Replaces a dense feed-forward sublayer with `K` parallel "experts" plus a
//// routing network that picks the top-`top_k` experts for each token. This is
//// the Switch Transformer / GShard style of sparse MoE: only the gated
//// experts contribute to a token's output, so per-token FLOPs stay bounded as
//// total parameter count grows.
////
//// References:
//// - Shazeer et al. (2017). "Outrageously Large Neural Networks: The
////   Sparsely-Gated Mixture-of-Experts Layer." (Noisy top-k gating.)
////   https://arxiv.org/abs/1701.06538
//// - Fedus, Zoph, Shazeer (2021). "Switch Transformer: Scaling to Trillion
////   Parameter Models with Simple and Efficient Sparsity." (Load-balancing
////   loss formula used below.)
////   https://arxiv.org/abs/2101.03961
////
//// Style choice: pure Gleam, no NIF, no autograd integration in this round.
//// Forward pass dispatches per (token, k) pair sequentially — clear semantics,
//// O(tokens * top_k * expert_cost) work, easy to validate against reference
//// implementations. Production-grade routing batches tokens per expert to
//// amortise the matmul cost; we keep the toy form here.
////
//// What is **NOT** implemented in this round:
//// - **Expert capacity** (the `capacity_factor` knob from Switch Transformer
////   that caps the per-expert token budget and drops the overflow). Every
////   token is processed by every gated expert here, no dropping. Tracked as
////   a v2 follow-up.
//// - Autograd: the load-balancing loss tensor is computed in forward only;
////   no backward pass through the router gate.
//// - Native dispatch / fused expert matmuls.

import gleam/float
import gleam/int
import gleam/list
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape, ShapeMismatch}
import viva_tensor/nn/activations
import viva_tensor/nn/init as nn_init
import viva_tensor/tensor.{type Tensor, Tensor}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Learned routing network that maps a `[tokens, embed_dim]` batch to a
/// per-token distribution over `num_experts` experts and selects the top
/// `top_k` of them.
///
/// Fields:
/// - `gate`: `[embed_dim, num_experts]` projection that produces the logits.
/// - `num_experts`: total expert count `E`.
/// - `top_k`: number of experts each token is routed to (`1 <= top_k <= E`).
/// - `noise_std`: standard deviation of optional Gaussian noise added to the
///   logits before top-k selection (noisy-top-k gating from Shazeer 2017).
///   `0.0` disables noise — deterministic routing.
pub type Router {
  Router(gate: Tensor, num_experts: Int, top_k: Int, noise_std: Float)
}

/// Build a `Router` with a zero-filled gate matrix.
///
/// The gate is `[embed_dim, num_experts]`. Real training code should swap in
/// a small-variance init (e.g. `nn_init.truncated_normal`) via record-update
/// syntax; tests typically inject hand-crafted gates to make the top-k
/// selection deterministic.
///
/// Defaults `noise_std` to `0.0` (deterministic routing). Override the field
/// via record-update syntax to enable noisy-top-k gating.
pub fn router_init(embed_dim: Int, num_experts: Int, top_k: Int) -> Router {
  Router(
    gate: tensor.zeros([embed_dim, num_experts]),
    num_experts: num_experts,
    top_k: top_k,
    noise_std: 0.0,
  )
}

/// Route a batch of tokens through the gate.
///
/// Steps:
/// 1. `logits = tokens @ gate` — shape `[tokens, num_experts]`.
/// 2. If `noise_std > 0`, add `N(0, noise_std^2)` noise to the logits.
/// 3. For each row, pick the indices of the `top_k` largest logits.
/// 4. Softmax over the gathered top-k slice → normalized `expert_weights`.
/// 5. Compute the Switch Transformer load-balancing auxiliary loss using the
///    **full** softmax over `logits` for the importance term and the top-k
///    assignment mask for the load term.
///
/// Returns:
/// - `expert_ids`: `[tokens, top_k]` chosen expert indices, stored as a float
///   tensor for storage uniformity. Callers must round to int before
///   indexing (`float.round`).
/// - `expert_weights`: `[tokens, top_k]` softmax-normalized weights that sum
///   to `1.0` along the `top_k` axis (within numeric tolerance).
/// - `aux_loss`: scalar `Tensor` with the load-balancing loss (see
///   `compute_load_balance_loss`).
pub fn router_route(
  router: Router,
  tokens: Tensor,
) -> Result(#(Tensor, Tensor, Tensor), TensorError) {
  case tokens.shape, router.gate.shape {
    [num_tokens, embed_dim], [gate_in, num_experts]
      if embed_dim == gate_in && num_experts == router.num_experts
    -> {
      case router.top_k > 0 && router.top_k <= router.num_experts {
        False ->
          Error(InvalidShape(
            "router_route: top_k must be in 1..="
            <> int.to_string(router.num_experts)
            <> " (got "
            <> int.to_string(router.top_k)
            <> ")",
          ))
        True -> {
          use logits <- result.try(tensor.matmul(tokens, router.gate))
          let noisy_logits = case router.noise_std >. 0.0 {
            True -> add_noise(logits, router.noise_std)
            False -> logits
          }

          use logits_data <- result.try(tensor.try_to_list(noisy_logits))
          let rows = list.sized_chunk(logits_data, num_experts)

          let top_per_row =
            list.map(rows, fn(row) { top_k_row(row, router.top_k) })

          let ids_data =
            list.flat_map(top_per_row, fn(picks) {
              list.map(picks, fn(p) {
                let #(idx, _logit) = p
                int.to_float(idx)
              })
            })
          let expert_ids =
            Tensor(data: ids_data, shape: [num_tokens, router.top_k])

          let weights_data =
            list.flat_map(top_per_row, fn(picks) {
              let top_logits = list.map(picks, fn(p) { p.1 })
              softmax_row(top_logits)
            })
          let expert_weights =
            Tensor(data: weights_data, shape: [num_tokens, router.top_k])

          // Full softmax over all experts → router probabilities used for the
          // load-balancing importance term.
          use router_probs <- result.try(activations.softmax(noisy_logits, 1))
          use aux_loss <- result.try(compute_load_balance_loss(
            router_probs,
            expert_ids,
            router.num_experts,
          ))

          Ok(#(expert_ids, expert_weights, aux_loss))
        }
      }
    }
    [_, _], [_, _] ->
      Error(ShapeMismatch(expected: router.gate.shape, got: tokens.shape))
    _, _ ->
      Error(InvalidShape(
        "router_route: tokens must be rank-2 [tokens, embed_dim] and gate must be [embed_dim, num_experts]",
      ))
  }
}

// Add Gaussian noise of standard deviation `std` to every element.
fn add_noise(t: Tensor, std: Float) -> Tensor {
  let noise = nn_init.normal(t.shape, 0.0, std)
  case tensor.add(t, noise) {
    Ok(sum) -> sum
    Error(_) -> t
  }
}

// Pick the top-k entries of a single row by value.
//
// Returns up to `k` `#(index, value)` pairs in descending-value order. For
// rows shorter than `k` we return whatever is there (defensive — should not
// happen since callers validate `top_k <= num_experts`).
fn top_k_row(row: List(Float), k: Int) -> List(#(Int, Float)) {
  let indexed = list.index_map(row, fn(v, i) { #(i, v) })
  let sorted =
    list.sort(indexed, fn(a, b) {
      let #(_, va) = a
      let #(_, vb) = b
      float.compare(vb, va)
    })
  list.take(sorted, k)
}

// Numerically-stable softmax over a single row (small list, no rank concerns).
fn softmax_row(values: List(Float)) -> List(Float) {
  case values {
    [] -> []
    [first, ..rest] -> {
      let max_v = list.fold(rest, first, fn(acc, v) { float.max(acc, v) })
      let shifted = list.map(values, fn(v) { v -. max_v })
      let exps = list.map(shifted, float.exponential)
      let sum_exp = list.fold(exps, 0.0, fn(acc, v) { acc +. v })
      case sum_exp >. 0.0 {
        True -> list.map(exps, fn(v) { v /. sum_exp })
        False -> list.map(exps, fn(_) { 0.0 })
      }
    }
  }
}

// ---------------------------------------------------------------------------
// MoE block
// ---------------------------------------------------------------------------

/// Mixture-of-Experts feed-forward block.
///
/// Fields:
/// - `router`: routing network (gate + top-k config).
/// - `expert_w1`: list of `num_experts` matrices, each `[embed_dim, hidden_dim]`.
/// - `expert_w2`: list of `num_experts` matrices, each `[hidden_dim, embed_dim]`.
///
/// Bias terms are omitted for parity with the existing transformer FFN init
/// (zero biases are equivalent and keep the record small).
pub type MoeBlock {
  MoeBlock(router: Router, expert_w1: List(Tensor), expert_w2: List(Tensor))
}

/// Build a `MoeBlock` with zero-weight experts and a fresh router.
///
/// Errors:
/// - `InvalidShape` when `top_k <= 0`, `top_k > num_experts`, or any dim is
///   non-positive.
pub fn moe_block_init(
  embed_dim: Int,
  hidden_dim: Int,
  num_experts: Int,
  top_k: Int,
) -> Result(MoeBlock, TensorError) {
  case embed_dim <= 0 || hidden_dim <= 0 || num_experts <= 0 {
    True ->
      Error(InvalidShape(
        "moe_block_init: embed_dim, hidden_dim, and num_experts must be > 0",
      ))
    False ->
      case top_k <= 0 || top_k > num_experts {
        True ->
          Error(InvalidShape(
            "moe_block_init: top_k must satisfy 1 <= top_k <= num_experts (got top_k="
            <> int.to_string(top_k)
            <> ", num_experts="
            <> int.to_string(num_experts)
            <> ")",
          ))
        False -> {
          let w1s =
            list.repeat(Nil, num_experts)
            |> list.map(fn(_) { tensor.zeros([embed_dim, hidden_dim]) })
          let w2s =
            list.repeat(Nil, num_experts)
            |> list.map(fn(_) { tensor.zeros([hidden_dim, embed_dim]) })
          Ok(MoeBlock(
            router: router_init(embed_dim, num_experts, top_k),
            expert_w1: w1s,
            expert_w2: w2s,
          ))
        }
      }
  }
}

/// MoE forward pass.
///
/// 1. Route the `[tokens, embed_dim]` batch → `(expert_ids, expert_weights,
///    aux_loss)`.
/// 2. For every `(token, k)` pair:
///      - look up `expert_id = round(expert_ids[token, k])`
///      - run that expert on the single-token row:
///        `expert_out = relu(row @ w1) @ w2`
///      - scale by `expert_weights[token, k]`
///      - accumulate into the output row for `token`
/// 3. Stack rows back into a `[tokens, embed_dim]` tensor.
///
/// Complexity: O(tokens * top_k * expert_cost). The naive dispatch does **not**
/// batch tokens per expert — production code should gather tokens per expert
/// and run a single matmul per expert.
///
/// Returns `#(output, aux_loss)`. The auxiliary loss is returned alongside
/// so training code can scale-and-add it to the main loss.
pub fn moe_block_forward(
  block: MoeBlock,
  tokens: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  case tokens.shape {
    [num_tokens, embed_dim] -> {
      use #(expert_ids, expert_weights, aux_loss) <- result.try(router_route(
        block.router,
        tokens,
      ))

      use token_data <- result.try(tensor.try_to_list(tokens))
      use ids_data <- result.try(tensor.try_to_list(expert_ids))
      use weight_data <- result.try(tensor.try_to_list(expert_weights))

      let token_rows = list.sized_chunk(token_data, embed_dim)
      let id_rows = list.sized_chunk(ids_data, block.router.top_k)
      let weight_rows = list.sized_chunk(weight_data, block.router.top_k)

      use out_rows <- result.try(
        list.zip(token_rows, list.zip(id_rows, weight_rows))
        |> list.try_map(fn(triple) {
          let #(row, rest) = triple
          let #(row_ids, row_weights) = rest
          process_token_row(
            row,
            row_ids,
            row_weights,
            block.expert_w1,
            block.expert_w2,
            embed_dim,
          )
        }),
      )

      let out_data = list.flatten(out_rows)
      let out = Tensor(data: out_data, shape: [num_tokens, embed_dim])
      let _ = num_tokens
      Ok(#(out, aux_loss))
    }
    _ ->
      Error(InvalidShape(
        "moe_block_forward: tokens must be rank-2 [tokens, embed_dim]",
      ))
  }
}

// Run the `top_k` experts assigned to a single token and combine their
// outputs by the routing weights.
fn process_token_row(
  row: List(Float),
  ids: List(Float),
  weights: List(Float),
  expert_w1: List(Tensor),
  expert_w2: List(Tensor),
  embed_dim: Int,
) -> Result(List(Float), TensorError) {
  let pairs = list.zip(ids, weights)
  list.try_fold(pairs, list.repeat(0.0, embed_dim), fn(acc, pair) {
    let #(id_f, weight) = pair
    let id = float.round(id_f)
    case list_at(expert_w1, id), list_at(expert_w2, id) {
      Ok(w1), Ok(w2) -> {
        let row_tensor = Tensor(data: row, shape: [1, embed_dim])
        use h <- result.try(tensor.matmul(row_tensor, w1))
        let activated = activations.relu(h)
        use o <- result.try(tensor.matmul(activated, w2))
        use o_data <- result.try(tensor.try_to_list(o))
        Ok(list.map2(acc, o_data, fn(a, v) { a +. weight *. v }))
      }
      _, _ ->
        Error(InvalidShape(
          "moe_block_forward: expert id "
          <> int.to_string(id)
          <> " out of range",
        ))
    }
  })
}

fn list_at(xs: List(a), idx: Int) -> Result(a, Nil) {
  case idx < 0 {
    True -> Error(Nil)
    False ->
      xs
      |> list.drop(idx)
      |> list.first
  }
}

// ---------------------------------------------------------------------------
// Load-balancing helpers
// ---------------------------------------------------------------------------

/// Switch Transformer load-balancing auxiliary loss.
///
/// Given the full router probabilities `P` (shape `[tokens, num_experts]`)
/// and the top-k assignments `A` (shape `[tokens, top_k]`), define:
///
/// - `f_i = mean over tokens of P[:, i]` — average router probability for
///   expert `i` (smooth, differentiable).
/// - `p_i = (count of tokens assigned to expert i across all top-k slots) /
///          (tokens * top_k)` — fraction of dispatch slots that landed on
///   expert `i` (non-differentiable mask term).
///
/// Loss:
///
/// ```text
/// L = num_experts * sum_i (f_i * p_i)
/// ```
///
/// When the router is perfectly uniform (every expert sees `1/num_experts` of
/// the probability mass and `1/num_experts` of the assignments) the loss
/// equals `num_experts * num_experts * (1/num_experts)^2 = 1.0`. Higher loss
/// values indicate the gate is collapsing onto a small subset of experts.
///
/// Returns a scalar `Tensor` (`shape: []`).
pub fn compute_load_balance_loss(
  router_probs: Tensor,
  expert_assignments: Tensor,
  num_experts: Int,
) -> Result(Tensor, TensorError) {
  use #(importance, load) <- result.try(expert_distribution(
    router_probs,
    expert_assignments,
    num_experts,
  ))

  use importance_data <- result.try(tensor.try_to_list(importance))
  use load_data <- result.try(tensor.try_to_list(load))

  let num_tokens = case router_probs.shape {
    [n, _] -> n
    _ -> 0
  }
  let top_k = case expert_assignments.shape {
    [_, k] -> k
    _ -> 0
  }
  let total_assignments = num_tokens * top_k

  case num_tokens <= 0 || total_assignments <= 0 {
    True -> Ok(Tensor(data: [0.0], shape: []))
    False -> {
      let n_f = int.to_float(num_tokens)
      let total_f = int.to_float(total_assignments)
      let dot =
        list.zip(importance_data, load_data)
        |> list.fold(0.0, fn(acc, pair) {
          let #(imp, ld) = pair
          // f_i = imp / num_tokens, p_i = ld / total_assignments
          acc +. imp /. n_f *. { ld /. total_f }
        })
      let loss = int.to_float(num_experts) *. dot
      Ok(Tensor(data: [loss], shape: []))
    }
  }
}

/// Per-expert distribution statistics.
///
/// Returns `#(importance, load)`:
/// - `importance`: shape `[num_experts]`, `importance[i] = sum over tokens
///   of router_probs[:, i]`. Smooth, differentiable.
/// - `load`: shape `[num_experts]`, `load[i] = number of (token, k) slots
///   that selected expert i`. Discrete, non-differentiable.
///
/// `expert_assignments` is the float-encoded `[tokens, top_k]` id tensor
/// returned by `router_route`; entries are rounded to the nearest int.
/// Out-of-range ids are silently skipped (defensive — should not happen if
/// the assignments came from `router_route`).
pub fn expert_distribution(
  router_probs: Tensor,
  expert_assignments: Tensor,
  num_experts: Int,
) -> Result(#(Tensor, Tensor), TensorError) {
  case router_probs.shape, expert_assignments.shape {
    [tokens_p, ne], [tokens_a, _top_k] if ne == num_experts -> {
      case tokens_p == tokens_a {
        False ->
          Error(ShapeMismatch(
            expected: router_probs.shape,
            got: expert_assignments.shape,
          ))
        True -> {
          use probs_data <- result.try(tensor.try_to_list(router_probs))
          use ids_data <- result.try(tensor.try_to_list(expert_assignments))

          // Importance: column sums.
          let prob_rows = list.sized_chunk(probs_data, num_experts)
          let importance_data =
            range_int(0, num_experts - 1)
            |> list.map(fn(i) {
              list.fold(prob_rows, 0.0, fn(acc, row) {
                acc
                +. {
                  row
                  |> list.drop(i)
                  |> list.first
                  |> result.unwrap(0.0)
                }
              })
            })

          // Load: count of assignments per expert.
          let counts =
            list.fold(ids_data, list.repeat(0, num_experts), fn(acc, id_f) {
              let id = float.round(id_f)
              case id >= 0 && id < num_experts {
                False -> acc
                True -> increment_at(acc, id)
              }
            })
          let load_data = list.map(counts, int.to_float)

          Ok(#(
            Tensor(data: importance_data, shape: [num_experts]),
            Tensor(data: load_data, shape: [num_experts]),
          ))
        }
      }
    }
    _, _ ->
      Error(ShapeMismatch(expected: [-1, num_experts], got: router_probs.shape))
  }
}

fn increment_at(xs: List(Int), idx: Int) -> List(Int) {
  list.index_map(xs, fn(v, i) {
    case i == idx {
      True -> v + 1
      False -> v
    }
  })
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
