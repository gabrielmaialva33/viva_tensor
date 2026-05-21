//// Scaled dot-product attention and Multi-Head Attention (MHA).
////
//// References:
//// - Vaswani et al. (2017). "Attention Is All You Need." NeurIPS.
////   https://arxiv.org/abs/1706.03762
////
//// Math:
//// - Scaled dot-product attention:
////     Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
////   where Q is [seq_q, d_k], K is [seq_k, d_k], V is [seq_k, d_v].
////
//// - Multi-Head Attention:
////     head_i = Attention(Q @ W_q_i, K @ W_k_i, V @ W_v_i)
////     MHA(Q, K, V) = Concat(head_1, ..., head_h) @ W_o
////   Each head sees a head_dim = embed_dim / num_heads slice of the projection.
////
//// Implementation notes:
//// - Pure Gleam, no NIF.
//// - We don't have batched matmul yet, so MHA loops sequentially over heads.
////   That's O(h) more BEAM-side overhead than a single batched matmul would
////   give, but each head's inner matmul still dispatches to whatever backend
////   `tensor.matmul` uses (potentially native).
//// - Masking: `mask` is element-wise multiplicative on the attention weights
////   before softmax (we subtract a very large negative for masked-out spots
////   so softmax zeros them out). `is_causal` is a convenience flag that
////   builds a lower-triangular mask internally; if both `mask` and
////   `is_causal` are provided, **`is_causal` wins** (the explicit mask is
////   ignored, matching PyTorch's `scaled_dot_product_attention` policy).

import gleam/float
import gleam/int
import gleam/list
import gleam/option.{type Option, None, Some}
import gleam/result
import viva_tensor/core/error.{type TensorError, InvalidShape, ShapeMismatch}
import viva_tensor/tensor.{type Tensor, Tensor}

// A large negative number used to mask attention logits before softmax.
// 1e9 is enough to drive softmax to ~0 in float32/float64.
const mask_neg_inf: Float = -1.0e9

// ---------------------------------------------------------------------------
// Scaled dot-product attention
// ---------------------------------------------------------------------------

/// Compute `softmax((Q @ K^T) / sqrt(d_k)) @ V`.
///
/// Shapes:
/// - `q`: `[seq_q, dim]`
/// - `k`: `[seq_k, dim]`
/// - `v`: `[seq_k, dim_v]`
/// - `mask` (optional): `[seq_q, seq_k]`, where `1.0` means visible and `0.0`
///   means masked-out. Internally we add `-1e9` to masked logits before
///   softmax, which makes them collapse to ~0 after normalization.
/// - `is_causal`: if `True`, an internal lower-triangular mask is applied so
///   that position `i` only attends to positions `j <= i`. When both a `mask`
///   and `is_causal=True` are given, `is_causal` takes precedence and the
///   explicit `mask` is ignored. This matches PyTorch's
///   `torch.nn.functional.scaled_dot_product_attention` semantics.
///
/// Output: `[seq_q, dim_v]`.
///
/// Errors:
/// - `ShapeMismatch` when `q.dim != k.dim`.
/// - `ShapeMismatch` when `k.seq != v.seq`.
/// - `InvalidShape` when any input is not rank-2.
pub fn scaled_dot_product_attention(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  mask: Option(Tensor),
  is_causal: Bool,
) -> Result(Tensor, TensorError) {
  case q.shape, k.shape, v.shape {
    [seq_q, dim_q], [seq_k, dim_k], [seq_v, _dim_v] -> {
      case dim_q == dim_k {
        False ->
          Error(ShapeMismatch(expected: [seq_q, dim_q], got: [seq_k, dim_k]))
        True ->
          case seq_k == seq_v {
            False -> Error(ShapeMismatch(expected: k.shape, got: v.shape))
            True -> sdpa_run(q, k, v, mask, is_causal, seq_q, seq_k, dim_q)
          }
      }
    }
    _, _, _ ->
      Error(InvalidShape(
        "scaled_dot_product_attention expects rank-2 q/k/v tensors",
      ))
  }
}

fn sdpa_run(
  q: Tensor,
  k: Tensor,
  v: Tensor,
  mask: Option(Tensor),
  is_causal: Bool,
  seq_q: Int,
  seq_k: Int,
  dim: Int,
) -> Result(Tensor, TensorError) {
  // scale = 1 / sqrt(dim)
  let scale = case float.square_root(int.to_float(dim)) {
    Ok(s) if s >. 0.0 -> 1.0 /. s
    _ -> 1.0
  }

  // scores = (Q @ K^T) * scale
  use k_t <- result.try(tensor.transpose(k))
  use raw_scores <- result.try(tensor.matmul(q, k_t))
  let scaled_scores = tensor.scale(raw_scores, scale)

  // Apply mask. is_causal wins over explicit mask when both are present.
  let effective_mask = case is_causal {
    True -> Some(causal_mask(int.max(seq_q, seq_k)))
    False -> mask
  }

  use masked_scores <- result.try(apply_mask(
    scaled_scores,
    effective_mask,
    seq_q,
    seq_k,
  ))

  // softmax over seq_k axis (axis=1 for [seq_q, seq_k])
  use weights <- result.try(tensor.softmax_axis(masked_scores, 1))

  // output = weights @ V
  tensor.matmul(weights, v)
}

fn apply_mask(
  scores: Tensor,
  mask: Option(Tensor),
  seq_q: Int,
  seq_k: Int,
) -> Result(Tensor, TensorError) {
  case mask {
    None -> Ok(scores)
    Some(m) -> {
      case m.shape {
        [mq, mk] if mq >= seq_q && mk >= seq_k -> {
          use score_data <- result.try(tensor.try_to_list(scores))
          use mask_data <- result.try(tensor.try_to_list(m))
          // If the mask is larger than the score matrix (causal mask uses
          // max(seq_q, seq_k)), crop it to [seq_q, seq_k] in row-major order.
          let cropped_mask = case mq == seq_q && mk == seq_k {
            True -> mask_data
            False -> crop_2d(mask_data, mq, mk, seq_q, seq_k)
          }
          let merged =
            list.map2(score_data, cropped_mask, fn(s, mv) {
              case mv >. 0.5 {
                True -> s
                False -> s +. mask_neg_inf
              }
            })
          Ok(Tensor(data: merged, shape: [seq_q, seq_k]))
        }
        _ -> Error(ShapeMismatch(expected: [seq_q, seq_k], got: m.shape))
      }
    }
  }
}

fn crop_2d(
  data: List(Float),
  rows: Int,
  cols: Int,
  new_rows: Int,
  new_cols: Int,
) -> List(Float) {
  let _ = rows
  data
  |> list.sized_chunk(cols)
  |> list.take(new_rows)
  |> list.map(fn(row) { list.take(row, new_cols) })
  |> list.flatten
}

// ---------------------------------------------------------------------------
// Causal mask
// ---------------------------------------------------------------------------

/// Lower-triangular `[seq_len, seq_len]` mask of `1.0`s (and `0.0` above the
/// diagonal). Useful when you want to inspect, debug, or customise the mask
/// used by `scaled_dot_product_attention(.., is_causal=True)`.
pub fn causal_mask(seq_len: Int) -> Tensor {
  let data =
    range_int(0, seq_len - 1)
    |> list.flat_map(fn(i) {
      range_int(0, seq_len - 1)
      |> list.map(fn(j) {
        case j <= i {
          True -> 1.0
          False -> 0.0
        }
      })
    })
  Tensor(data: data, shape: [seq_len, seq_len])
}

// ---------------------------------------------------------------------------
// Multi-Head Attention
// ---------------------------------------------------------------------------

/// Multi-Head Attention layer.
///
/// Weights `w_q`, `w_k`, `w_v`, `w_o` are all `[embed_dim, embed_dim]`.
/// Biases are optional and `[embed_dim]` when present.
///
/// Use `multi_head_attention_init` for the default zero-weights constructor
/// (toy default - tests typically build weights explicitly).
pub type MultiHeadAttention {
  MultiHeadAttention(
    num_heads: Int,
    embed_dim: Int,
    head_dim: Int,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    w_o: Tensor,
    b_q: Option(Tensor),
    b_k: Option(Tensor),
    b_v: Option(Tensor),
    b_o: Option(Tensor),
  )
}

/// Initialize an MHA module with zero-filled weight matrices.
///
/// This is a toy default: real training code should swap in
/// Xavier/Glorot or Kaiming initialization. The signature lets tests build
/// weights explicitly via record-update syntax.
///
/// Errors:
/// - `InvalidShape` when `embed_dim` is not divisible by `num_heads`.
pub fn multi_head_attention_init(
  num_heads: Int,
  embed_dim: Int,
  use_bias: Bool,
) -> Result(MultiHeadAttention, TensorError) {
  case num_heads <= 0 || embed_dim <= 0 {
    True ->
      Error(InvalidShape(
        "multi_head_attention: num_heads and embed_dim must be positive",
      ))
    False ->
      case embed_dim % num_heads {
        0 -> {
          let head_dim = embed_dim / num_heads
          let w = tensor.zeros([embed_dim, embed_dim])
          let bias = case use_bias {
            True -> Some(tensor.zeros([embed_dim]))
            False -> None
          }
          Ok(MultiHeadAttention(
            num_heads: num_heads,
            embed_dim: embed_dim,
            head_dim: head_dim,
            w_q: w,
            w_k: w,
            w_v: w,
            w_o: w,
            b_q: bias,
            b_k: bias,
            b_v: bias,
            b_o: bias,
          ))
        }
        _ ->
          Error(InvalidShape(
            "multi_head_attention: embed_dim ("
            <> int.to_string(embed_dim)
            <> ") not divisible by num_heads ("
            <> int.to_string(num_heads)
            <> ")",
          ))
      }
  }
}

/// Multi-Head Attention forward pass.
///
/// Steps:
/// 1. Linear-project q, k, v: `q' = q @ w_q (+ b_q)`, same for k and v.
/// 2. Reshape each projection to `[seq, num_heads, head_dim]` then transpose
///    to `[num_heads, seq, head_dim]` so we can iterate per head.
/// 3. For each head, run `scaled_dot_product_attention(q_h, k_h, v_h, None,
///    is_causal)`.
/// 4. Concatenate head outputs back to `[seq, embed_dim]`.
/// 5. Linear output projection: `out @ w_o (+ b_o)`.
///
/// Inputs:
/// - `q`, `k`, `v`: `[seq, embed_dim]` (same `seq` here for simplicity; the
///   per-head SDPA already handles different `seq_q` vs `seq_k`, but the
///   reshape logic assumes equal sequence lengths in this entrypoint).
///
/// Output: `[seq, embed_dim]`.
pub fn multi_head_attention_forward(
  mha: MultiHeadAttention,
  q: Tensor,
  k: Tensor,
  v: Tensor,
  is_causal: Bool,
) -> Result(Tensor, TensorError) {
  // Validate input shapes early.
  use seq <- result.try(check_mha_input("q", q, mha.embed_dim))
  use seq_k <- result.try(check_mha_input("k", k, mha.embed_dim))
  use seq_v <- result.try(check_mha_input("v", v, mha.embed_dim))
  case seq == seq_k && seq_k == seq_v {
    False -> Error(ShapeMismatch(expected: [seq, mha.embed_dim], got: k.shape))
    True -> {
      // Linear projections.
      use q_proj <- result.try(linear(q, mha.w_q, mha.b_q))
      use k_proj <- result.try(linear(k, mha.w_k, mha.b_k))
      use v_proj <- result.try(linear(v, mha.w_v, mha.b_v))

      // Split per head: [seq, embed_dim] -> List of [seq, head_dim] tensors
      // (one per head).
      use q_heads <- result.try(split_heads(
        q_proj,
        seq,
        mha.num_heads,
        mha.head_dim,
      ))
      use k_heads <- result.try(split_heads(
        k_proj,
        seq,
        mha.num_heads,
        mha.head_dim,
      ))
      use v_heads <- result.try(split_heads(
        v_proj,
        seq,
        mha.num_heads,
        mha.head_dim,
      ))

      // Run SDPA per head (BEAM-side loop — there's no batched matmul yet).
      let heads_zipped =
        list.zip(q_heads, list.zip(k_heads, v_heads))
        |> list.map(fn(triple) {
          let #(qh, rest) = triple
          let #(kh, vh) = rest
          #(qh, kh, vh)
        })

      use head_outputs <- result.try(
        list.try_map(heads_zipped, fn(t) {
          let #(qh, kh, vh) = t
          scaled_dot_product_attention(qh, kh, vh, None, is_causal)
        }),
      )

      // Concatenate heads back: List of [seq, head_dim] -> [seq, embed_dim].
      use concat <- result.try(concat_heads(
        head_outputs,
        seq,
        mha.num_heads,
        mha.head_dim,
      ))

      // Output projection.
      linear(concat, mha.w_o, mha.b_o)
    }
  }
}

fn check_mha_input(
  _name: String,
  t: Tensor,
  embed_dim: Int,
) -> Result(Int, TensorError) {
  case t.shape {
    [seq, e] if e == embed_dim -> Ok(seq)
    _ -> Error(ShapeMismatch(expected: [-1, embed_dim], got: t.shape))
  }
}

fn linear(
  x: Tensor,
  w: Tensor,
  b: Option(Tensor),
) -> Result(Tensor, TensorError) {
  use y <- result.try(tensor.matmul(x, w))
  case b {
    None -> Ok(y)
    Some(bias) -> add_bias_row(y, bias)
  }
}

// Add a [out_features] bias to every row of a [seq, out_features] tensor.
fn add_bias_row(t: Tensor, bias: Tensor) -> Result(Tensor, TensorError) {
  case t.shape, bias.shape {
    [seq, out], [bo] if bo == out -> {
      use t_data <- result.try(tensor.try_to_list(t))
      use b_data <- result.try(tensor.try_to_list(bias))
      let new_data =
        list.sized_chunk(t_data, out)
        |> list.flat_map(fn(row) { list.map2(row, b_data, float.add) })
      let _ = seq
      Ok(Tensor(data: new_data, shape: t.shape))
    }
    _, _ -> Error(ShapeMismatch(expected: t.shape, got: bias.shape))
  }
}

// Reshape [seq, embed_dim] into a list of `num_heads` tensors of shape
// [seq, head_dim]. Conceptually:
//   x_reshaped[s, h, d] = x[s, h * head_dim + d]
// then we transpose to [num_heads, seq, head_dim] and emit each head.
fn split_heads(
  x: Tensor,
  seq: Int,
  num_heads: Int,
  head_dim: Int,
) -> Result(List(Tensor), TensorError) {
  use data <- result.try(tensor.try_to_list(x))
  let rows = list.sized_chunk(data, num_heads * head_dim)
  // For each head h, gather the slice [h*head_dim .. h*head_dim+head_dim) from
  // every row.
  let heads =
    range_int(0, num_heads - 1)
    |> list.map(fn(h) {
      let head_data =
        rows
        |> list.flat_map(fn(row) {
          row
          |> list.drop(h * head_dim)
          |> list.take(head_dim)
        })
      Tensor(data: head_data, shape: [seq, head_dim])
    })
  Ok(heads)
}

// Inverse of split_heads: given a list of [seq, head_dim] tensors (one per
// head), rebuild a [seq, num_heads * head_dim] tensor by concatenating along
// the feature axis (per row).
fn concat_heads(
  heads: List(Tensor),
  seq: Int,
  num_heads: Int,
  head_dim: Int,
) -> Result(Tensor, TensorError) {
  // Extract per-head data as List(List(Float)) of rows.
  use head_rows <- result.try(
    list.try_map(heads, fn(h) {
      use d <- result.try(tensor.try_to_list(h))
      Ok(list.sized_chunk(d, head_dim))
    }),
  )
  // For each row index s, concat head_rows[h][s] for h in 0..num_heads.
  let combined =
    range_int(0, seq - 1)
    |> list.flat_map(fn(s) {
      head_rows
      |> list.flat_map(fn(rows) {
        rows
        |> list.drop(s)
        |> list.first
        |> result.unwrap([])
      })
    })
  let _ = num_heads
  Ok(Tensor(data: combined, shape: [seq, num_heads * head_dim]))
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
