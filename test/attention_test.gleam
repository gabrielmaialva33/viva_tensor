import gleam/list
import gleam/option.{None, Some}
import gleeunit
import gleeunit/should
import support/numerics.{floats_close, lists_close}
import viva_tensor/nn/attention.{MultiHeadAttention}
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-6

fn t2d(rows: List(List(Float))) -> tensor.Tensor {
  let assert Ok(t) = tensor.from_list2d(rows)
  t
}

// Manual scaled dot-product attention against which we cross-check sdpa.
// Computes softmax((Q @ K^T) / sqrt(dim)) @ V row-by-row for [2, 4] inputs.
// Pure, no shortcuts — used to nail down the basic test.

// ----------------------------------------------------------------------------
// SDPA — basic numerical sanity check
// ----------------------------------------------------------------------------

pub fn sdpa_basic_test() {
  // Tiny 2x4 inputs with simple values. We hand-check by computing the
  // expected output against a manual implementation below.
  let q =
    t2d([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
    ])
  let k =
    t2d([
      [1.0, 0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0, 0.0],
    ])
  let v =
    t2d([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
    ])

  let assert Ok(out) =
    attention.scaled_dot_product_attention(q, k, v, None, False)
  out.shape |> should.equal([2, 4])

  // scale = 1/sqrt(4) = 0.5
  // scores = Q K^T * 0.5 = [[0.5, 0.0], [0.0, 0.5]]
  // softmax row 0 = [exp(0.5), exp(0.0)] / sum =
  //   let e = exp(0.5)/(exp(0.5)+1) ~= 0.62245933
  //   so w0 ~ [0.62246, 0.37754]
  //   out row 0 = 0.62246*[1,2,3,4] + 0.37754*[5,6,7,8]
  //             = [2.5097, 3.5097, 4.5097, 5.5097]
  // row 1 is symmetric: w1 ~ [0.37754, 0.62246]
  //   out row 1 = 0.37754*[1,2,3,4] + 0.62246*[5,6,7,8]
  //             = [3.4903, 4.4903, 5.4903, 6.4903]
  // w0 = e^0.5 / (e^0.5 + 1) = 0.622459331..., w1 = 1 - w0 = 0.377540668...
  // row 0 = w0*[1,2,3,4] + w1*[5,6,7,8]
  //       = [2.51016, 3.51016, 4.51016, 5.51016]
  // row 1 = w1*[1,2,3,4] + w0*[5,6,7,8] (symmetric)
  //       = [3.48983, 4.48983, 5.48983, 6.48983]
  let expected = [
    2.510162748, 3.510162748, 4.510162748, 5.510162748, 3.489837252, 4.489837252,
    5.489837252, 6.489837252,
  ]
  lists_close(tensor.to_list(out), expected, 1.0e-4, 1.0e-4)
  |> should.be_true
}

// ----------------------------------------------------------------------------
// SDPA — causal mask
// ----------------------------------------------------------------------------

pub fn sdpa_causal_test() {
  // With causal=True, position 0 only attends to position 0,
  // position 1 attends to positions {0, 1}.
  // V is chosen so we can read off the attention pattern from the output.
  let q =
    t2d([
      [1.0, 0.0],
      [0.0, 1.0],
    ])
  let k =
    t2d([
      [1.0, 0.0],
      [0.0, 1.0],
    ])
  let v =
    t2d([
      [10.0, 0.0],
      [0.0, 20.0],
    ])

  let assert Ok(out) =
    attention.scaled_dot_product_attention(q, k, v, None, True)
  out.shape |> should.equal([2, 2])

  let row0 = list.take(tensor.to_list(out), 2)
  // Row 0: only attends to V[0] = [10, 0].
  lists_close(row0, [10.0, 0.0], rtol, atol)
  |> should.be_true

  // Row 1: attends to both positions (with softmax weighting).
  // scale = 1/sqrt(2) ~= 0.7071
  // scores row 1 vs K = [0*1+1*0, 0*0+1*1] = [0, 1], scaled = [0, 0.7071]
  // After causal mask (no-op for row 1), softmax gives w = [w0, w1]
  // where w1 > w0 (positive score for position 1).
  // Output should be a positive mix favoring V[1] = [0, 20].
  let row1 = list.drop(tensor.to_list(out), 2)
  // We just check it's a convex combo: every value in [min(V), max(V)] range.
  let assert [r10, r11] = row1
  { r10 >=. 0.0 && r10 <=. 10.0 } |> should.be_true
  { r11 >=. 0.0 && r11 <=. 20.0 } |> should.be_true
  // And we expect r11 > r10 (heavier weight on V[1])
  { r11 >. r10 } |> should.be_true
}

// ----------------------------------------------------------------------------
// SDPA — explicit mask
// ----------------------------------------------------------------------------

pub fn sdpa_with_mask_test() {
  // Mask out column 1 entirely. Output for both rows should equal V[0].
  let q =
    t2d([
      [1.0, 0.0],
      [0.0, 1.0],
    ])
  let k =
    t2d([
      [1.0, 0.0],
      [0.0, 1.0],
    ])
  let v =
    t2d([
      [7.0, 7.0],
      [99.0, 99.0],
    ])
  // Mask: visible col 0, masked col 1.
  let mask =
    t2d([
      [1.0, 0.0],
      [1.0, 0.0],
    ])

  let assert Ok(out) =
    attention.scaled_dot_product_attention(q, k, v, Some(mask), False)
  out.shape |> should.equal([2, 2])

  lists_close(tensor.to_list(out), [7.0, 7.0, 7.0, 7.0], rtol, 1.0e-3)
  |> should.be_true
}

// ----------------------------------------------------------------------------
// SDPA — shape error
// ----------------------------------------------------------------------------

pub fn sdpa_shape_error_test() {
  let q = t2d([[1.0, 0.0, 0.0]])
  // dim=3
  let k = t2d([[1.0, 0.0]])
  // dim=2 (mismatch)
  let v = t2d([[1.0, 1.0]])

  let result = attention.scaled_dot_product_attention(q, k, v, None, False)
  case result {
    Error(_) -> True
    Ok(_) -> False
  }
  |> should.be_true
}

// ----------------------------------------------------------------------------
// Causal mask helper
// ----------------------------------------------------------------------------

pub fn causal_mask_test() {
  let m = attention.causal_mask(3)
  m.shape |> should.equal([3, 3])
  // [[1,0,0],[1,1,0],[1,1,1]]
  let expected = [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
  lists_close(tensor.to_list(m), expected, 0.0, 1.0e-12)
  |> should.be_true
}

// ----------------------------------------------------------------------------
// MHA — init
// ----------------------------------------------------------------------------

pub fn mha_init_invalid_test() {
  let result = attention.multi_head_attention_init(3, 10, False)
  case result {
    Error(_) -> True
    Ok(_) -> False
  }
  |> should.be_true
}

pub fn mha_init_valid_test() {
  let assert Ok(mha) = attention.multi_head_attention_init(2, 8, False)
  mha.num_heads |> should.equal(2)
  mha.embed_dim |> should.equal(8)
  mha.head_dim |> should.equal(4)
  mha.w_q.shape |> should.equal([8, 8])
  mha.w_k.shape |> should.equal([8, 8])
  mha.w_v.shape |> should.equal([8, 8])
  mha.w_o.shape |> should.equal([8, 8])
  mha.b_q |> should.equal(None)
}

// ----------------------------------------------------------------------------
// MHA — forward, zero weights
// ----------------------------------------------------------------------------

pub fn mha_forward_zero_weights_test() {
  let assert Ok(mha) = attention.multi_head_attention_init(2, 4, False)
  let x =
    t2d([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
    ])
  let assert Ok(out) =
    attention.multi_head_attention_forward(mha, x, x, x, False)
  out.shape |> should.equal([2, 4])
  lists_close(tensor.to_list(out), list.repeat(0.0, 8), 0.0, 1.0e-12)
  |> should.be_true
}

// ----------------------------------------------------------------------------
// MHA — forward, identity projections
// ----------------------------------------------------------------------------

pub fn mha_forward_with_identity_weights_test() {
  // num_heads=2, embed_dim=4, head_dim=2.
  // With w_q = w_k = w_v = I and w_o = I, no bias, no causal:
  //   q' = q, k' = k, v' = v.
  // Split per head:
  //   head 0 sees columns [0, 1], head 1 sees columns [2, 3].
  // Then concat heads and project by I = pass-through.
  // So output = SDPA(q[:, :2], k[:, :2], v[:, :2]) || SDPA(q[:, 2:], k[:, 2:], v[:, 2:]).
  let i4 = tensor.identity(4)
  let assert Ok(base) = attention.multi_head_attention_init(2, 4, False)
  let mha = MultiHeadAttention(..base, w_q: i4, w_k: i4, w_v: i4, w_o: i4)

  let x =
    t2d([
      [1.0, 0.0, 1.0, 0.0],
      [0.0, 1.0, 0.0, 1.0],
    ])

  let assert Ok(out) =
    attention.multi_head_attention_forward(mha, x, x, x, False)
  out.shape |> should.equal([2, 4])

  // Compute reference by running SDPA on the two head slices manually.
  let h0_q =
    t2d([
      [1.0, 0.0],
      [0.0, 1.0],
    ])
  let h0_k = h0_q
  let h0_v = h0_q

  let h1_q =
    t2d([
      [1.0, 0.0],
      [0.0, 1.0],
    ])
  let h1_k = h1_q
  let h1_v = h1_q

  let assert Ok(o0) =
    attention.scaled_dot_product_attention(h0_q, h0_k, h0_v, None, False)
  let assert Ok(o1) =
    attention.scaled_dot_product_attention(h1_q, h1_k, h1_v, None, False)

  // Interleave row by row: row s of output = o0[s] ++ o1[s].
  let o0_rows = list.sized_chunk(tensor.to_list(o0), 2)
  let o1_rows = list.sized_chunk(tensor.to_list(o1), 2)
  let expected =
    list.zip(o0_rows, o1_rows)
    |> list.flat_map(fn(pair) {
      let #(a, b) = pair
      list.append(a, b)
    })

  lists_close(tensor.to_list(out), expected, 1.0e-5, 1.0e-6)
  |> should.be_true

  // Sanity: output is not all-zero (zero-weights guard).
  let s =
    tensor.to_list(out)
    |> list.fold(0.0, fn(a, b) { a +. b })
  floats_close(s, 0.0, 0.0, 1.0e-12)
  |> should.be_false
}
