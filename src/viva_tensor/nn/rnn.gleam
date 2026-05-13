//// Recurrent neural network cells: vanilla RNN, GRU, and LSTM.
////
//// Pure Gleam, no autograd integration, no NIF. Each cell operates on
//// 1D vector inputs and hidden states ([input_size] / [hidden_size]).
//// Weight matrices follow PyTorch's `nn.RNNCell` / `nn.GRUCell` /
//// `nn.LSTMCell` layout, with multi-gate weights stacked along the row
//// dimension.
////
//// References:
//// - Elman (1990). "Finding Structure in Time." Vanilla RNN.
//// - Cho et al. (2014). "Learning Phrase Representations using RNN
////   Encoder-Decoder for Statistical Machine Translation." GRU.
//// - Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." LSTM.

import gleam/list
import gleam/result
import viva_tensor/core/error.{ShapeMismatch}
import viva_tensor/nn/activations
import viva_tensor/tensor.{type Tensor, type TensorError, Tensor}

// --- Vanilla RNN cell ------------------------------------------------------

/// Parameters for a single-step Elman RNN cell.
///
/// Weight shapes follow PyTorch's `nn.RNNCell`:
/// - `w_ih`: `[hidden_size, input_size]`
/// - `w_hh`: `[hidden_size, hidden_size]`
/// - `b_ih`, `b_hh`: `[hidden_size]`
pub type RnnCell {
  RnnCell(
    input_size: Int,
    hidden_size: Int,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
  )
}

/// Build an RNN cell with Xavier-initialized weights and zero biases.
///
/// `w_ih` has shape `[hidden_size, input_size]`, `w_hh` has shape
/// `[hidden_size, hidden_size]`, biases have shape `[hidden_size]`.
pub fn rnn_cell_init(input_size: Int, hidden_size: Int) -> RnnCell {
  RnnCell(
    input_size: input_size,
    hidden_size: hidden_size,
    w_ih: tensor.xavier_init(input_size, hidden_size),
    w_hh: tensor.xavier_init(hidden_size, hidden_size),
    b_ih: tensor.zeros([hidden_size]),
    b_hh: tensor.zeros([hidden_size]),
  )
}

/// One Elman RNN time step.
///
/// Computes `h' = tanh(W_ih @ x + b_ih + W_hh @ h + b_hh)`.
///
/// Errors with `ShapeMismatch` if `input` is not `[input_size]` or
/// `hidden` is not `[hidden_size]`.
pub fn rnn_cell_step(
  cell: RnnCell,
  input: Tensor,
  hidden: Tensor,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(check_vector(input, cell.input_size))
  use _ <- result.try(check_vector(hidden, cell.hidden_size))

  use wx <- result.try(tensor.matmul_vec(cell.w_ih, input))
  use wh <- result.try(tensor.matmul_vec(cell.w_hh, hidden))
  use s1 <- result.try(tensor.add(wx, cell.b_ih))
  use s2 <- result.try(tensor.add(wh, cell.b_hh))
  use pre <- result.try(tensor.add(s1, s2))
  Ok(activations.tanh(pre))
}

// --- GRU cell --------------------------------------------------------------

/// Parameters for a single-step GRU cell.
///
/// Weight matrices stack the reset, update, and new-gate rows in that
/// order:
/// - `w_ih`: `[3 * hidden_size, input_size]`
/// - `w_hh`: `[3 * hidden_size, hidden_size]`
/// - `b_ih`, `b_hh`: `[3 * hidden_size]`
pub type GruCell {
  GruCell(
    input_size: Int,
    hidden_size: Int,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
  )
}

/// Build a GRU cell with Xavier-initialized stacked weights and zero
/// biases.
pub fn gru_cell_init(input_size: Int, hidden_size: Int) -> GruCell {
  GruCell(
    input_size: input_size,
    hidden_size: hidden_size,
    w_ih: tensor.xavier_init(input_size, 3 * hidden_size),
    w_hh: tensor.xavier_init(hidden_size, 3 * hidden_size),
    b_ih: tensor.zeros([3 * hidden_size]),
    b_hh: tensor.zeros([3 * hidden_size]),
  )
}

/// One GRU time step (PyTorch `nn.GRUCell` convention).
///
/// ```
/// r = sigmoid(W_ir @ x + b_ir + W_hr @ h + b_hr)
/// z = sigmoid(W_iz @ x + b_iz + W_hz @ h + b_hz)
/// n = tanh(W_in @ x + b_in + r * (W_hn @ h + b_hn))
/// h' = (1 - z) * n + z * h
/// ```
///
/// Errors with `ShapeMismatch` if shapes do not match the cell.
pub fn gru_cell_step(
  cell: GruCell,
  input: Tensor,
  hidden: Tensor,
) -> Result(Tensor, TensorError) {
  use _ <- result.try(check_vector(input, cell.input_size))
  use _ <- result.try(check_vector(hidden, cell.hidden_size))

  let h = cell.hidden_size

  use ix <- result.try(tensor.matmul_vec(cell.w_ih, input))
  use ix <- result.try(tensor.add(ix, cell.b_ih))
  use #(ir, iz, in_) <- result.try(split3(ix, h))

  use hx <- result.try(tensor.matmul_vec(cell.w_hh, hidden))
  use hx <- result.try(tensor.add(hx, cell.b_hh))
  use #(hr, hz, hn) <- result.try(split3(hx, h))

  use r_pre <- result.try(tensor.add(ir, hr))
  let r = activations.sigmoid(r_pre)

  use z_pre <- result.try(tensor.add(iz, hz))
  let z = activations.sigmoid(z_pre)

  use rh_hn <- result.try(tensor.mul(r, hn))
  use n_pre <- result.try(tensor.add(in_, rh_hn))
  let n = activations.tanh(n_pre)

  // h' = (1 - z) * n + z * h
  let one_minus_z = tensor.map(z, fn(v) { 1.0 -. v })
  use left <- result.try(tensor.mul(one_minus_z, n))
  use right <- result.try(tensor.mul(z, hidden))
  tensor.add(left, right)
}

// --- LSTM cell -------------------------------------------------------------

/// Parameters for a single-step LSTM cell.
///
/// Gate weights stack input/forget/cell/output rows in that order:
/// - `w_ih`: `[4 * hidden_size, input_size]`
/// - `w_hh`: `[4 * hidden_size, hidden_size]`
/// - `b_ih`, `b_hh`: `[4 * hidden_size]`
pub type LstmCell {
  LstmCell(
    input_size: Int,
    hidden_size: Int,
    w_ih: Tensor,
    w_hh: Tensor,
    b_ih: Tensor,
    b_hh: Tensor,
  )
}

/// Build an LSTM cell with Xavier-initialized stacked weights and zero
/// biases.
pub fn lstm_cell_init(input_size: Int, hidden_size: Int) -> LstmCell {
  LstmCell(
    input_size: input_size,
    hidden_size: hidden_size,
    w_ih: tensor.xavier_init(input_size, 4 * hidden_size),
    w_hh: tensor.xavier_init(hidden_size, 4 * hidden_size),
    b_ih: tensor.zeros([4 * hidden_size]),
    b_hh: tensor.zeros([4 * hidden_size]),
  )
}

/// One LSTM time step.
///
/// ```
/// i = sigmoid(W_ii @ x + W_hi @ h + b_i)
/// f = sigmoid(W_if @ x + W_hf @ h + b_f)
/// g = tanh(   W_ig @ x + W_hg @ h + b_g)
/// o = sigmoid(W_io @ x + W_ho @ h + b_o)
/// c' = f * c + i * g
/// h' = o * tanh(c')
/// ```
///
/// Returns `(new_hidden, new_cell_state)` or `ShapeMismatch` if any
/// vector has the wrong length.
pub fn lstm_cell_step(
  cell: LstmCell,
  input: Tensor,
  hidden: Tensor,
  cell_state: Tensor,
) -> Result(#(Tensor, Tensor), TensorError) {
  use _ <- result.try(check_vector(input, cell.input_size))
  use _ <- result.try(check_vector(hidden, cell.hidden_size))
  use _ <- result.try(check_vector(cell_state, cell.hidden_size))

  let h = cell.hidden_size

  use ix <- result.try(tensor.matmul_vec(cell.w_ih, input))
  use ix <- result.try(tensor.add(ix, cell.b_ih))

  use hx <- result.try(tensor.matmul_vec(cell.w_hh, hidden))
  use hx <- result.try(tensor.add(hx, cell.b_hh))

  use gates <- result.try(tensor.add(ix, hx))
  use #(i_pre, f_pre, g_pre, o_pre) <- result.try(split4(gates, h))

  let i = activations.sigmoid(i_pre)
  let f = activations.sigmoid(f_pre)
  let g = activations.tanh(g_pre)
  let o = activations.sigmoid(o_pre)

  use fc <- result.try(tensor.mul(f, cell_state))
  use ig <- result.try(tensor.mul(i, g))
  use c_new <- result.try(tensor.add(fc, ig))

  let c_tanh = activations.tanh(c_new)
  use h_new <- result.try(tensor.mul(o, c_tanh))
  Ok(#(h_new, c_new))
}

// --- Sequence helpers -----------------------------------------------------

/// Run an RNN cell over a list of time steps.
///
/// Returns `(all_hidden_states, final_hidden)` where the list of hidden
/// states is in input order (one per time step).
pub fn rnn_sequence(
  cell: RnnCell,
  inputs: List(Tensor),
  initial_hidden: Tensor,
) -> Result(#(List(Tensor), Tensor), TensorError) {
  use #(rev_hs, final_h) <- result.try(
    list.try_fold(inputs, #([], initial_hidden), fn(acc, x) {
      let #(rev, h) = acc
      use h_new <- result.try(rnn_cell_step(cell, x, h))
      Ok(#([h_new, ..rev], h_new))
    }),
  )
  Ok(#(list.reverse(rev_hs), final_h))
}

/// Run a GRU cell over a list of time steps.
pub fn gru_sequence(
  cell: GruCell,
  inputs: List(Tensor),
  initial_hidden: Tensor,
) -> Result(#(List(Tensor), Tensor), TensorError) {
  use #(rev_hs, final_h) <- result.try(
    list.try_fold(inputs, #([], initial_hidden), fn(acc, x) {
      let #(rev, h) = acc
      use h_new <- result.try(gru_cell_step(cell, x, h))
      Ok(#([h_new, ..rev], h_new))
    }),
  )
  Ok(#(list.reverse(rev_hs), final_h))
}

/// Run an LSTM cell over a list of time steps.
///
/// Returns `(all_hidden_states, final_hidden, final_cell_state)`. The
/// cell state propagates implicitly across steps; only hidden states are
/// collected.
pub fn lstm_sequence(
  cell: LstmCell,
  inputs: List(Tensor),
  initial_hidden: Tensor,
  initial_cell: Tensor,
) -> Result(#(List(Tensor), Tensor, Tensor), TensorError) {
  use #(rev_hs, final_h, final_c) <- result.try(
    list.try_fold(inputs, #([], initial_hidden, initial_cell), fn(acc, x) {
      let #(rev, h, c) = acc
      use #(h_new, c_new) <- result.try(lstm_cell_step(cell, x, h, c))
      Ok(#([h_new, ..rev], h_new, c_new))
    }),
  )
  Ok(#(list.reverse(rev_hs), final_h, final_c))
}

// --- Internal helpers ------------------------------------------------------

fn check_vector(t: Tensor, expected: Int) -> Result(Nil, TensorError) {
  case tensor.shape(t) {
    [n] if n == expected -> Ok(Nil)
    other -> Error(ShapeMismatch(expected: [expected], got: other))
  }
}

fn split3(t: Tensor, h: Int) -> Result(#(Tensor, Tensor, Tensor), TensorError) {
  let data = tensor.to_list(t)
  let #(a, rest1) = list.split(data, h)
  let #(b, c) = list.split(rest1, h)
  Ok(#(
    Tensor(data: a, shape: [h]),
    Tensor(data: b, shape: [h]),
    Tensor(data: c, shape: [h]),
  ))
}

fn split4(
  t: Tensor,
  h: Int,
) -> Result(#(Tensor, Tensor, Tensor, Tensor), TensorError) {
  let data = tensor.to_list(t)
  let #(a, rest1) = list.split(data, h)
  let #(b, rest2) = list.split(rest1, h)
  let #(c, d) = list.split(rest2, h)
  Ok(#(
    Tensor(data: a, shape: [h]),
    Tensor(data: b, shape: [h]),
    Tensor(data: c, shape: [h]),
    Tensor(data: d, shape: [h]),
  ))
}
