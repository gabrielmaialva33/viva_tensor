//// Tests for recurrent cells (RNN / GRU / LSTM) in viva_tensor/nn/rnn.

import gleam/list
import gleeunit
import gleeunit/should
import support/numerics.{floats_close, lists_close}
import viva_tensor/core/error.{ShapeMismatch}
import viva_tensor/nn/rnn.{
  type GruCell, type LstmCell, type RnnCell, GruCell, LstmCell, RnnCell,
  gru_cell_step, gru_sequence, lstm_cell_step, lstm_sequence, rnn_cell_step,
  rnn_sequence,
}
import viva_tensor/tensor.{Tensor}

pub fn main() {
  gleeunit.main()
}

// --- Precomputed activation constants (avoid recomputing in every test) ----

const tol = 1.0e-9

// tanh(0.5)
const tanh_half = 0.46211715726000974

// tanh(1.0)
const tanh_one = 0.7615941559557649

// 0.5 * tanh(1.0)
const lstm_zero_h = 0.38079707797788245

// --- Helpers ---------------------------------------------------------------

fn vec(xs: List(Float)) -> tensor.Tensor {
  Tensor(data: xs, shape: [list.length(xs)])
}

fn mat(rows: List(List(Float))) -> tensor.Tensor {
  let rows_n = list.length(rows)
  let cols = case rows {
    [] -> 0
    [first, ..] -> list.length(first)
  }
  Tensor(data: list.flatten(rows), shape: [rows_n, cols])
}

fn zero_rnn_cell(input_size: Int, hidden_size: Int) -> RnnCell {
  RnnCell(
    input_size: input_size,
    hidden_size: hidden_size,
    w_ih: tensor.zeros([hidden_size, input_size]),
    w_hh: tensor.zeros([hidden_size, hidden_size]),
    b_ih: tensor.zeros([hidden_size]),
    b_hh: tensor.zeros([hidden_size]),
  )
}

fn zero_gru_cell(input_size: Int, hidden_size: Int) -> GruCell {
  GruCell(
    input_size: input_size,
    hidden_size: hidden_size,
    w_ih: tensor.zeros([3 * hidden_size, input_size]),
    w_hh: tensor.zeros([3 * hidden_size, hidden_size]),
    b_ih: tensor.zeros([3 * hidden_size]),
    b_hh: tensor.zeros([3 * hidden_size]),
  )
}

fn zero_lstm_cell(input_size: Int, hidden_size: Int) -> LstmCell {
  LstmCell(
    input_size: input_size,
    hidden_size: hidden_size,
    w_ih: tensor.zeros([4 * hidden_size, input_size]),
    w_hh: tensor.zeros([4 * hidden_size, hidden_size]),
    b_ih: tensor.zeros([4 * hidden_size]),
    b_hh: tensor.zeros([4 * hidden_size]),
  )
}

// --- Vanilla RNN -----------------------------------------------------------

pub fn rnn_cell_zero_test() {
  let cell = zero_rnn_cell(3, 2)
  let assert Ok(out) =
    rnn_cell_step(cell, vec([1.0, 2.0, 3.0]), vec([4.0, 5.0]))
  lists_close(tensor.to_list(out), [0.0, 0.0], tol, tol) |> should.be_true
}

pub fn rnn_cell_step_test() {
  // w_ih = I, w_hh = 0, biases = 0
  // pre = input -> output = tanh(input)
  let cell =
    RnnCell(
      input_size: 2,
      hidden_size: 2,
      w_ih: mat([[1.0, 0.0], [0.0, 1.0]]),
      w_hh: tensor.zeros([2, 2]),
      b_ih: tensor.zeros([2]),
      b_hh: tensor.zeros([2]),
    )
  let assert Ok(out) =
    rnn_cell_step(cell, vec([0.5, 0.0 -. 0.5]), vec([0.0, 0.0]))
  lists_close(tensor.to_list(out), [tanh_half, 0.0 -. tanh_half], tol, tol)
  |> should.be_true
}

pub fn rnn_cell_shape_error_test() {
  let cell = zero_rnn_cell(3, 2)
  let assert Error(err) = rnn_cell_step(cell, vec([1.0, 2.0]), vec([0.0, 0.0]))
  case err {
    ShapeMismatch(expected: [3], got: [2]) -> True
    _ -> False
  }
  |> should.be_true
}

pub fn rnn_sequence_test() {
  let cell = zero_rnn_cell(2, 2)
  let inputs = [vec([1.0, 2.0]), vec([3.0, 4.0]), vec([5.0, 6.0])]
  let assert Ok(#(all, final_h)) = rnn_sequence(cell, inputs, vec([0.0, 0.0]))
  list.length(all) |> should.equal(3)
  lists_close(tensor.to_list(final_h), [0.0, 0.0], tol, tol) |> should.be_true
}

// --- GRU -------------------------------------------------------------------

pub fn gru_cell_zero_test() {
  // All weights zero -> r = z = sigmoid(0) = 0.5, n = tanh(0) = 0
  // h' = (1 - 0.5) * 0 + 0.5 * h = 0.5 * h
  let cell = zero_gru_cell(2, 3)
  let h0 = vec([2.0, 0.0 -. 4.0, 6.0])
  let assert Ok(out) = gru_cell_step(cell, vec([1.0, 1.0]), h0)
  lists_close(tensor.to_list(out), [1.0, 0.0 -. 2.0, 3.0], tol, tol)
  |> should.be_true
}

pub fn gru_cell_step_test() {
  // input_size=1, hidden_size=1, all weights zero.
  // x = 0, h = 2 -> r=z=0.5, n=0, h' = 0.5*0 + 0.5*2 = 1.0
  let cell = zero_gru_cell(1, 1)
  let assert Ok(out) = gru_cell_step(cell, vec([0.0]), vec([2.0]))
  let assert [v] = tensor.to_list(out)
  floats_close(v, 1.0, tol, tol) |> should.be_true
}

pub fn gru_sequence_test() {
  let cell = zero_gru_cell(1, 1)
  let inputs = [vec([0.0]), vec([0.0]), vec([0.0])]
  let assert Ok(#(all, final_h)) = gru_sequence(cell, inputs, vec([8.0]))
  list.length(all) |> should.equal(3)
  // Each step halves the hidden state: 8 -> 4 -> 2 -> 1
  let assert [v] = tensor.to_list(final_h)
  floats_close(v, 1.0, tol, tol) |> should.be_true
}

// --- LSTM ------------------------------------------------------------------

pub fn lstm_cell_zero_test() {
  // All weights zero. i = f = o = 0.5, g = 0.
  // c' = 0.5 * c + 0.5 * 0 = 0.5 * c
  // h' = 0.5 * tanh(c')
  let cell = zero_lstm_cell(2, 1)
  let assert Ok(#(h_new, c_new)) =
    lstm_cell_step(cell, vec([1.0, 1.0]), vec([0.0]), vec([2.0]))
  let assert [c_val] = tensor.to_list(c_new)
  floats_close(c_val, 1.0, tol, tol) |> should.be_true
  let assert [h_val] = tensor.to_list(h_new)
  floats_close(h_val, lstm_zero_h, tol, tol) |> should.be_true
}

pub fn lstm_cell_step_test() {
  // Same as zero test but with input_size=1, hidden_size=1.
  // x=0, h=0, c=2 -> c'=1.0, h'=0.5*tanh(1.0)
  let cell = zero_lstm_cell(1, 1)
  let assert Ok(#(h_new, c_new)) =
    lstm_cell_step(cell, vec([0.0]), vec([0.0]), vec([2.0]))
  let assert [c_val] = tensor.to_list(c_new)
  floats_close(c_val, 1.0, tol, tol) |> should.be_true
  let assert [h_val] = tensor.to_list(h_new)
  floats_close(h_val, 0.5 *. tanh_one, tol, tol) |> should.be_true
}

pub fn lstm_sequence_test() {
  // All weights zero, input always [0]. Cell evolves as c_{t+1} = 0.5 * c_t.
  // c0 = 4 -> 2 -> 1 -> 0.5
  let cell = zero_lstm_cell(1, 1)
  let inputs = [vec([0.0]), vec([0.0]), vec([0.0])]
  let assert Ok(#(all_h, _final_h, final_c)) =
    lstm_sequence(cell, inputs, vec([0.0]), vec([4.0]))
  list.length(all_h) |> should.equal(3)
  let assert [c_val] = tensor.to_list(final_c)
  floats_close(c_val, 0.5, tol, tol) |> should.be_true
}
