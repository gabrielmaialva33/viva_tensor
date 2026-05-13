//// Tests for `viva_tensor/nn/activations`.
////
//// Tolerances follow NumPy: `rtol=1e-5, atol=1e-7` is enough headroom for
//// our pure-Gleam `exp`/`tanh`/`erf` implementations and the inevitable
//// rounding in chained float ops.

import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/nn/activations
import viva_tensor/tensor.{Tensor}

pub fn main() -> Nil {
  gleeunit.main()
}

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-7

// --- sigmoid ----------------------------------------------------------------

pub fn sigmoid_test() {
  let t = Tensor([-50.0, 0.0, 50.0], [3])
  let out = activations.sigmoid(t)
  case tensor.to_list(out) {
    [neg, mid, pos] -> {
      numerics.floats_close(mid, 0.5, rtol, atol) |> should.be_true()
      // saturated tails — tolerant atol because exp(50) is huge
      numerics.floats_close(neg, 0.0, rtol, 1.0e-6) |> should.be_true()
      numerics.floats_close(pos, 1.0, rtol, 1.0e-6) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- tanh -------------------------------------------------------------------

pub fn tanh_test() {
  let t = Tensor([-50.0, 0.0, 50.0], [3])
  let out = activations.tanh(t)
  case tensor.to_list(out) {
    [neg, mid, pos] -> {
      numerics.floats_close(mid, 0.0, rtol, atol) |> should.be_true()
      numerics.floats_close(neg, -1.0, rtol, 1.0e-6) |> should.be_true()
      numerics.floats_close(pos, 1.0, rtol, 1.0e-6) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- relu -------------------------------------------------------------------

pub fn relu_test() {
  let t = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], [5])
  let out = activations.relu(t)
  tensor.to_list(out) |> should.equal([0.0, 0.0, 0.0, 0.5, 2.0])
}

// --- leaky_relu -------------------------------------------------------------

pub fn leaky_relu_test() {
  let t = Tensor([-2.0, 0.0, 3.0], [3])
  let out = activations.leaky_relu(t, 0.01)
  numerics.lists_close(tensor.to_list(out), [-0.02, 0.0, 3.0], rtol, atol)
}

// --- elu --------------------------------------------------------------------

pub fn elu_test() {
  let t = Tensor([-1.0, 0.0, 2.0], [3])
  let out = activations.elu(t, 1.0)
  // elu(-1, 1) = 1*(exp(-1)-1) ~ -0.6321205588
  // elu(0, 1) = 0 (since 0 is not > 0, falls into 1*(exp(0)-1) = 0)
  numerics.lists_close(
    tensor.to_list(out),
    [-0.6321205588285577, 0.0, 2.0],
    rtol,
    atol,
  )
}

// --- selu -------------------------------------------------------------------

pub fn selu_test() {
  let t = Tensor([0.0, 1.0], [2])
  let out = activations.selu(t)
  // selu(0) = scale * alpha * (exp(0) - 1) = 0
  // selu(1) = scale * 1 = 1.0507009873554804
  numerics.lists_close(
    tensor.to_list(out),
    [0.0, 1.0507009873554804934193349852946],
    rtol,
    atol,
  )
}

// --- gelu -------------------------------------------------------------------

pub fn gelu_test() {
  let t = Tensor([0.0, 1.0], [2])
  let out = activations.gelu(t)
  case tensor.to_list(out) {
    [g0, g1] -> {
      numerics.floats_close(g0, 0.0, rtol, atol) |> should.be_true()
      // gelu(1) ~ 0.8413447460685429 with exact erf-based formula.
      numerics.floats_close(g1, 0.8413447460685429, 1.0e-3, 1.0e-4)
      |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- swish ------------------------------------------------------------------

pub fn swish_test() {
  let t = Tensor([0.0, 1.0], [2])
  let out = activations.swish(t)
  // swish(0) = 0 * 0.5 = 0
  // swish(1) = 1 * sigmoid(1) = 1 / (1 + exp(-1)) ~ 0.7310585786300049
  numerics.lists_close(
    tensor.to_list(out),
    [0.0, 0.7310585786300049],
    rtol,
    atol,
  )
}

// --- mish -------------------------------------------------------------------

pub fn mish_test() {
  let t = Tensor([0.0, 1.0], [2])
  let out = activations.mish(t)
  // mish(0) = 0 * tanh(log(2)) = 0
  // mish(1) = 1 * tanh(softplus(1)) = tanh(log(1+e)) ~ 0.8650983882673103
  case tensor.to_list(out) {
    [m0, m1] -> {
      numerics.floats_close(m0, 0.0, rtol, atol) |> should.be_true()
      numerics.floats_close(m1, 0.8650983882673103, 1.0e-4, 1.0e-5)
      |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- softplus ---------------------------------------------------------------

pub fn softplus_test() {
  let t = Tensor([-50.0, 0.0, 50.0], [3])
  let out = activations.softplus(t)
  case tensor.to_list(out) {
    [neg, mid, pos] -> {
      // softplus(0) = log(2) ~ 0.6931471805599453
      numerics.floats_close(mid, 0.6931471805599453, rtol, atol)
      |> should.be_true()
      // softplus(-50) ~ 0 (no underflow because stable formulation)
      numerics.floats_close(neg, 0.0, rtol, 1.0e-6) |> should.be_true()
      // softplus(50) ~ 50 (large x: behaves like x)
      numerics.floats_close(pos, 50.0, rtol, 1.0e-6) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- softmax ----------------------------------------------------------------

pub fn softmax_1d_test() {
  let t = Tensor([1.0, 2.0, 3.0], [3])
  let out = case activations.softmax(t, 0) {
    Ok(x) -> x
    Error(_) -> panic as "softmax failed"
  }
  let values = tensor.to_list(out)
  let total = list.fold(values, 0.0, fn(acc, v) { acc +. v })
  numerics.floats_close(total, 1.0, rtol, atol) |> should.be_true()
}

pub fn softmax_axis0_2d_test() {
  // shape [2, 3] - softmax along axis 0 means each column sums to 1
  let t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let out = case activations.softmax(t, 0) {
    Ok(x) -> x
    Error(_) -> panic as "softmax axis 0 failed"
  }
  let values = tensor.to_list(out)
  case values {
    [a, b, c, d, e, f] -> {
      numerics.floats_close(a +. d, 1.0, rtol, atol) |> should.be_true()
      numerics.floats_close(b +. e, 1.0, rtol, atol) |> should.be_true()
      numerics.floats_close(c +. f, 1.0, rtol, atol) |> should.be_true()
    }
    _ -> should.fail()
  }
}

pub fn softmax_axis1_2d_test() {
  // shape [2, 3] - softmax along axis 1 means each row sums to 1
  let t = Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
  let out = case activations.softmax(t, 1) {
    Ok(x) -> x
    Error(_) -> panic as "softmax axis 1 failed"
  }
  let values = tensor.to_list(out)
  case values {
    [a, b, c, d, e, f] -> {
      numerics.floats_close(a +. b +. c, 1.0, rtol, atol) |> should.be_true()
      numerics.floats_close(d +. e +. f, 1.0, rtol, atol) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- log_softmax ------------------------------------------------------------

pub fn log_softmax_test() {
  let t = Tensor([1.0, 2.0, 3.0, 1000.0, 1001.0, 1002.0], [2, 3])
  let out = case activations.log_softmax(t, 1) {
    Ok(x) -> x
    Error(_) -> panic as "log_softmax failed"
  }
  let values = tensor.to_list(out)
  case values {
    [a, b, c, d, e, f] -> {
      // sum(exp(log_softmax)) along axis must be 1, even for huge inputs
      // (numerical stability test).
      let row0 =
        float.exponential(a) +. float.exponential(b) +. float.exponential(c)
      let row1 =
        float.exponential(d) +. float.exponential(e) +. float.exponential(f)
      numerics.floats_close(row0, 1.0, rtol, atol) |> should.be_true()
      numerics.floats_close(row1, 1.0, rtol, atol) |> should.be_true()
    }
    _ -> should.fail()
  }
}

// --- hardswish --------------------------------------------------------------

pub fn hardswish_test() {
  let t = Tensor([-4.0, -3.0, 0.0, 3.0, 6.0], [5])
  let out = activations.hardswish(t)
  // hardswish(-4) = -4 * max(0, min(-1,6)) / 6 = -4 * 0 / 6 = 0
  // hardswish(-3) = -3 * 0 / 6 = 0
  // hardswish(0)  = 0 * 3 / 6 = 0
  // hardswish(3)  = 3 * 6 / 6 = 3
  // hardswish(6)  = 6 * min(9,6) / 6 = 6 * 6 / 6 = 6
  numerics.lists_close(
    tensor.to_list(out),
    [0.0, 0.0, 0.0, 3.0, 6.0],
    rtol,
    atol,
  )
}

// --- hardtanh ---------------------------------------------------------------

pub fn hardtanh_test() {
  let t = Tensor([-2.0, -0.5, 0.0, 0.5, 2.0], [5])
  let out = activations.hardtanh(t, -1.0, 1.0)
  numerics.lists_close(
    tensor.to_list(out),
    [-1.0, -0.5, 0.0, 0.5, 1.0],
    rtol,
    atol,
  )
}
