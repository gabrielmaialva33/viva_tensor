import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/tensor
import viva_tensor/nn/autograd.{type Tape, type Traced, type Variable, Traced}

/// Linear Layer (Fully Connected)
/// y = x @ W.T + b
pub type Linear {
  Linear(w: Variable, b: Variable)
}

/// Initializes a new Linear layer
pub fn linear(tape: Tape, in_features: Int, out_features: Int) -> Traced(Linear) {
  // Weight initialization (Xavier/Glorot)
  let w_data = tensor.xavier_init(in_features, out_features)
  let b_data = tensor.zeros([out_features])

  let Traced(w, tape1) = autograd.new_variable(tape, w_data)
  let Traced(b, tape2) = autograd.new_variable(tape1, b_data)

  Traced(value: Linear(w, b), tape: tape2)
}

/// Forward pass of the Linear layer
pub fn linear_forward(
  tape: Tape,
  layer: Linear,
  x: Variable,
) -> Result(Traced(Variable), TensorError) {
  // 1. Transpose weights: [out, in] -> [in, out]
  //    Note: PyTorch stores weights as [out, in] for efficiency
  use Traced(wt, tape1) <- result.try(autograd.transpose(tape, layer.w))

  // 2. Matmul: [batch, in] @ [in, out] -> [batch, out]
  use Traced(xw, tape2) <- result.try(autograd.matmul(tape1, x, wt))

  // 3. Add Bias: [batch, out] + [out]
  //    TODO: Implement real broadcast in autograd.add. 
  //    For now, assumes backend supports it or shapes are compatible.
  autograd.add(tape2, xw, layer.b)
}

/// ReLU activation function
pub fn relu(tape: Tape, x: Variable) -> Traced(Variable) {
  autograd.relu(tape, x)
}

/// Loss function: Mean Squared Error (MSE)
/// L = mean((pred - target)^2)
pub fn mse_loss(
  tape: Tape,
  pred: Variable,
  target: Variable,
) -> Result(Traced(Variable), TensorError) {
  // 1. diff = pred - target
  use Traced(diff, tape1) <- result.try(autograd.sub(tape, pred, target))

  // 2. square = diff * diff
  use Traced(square, tape2) <- result.try(autograd.mul(tape1, diff, diff))

  // 3. loss = mean(square)
  Ok(autograd.mean(tape2, square))
}
