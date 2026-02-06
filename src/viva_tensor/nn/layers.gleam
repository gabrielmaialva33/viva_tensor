//// Neural Network Layers - The Building Blocks of Deep Learning
////
//// "A neural network is just a differentiable program."
////   — Yann LeCun (paraphrased, but he'd probably agree)
////
//// References:
//// - Rumelhart, Hinton & Williams (1986). "Learning representations by
////   back-propagating errors." Nature. THE paper that started it all.
//// - Glorot & Bengio (2010). "Understanding the difficulty of training
////   deep feedforward neural networks." Xavier initialization lives here.
//// - He et al. (2015). "Delving Deep into Rectifiers." Kaiming init for ReLU.
////
//// Design philosophy:
//// - Layers are values, not classes. No hidden state, no surprises.
//// - Forward pass returns Traced(Variable) - computation graph included.
//// - Everything flows through the tape. Backprop just works.

import gleam/result
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/ffi
import viva_tensor/core/tensor
import viva_tensor/nn/autograd.{type Tape, type Traced, type Variable, Traced}
import viva_tensor/telemetry

// -------------------------------------------------------------------------
// Linear Layer - The Workhorse of Neural Networks
// -------------------------------------------------------------------------
//
// y = xW^T + b
//
// This is an affine transformation: rotation/scaling (W) + translation (b).
// Every MLP, every attention head, every prediction layer uses this.
// Simple math, surprisingly powerful when stacked with nonlinearities.

/// Linear (fully connected) layer.
///
/// Stores weights W: [out_features, in_features] and bias b: [out_features].
/// PyTorch convention: weights are [out, in] for efficiency in y = xW^T.
pub type Linear {
  Linear(w: Variable, b: Variable)
}

/// Creates a new Linear layer with Xavier/Glorot initialization.
///
/// Xavier init: W ~ Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
/// This keeps variance stable across layers for tanh/sigmoid.
/// For ReLU, you'd want He init (scale by sqrt(2/fan_in)) instead.
///
/// The bias starts at zero. Some argue for small positive values
/// to ensure ReLU neurons fire initially, but zero works fine.
pub fn linear(tape: Tape, in_features: Int, out_features: Int) -> Traced(Linear) {
  // Xavier initialization: the sweet spot for gradient flow
  let w_data = tensor.xavier_init(in_features, out_features)
  let b_data = tensor.zeros([out_features])

  let Traced(w, tape1) = autograd.new_variable(tape, w_data)
  let Traced(b, tape2) = autograd.new_variable(tape1, b_data)

  Traced(value: Linear(w, b), tape: tape2)
}

/// Forward pass: y = xW^T + b
/// Instrumented: records forward pass latency.
///
/// The order of operations matters for gradient computation:
/// 1. Transpose W: [out, in] -> [in, out]
/// 2. Matmul: [batch, in] @ [in, out] -> [batch, out]
/// 3. Add bias: [batch, out] + [out] (broadcast over batch dim)
///
/// Each step is traced, so backward() will compute dL/dW, dL/db, dL/dx.
pub fn linear_forward(
  tape: Tape,
  layer: Linear,
  x: Variable,
) -> Result(Traced(Variable), TensorError) {
  let t0 = ffi.now_microseconds()

  // Step 1: Transpose weights
  // PyTorch stores [out, in], we need [in, out] for x @ W^T
  use Traced(wt, tape1) <- result.try(autograd.transpose(tape, layer.w))

  // Step 2: Matrix multiply
  // [batch, in] @ [in, out] -> [batch, out]
  // This is where most of the FLOPs happen
  use Traced(xw, tape2) <- result.try(autograd.matmul(tape1, x, wt))

  // Step 3: Add bias
  // Bias broadcasts over the batch dimension
  // y_i = (xW^T)_i + b for each sample i in the batch
  let result = autograd.add(tape2, xw, layer.b)
  telemetry.record_op("linear_forward", ffi.now_microseconds() - t0)
  result
}

// -------------------------------------------------------------------------
// Activation Functions - The Source of All Nonlinearity
// -------------------------------------------------------------------------
//
// Without activations, a deep network is just one big linear transform.
// stack(Linear, Linear, Linear) = just_another_Linear
// Activations break this linearity, enabling universal approximation.

/// ReLU activation: f(x) = max(0, x)
///
/// The most popular activation function. Simple, effective, sometimes dead.
///
/// Why ReLU wins:
/// - Sparse activations (many zeros = efficient)
/// - No vanishing gradient for positive values (gradient = 1)
/// - Computationally trivial (just a comparison)
///
/// Why ReLU loses:
/// - "Dying ReLU": neurons that output 0 have 0 gradient forever
/// - Unbounded output can cause numerical issues
///
/// Alternatives: LeakyReLU, GELU, SiLU/Swish. Each has tradeoffs.
pub fn relu(tape: Tape, x: Variable) -> Traced(Variable) {
  autograd.relu(tape, x)
}

// -------------------------------------------------------------------------
// Loss Functions - How We Measure Wrongness
// -------------------------------------------------------------------------
//
// The loss function defines what "good" means for our model.
// Backprop minimizes it, so choose wisely.
//
// Common losses:
// - MSE: (pred - target)^2 - Good for regression, smooth gradients
// - Cross-entropy: -sum(target * log(pred)) - Good for classification
// - Huber: MSE near zero, MAE far away - Robust to outliers

/// Mean Squared Error loss: L = mean((pred - target)^2)
///
/// MSE: the L2 norm's favorite child.
///
/// Properties:
/// - Gradient: dL/dpred = 2(pred - target) / n
/// - Strongly convex (unique minimum)
/// - Penalizes large errors quadratically (sensitive to outliers)
/// - The MLE under Gaussian noise assumption
///
/// When to use:
/// - Regression problems with Gaussian-distributed errors
/// - When you want smooth, well-behaved gradients
///
/// When NOT to use:
/// - Outlier-heavy data (use Huber or MAE instead)
/// - Classification (use cross-entropy)
pub fn mse_loss(
  tape: Tape,
  pred: Variable,
  target: Variable,
) -> Result(Traced(Variable), TensorError) {
  // Step 1: diff = pred - target
  // This is where we compute the residuals
  use Traced(diff, tape1) <- result.try(autograd.sub(tape, pred, target))

  // Step 2: square = diff * diff
  // Squaring makes all errors positive and penalizes large ones more
  // Note: diff appears twice, so its gradient gets 2x contribution (product rule)
  use Traced(square, tape2) <- result.try(autograd.mul(tape1, diff, diff))

  // Step 3: loss = mean(square)
  // Averaging normalizes for batch size (important for learning rate tuning)
  Ok(autograd.mean(tape2, square))
}
