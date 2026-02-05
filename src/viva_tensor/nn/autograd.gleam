//// Autograd - Automatic Differentiation Engine
////
//// Implements reverse-mode automatic differentiation with an explicit tape.
//// This enables computing gradients for neural network training.
////
//// ## Key Concepts
////
//// - **Tape**: Records operations for backpropagation
//// - **Variable**: A tensor tracked in the computational graph
//// - **Traced(a)**: Result of a traced operation (value + new tape)
////
//// ## Example
////
//// ```gleam
//// import viva_tensor/core/tensor
//// import viva_tensor/nn/autograd.{Traced}
////
//// // Create tape and variables
//// let tape = autograd.new_tape()
//// let Traced(x, tape1) = autograd.new_variable(tape, tensor.from_list([2.0]))
//// let Traced(y, tape2) = autograd.new_variable(tape1, tensor.from_list([3.0]))
////
//// // Perform traced operations
//// let assert Ok(Traced(z, tape3)) = autograd.mul(tape2, x, y)
////
//// // Compute gradients via backpropagation
//// let assert Ok(grads) = autograd.backward(tape3, z)
////
//// // dz/dx = y = 3.0
//// let assert Ok(dx) = dict.get(grads, x.id)
//// // dz/dy = x = 2.0
//// let assert Ok(dy) = dict.get(grads, y.id)
//// ```

import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/result
import gleam/string
import viva_tensor/core/error.{type TensorError}
import viva_tensor/core/ops
import viva_tensor/core/tensor.{type Tensor}

/// Unique identifier for each node in the computational graph
pub type NodeId =
  Int

/// Function that calculates parent gradients based on this node's gradient
pub type BackwardFn =
  fn(Tensor) -> List(#(NodeId, Tensor))

/// The Tape maintains the history of operations in the computational graph.
/// Unlike PyTorch (global/mutable), here the Tape is an explicit immutable value.
pub type Tape {
  Tape(
    next_id: NodeId,
    // Maps resulting node ID -> Closure that computes gradients for parents
    operations: Dict(NodeId, BackwardFn),
  )
}

/// A variable tracked in the autograd system
pub type Variable {
  Variable(id: NodeId, data: Tensor)
}

/// The result of a traced operation.
/// Encapsulates the produced value (e.g., Variable) and the new tape state.
/// Analogous to a State Monad.
pub type Traced(a) {
  Traced(value: a, tape: Tape)
}

/// Creates a new empty tape
pub fn new_tape() -> Tape {
  Tape(next_id: 0, operations: dict.new())
}

/// Registers a new variable (leaf node) in the graph
pub fn new_variable(tape: Tape, data: Tensor) -> Traced(Variable) {
  let id = tape.next_id
  let var = Variable(id: id, data: data)
  let new_tape = Tape(..tape, next_id: id + 1)
  Traced(value: var, tape: new_tape)
}

// =============================================================================
// TRACED OPERATIONS
// =============================================================================

/// Operation sequencing (Monadic Pipe)
/// Allows chaining layers: x |> sequence(layer1) |> sequence(layer2)
pub fn sequence(
  input: Result(Traced(Variable), e),
  layer_fn: fn(Tape, Variable) -> Result(Traced(Variable), e),
) -> Result(Traced(Variable), e) {
  use Traced(var, tape) <- result.try(input)
  layer_fn(tape, var)
}

/// Traced addition: c = a + b (supports broadcasting)
pub fn add(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.add_broadcast(a.data, b.data))

  let res_id = tape.next_id
  let a_shape = tensor.shape(a.data)
  let b_shape = tensor.shape(b.data)

  // Backward: y = a + b
  // If broadcasting occurred, we need to sum gradients over expanded dimensions
  let backward = fn(grad: Tensor) {
    let grad_shape = tensor.shape(grad)
    let grad_a = case grad_shape == a_shape {
      True -> grad
      False -> {
        // Simplified: sum over first axis for broadcast reduction
        sum_to_shape(grad, a_shape)
      }
    }

    let grad_b = case grad_shape == b_shape {
      True -> grad
      False -> {
        sum_to_shape(grad, b_shape)
      }
    }

    [#(a.id, grad_a), #(b.id, grad_b)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced subtraction: c = a - b
pub fn sub(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.sub(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a - b
  // dy/da = 1 * grad
  // dy/db = -1 * grad
  let backward = fn(grad: Tensor) {
    let neg_grad = ops.negate(grad)
    [#(a.id, grad), #(b.id, neg_grad)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced Element-wise Multiplication: c = a * b
pub fn mul(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.mul(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a * b
  // dy/da = b * grad
  // dy/db = a * grad
  let backward = fn(grad: Tensor) {
    let assert Ok(grad_a) = ops.mul(grad, b.data)
    let assert Ok(grad_b) = ops.mul(grad, a.data)
    [#(a.id, grad_a), #(b.id, grad_b)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced Mean (Reduce Mean): y = mean(x)
/// Returns a scalar Tensor (rank 0 or 1 depending on base implementation, here we force 1)
pub fn mean(tape: Tape, a: Variable) -> Traced(Variable) {
  let val = ops.mean(a.data)
  let res_data = tensor.from_list([val])

  let res_id = tape.next_id
  let a_shape = tensor.shape(a.data)

  // Backward: y = sum(x) / n
  // dy/dx = (1/n) * grad
  // The input gradient (grad) is a scalar (or 1-element tensor)
  // We need to expand it to x's shape and divide by n
  let backward = fn(grad: Tensor) {
    let n = tensor.size(a.data) |> int.to_float
    let grad_val = tensor.to_list(grad) |> list.first |> result.unwrap(1.0)
    let scaled_grad_val = grad_val /. n

    // Creates a filled tensor with the scaled gradient value
    let grad_input = tensor.fill(a_shape, scaled_grad_val)
    [#(a.id, grad_input)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

/// Traced Matrix Multiplication: c = a @ b
pub fn matmul(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.matmul(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a @ b
  // dy/da = grad @ b.T
  // dy/db = a.T @ grad
  let backward = fn(grad: Tensor) {
    let assert Ok(bt) = ops.transpose(b.data)
    let assert Ok(at) = ops.transpose(a.data)

    let assert Ok(grad_a) = ops.matmul(grad, bt)
    let assert Ok(grad_b) = ops.matmul(at, grad)

    [#(a.id, grad_a), #(b.id, grad_b)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced Transpose: c = a.T
pub fn transpose(
  tape: Tape,
  a: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.transpose(a.data))

  let res_id = tape.next_id

  // Backward: y = a.T => dy/da = grad.T
  let backward = fn(grad: Tensor) {
    let assert Ok(grad_t) = ops.transpose(grad)
    [#(a.id, grad_t)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced ReLU
pub fn relu(tape: Tape, a: Variable) -> Traced(Variable) {
  let res_data =
    ops.map(a.data, fn(x) {
      case x >. 0.0 {
        True -> x
        False -> 0.0
      }
    })

  let res_id = tape.next_id

  // Backward: y = relu(a) => dy/da = 1 if a > 0 else 0
  let backward = fn(grad: Tensor) {
    let mask =
      ops.map(a.data, fn(x) {
        case x >. 0.0 {
          True -> 1.0
          False -> 0.0
        }
      })
    let assert Ok(grad_a) = ops.mul(grad, mask)
    [#(a.id, grad_a)]
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

// =============================================================================
// BACKPROPAGATION ENGINE
// =============================================================================

/// Executes backpropagation starting from a scalar variable (loss).
/// Returns a Map of NodeId -> Gradient (Tensor)
pub fn backward(
  tape: Tape,
  loss: Variable,
) -> Result(Dict(NodeId, Tensor), String) {
  // Initial gradient dLoss/dLoss = 1.0
  let loss_shape = tensor.shape(loss.data)
  let initial_grad = tensor.ones(loss_shape)
  let initial_grads = dict.from_list([#(loss.id, initial_grad)])

  // Process nodes in reverse creation order (Implicit Topological Sort)
  // IDs are sequential, so (next_id - 1) down to 0 ensures correct order.
  let all_ids = list.range(tape.next_id - 1, 0)

  let final_grads =
    list.fold(all_ids, initial_grads, fn(grads, current_id) {
      case dict.get(grads, current_id) {
        Error(_) -> grads
        // Node does not contribute to loss or hasn't been computed
        Ok(current_grad) -> {
          // If this node has an operation registered on the tape, expand gradient
          case dict.get(tape.operations, current_id) {
            Error(_) -> grads
            // Leaf node (input), no parents to propagate to
            Ok(back_fn) -> {
              let parent_grads = back_fn(current_grad)

              // Accumulate gradients in parents (sum if gradient already exists)
              list.fold(parent_grads, grads, fn(acc_grads, pair) {
                let #(pid, pgrad) = pair
                case dict.get(acc_grads, pid) {
                  Error(_) -> dict.insert(acc_grads, pid, pgrad)
                  Ok(existing) -> {
                    let existing_shape = tensor.shape(existing)
                    let pgrad_shape = tensor.shape(pgrad)
                    case existing_shape == pgrad_shape {
                      True -> {
                        let assert Ok(sum) = ops.add(existing, pgrad)
                        dict.insert(acc_grads, pid, sum)
                      }
                      False -> {
                        let msg =
                          "ShapeMismatch at node "
                          <> int.to_string(pid)
                          <> ": existing="
                          <> string_shape(existing_shape)
                          <> ", new="
                          <> string_shape(pgrad_shape)
                        panic as msg
                      }
                    }
                  }
                }
              })
            }
          }
        }
      }
    })

  Ok(final_grads)
}

// =============================================================================
// INTERNAL HELPERS
// =============================================================================

fn string_shape(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), with: ", ") <> "]"
}

/// Sum tensor to match target shape (for broadcast gradient reduction)
fn sum_to_shape(t: Tensor, target_shape: List(Int)) -> Tensor {
  let _data = tensor.to_list(t)
  let t_shape = tensor.shape(t)

  case t_shape == target_shape {
    True -> t
    False -> {
      // Simple reduction: sum all elements and fill target shape
      let total = ops.sum(t)
      let target_size = list.fold(target_shape, 1, fn(acc, d) { acc * d })
      let avg = total /. int.to_float(target_size)
      tensor.fill(target_shape, avg)
    }
  }
}
