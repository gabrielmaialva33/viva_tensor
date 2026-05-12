//// Autograd - Reverse-Mode Automatic Differentiation
////
//// "The chain rule is the unsung hero of machine learning."
////   — Every ML practitioner who debugged NaN gradients at 3am
////
//// Implements reverse-mode AD (Speelpenning, 1980) with an explicit tape.
//// Why reverse-mode? Because we have few outputs (loss) and many inputs (params).
//// Forward-mode would require O(n) passes; reverse needs just one. Math wins.
////
//// References:
//// - Speelpenning, B. (1980). "Compiling Fast Partial Derivatives of Functions
////   Given by Algorithms." PhD thesis, UIUC. The OG automatic differentiation.
//// - Baydin et al. (2018). "Automatic Differentiation in Machine Learning: a Survey"
////   https://arxiv.org/abs/1502.05767 - If you read one AD paper, make it this one.
//// - Paszke et al. (2017). "Automatic differentiation in PyTorch" - Dynamic graphs done right.
////
//// Design choice: Explicit tape > implicit global graph. Fight me.
//// PyTorch uses dynamic graphs because Chainer proved it works (Tokui et al., 2015).
//// We take it further: the tape is a value you pass around. Pure FP, no spooky action.
////
//// The math that makes it all work:
////   Chain rule: dL/dx = dL/dy * dy/dx
////   In reverse-mode, we propagate dL/dy backward, accumulating dL/dx.
////
//// ## Key Concepts
////
//// - **Tape**: The computation graph. Records ops for the backward pass.
//// - **Variable**: A tensor with an identity. It knows who it is in the graph.
//// - **Traced(a)**: State monad in disguise. Carries the value AND updated tape.
////
//// ## Example
////
//// ```gleam
//// import viva_tensor/core/tensor
//// import viva_tensor/nn/autograd.{Traced}
////
//// let tape = autograd.new_tape()
//// let Traced(x, tape1) = autograd.new_variable(tape, tensor.from_list([2.0]))
//// let Traced(y, tape2) = autograd.new_variable(tape1, tensor.from_list([3.0]))
////
//// use Traced(z, tape3) <- result.try(autograd.mul(tape2, x, y))
//// use grads <- result.try(autograd.backward(tape3, z))
////
//// // dz/dx = y = 3.0  (partial derivative w.r.t. first input)
//// // dz/dy = x = 2.0  (partial derivative w.r.t. second input)
//// let dx = dict.get(grads, x.id)
//// let dy = dict.get(grads, y.id)
//// ```

import gleam/dict.{type Dict}
import gleam/int
import gleam/list
import gleam/result
import gleam/string
import viva_tensor/core/error.{type TensorError, DimensionError}
import viva_tensor/core/ffi
import viva_tensor/core/ops
import viva_tensor/core/tensor.{type Tensor}
import viva_tensor/telemetry

// -------------------------------------------------------------------------
// Core Types - The Building Blocks of Differentiation
// -------------------------------------------------------------------------

/// Unique identifier for each node in the computational graph.
/// Sequential IDs give us implicit topological ordering for free.
/// Sometimes the simplest solution is the best one.
pub type NodeId =
  Int

/// The closure that computes gradients for parent nodes.
/// Given dL/dself, returns [(parent_id, dL/dparent), ...].
/// This is where the chain rule lives.
pub type BackwardFn =
  fn(Tensor) -> Result(List(#(NodeId, Tensor)), TensorError)

/// The Tape: our explicit computation graph.
///
/// Unlike PyTorch's implicit global state, we pass this around explicitly.
/// Functional programming purists rejoice. Debugging becomes tractable.
/// Trade-off: slightly more verbose code, but no hidden state surprises.
pub type Tape {
  Tape(
    next_id: NodeId,
    /// Maps node ID -> backward function that computes parent gradients.
    /// Only non-leaf nodes have entries here.
    operations: Dict(NodeId, BackwardFn),
  )
}

/// A variable tracked in the autograd system.
/// Think of it as a tensor that remembers its place in the computation graph.
pub type Variable {
  Variable(id: NodeId, data: Tensor)
}

/// The result of a traced operation: value + updated tape.
///
/// This is secretly a State monad: State s a = s -> (a, s)
/// We just make the state threading explicit. Gleam doesn't have do-notation,
/// so explicit is actually clearer here.
pub type Traced(a) {
  Traced(value: a, tape: Tape)
}

// -------------------------------------------------------------------------
// Tape Management - Where Gradients Begin Their Journey
// -------------------------------------------------------------------------

/// Creates a fresh tape. The beginning of every gradient computation.
pub fn new_tape() -> Tape {
  Tape(next_id: 0, operations: dict.new())
}

/// Registers a new variable (leaf node) in the graph.
/// Leaf nodes have no backward function - they're where gradients accumulate.
pub fn new_variable(tape: Tape, data: Tensor) -> Traced(Variable) {
  let id = tape.next_id
  let var = Variable(id: id, data: data)
  let new_tape = Tape(..tape, next_id: id + 1)
  Traced(value: var, tape: new_tape)
}

// -------------------------------------------------------------------------
// Traced Operations - Forward Pass with Gradient Recording
// -------------------------------------------------------------------------
//
// Each operation does two things:
// 1. Compute the forward result (the easy part)
// 2. Register a backward function (the chain rule part)
//
// The backward function captures the inputs in a closure.
// When we call backward(), these closures unwind the computation.

/// Operation sequencing (monadic bind, essentially).
/// Allows chaining: x |> sequence(layer1) |> sequence(layer2)
///
/// This is >>= from Haskell, but we call it sequence because
/// Gleam users shouldn't need a category theory PhD to read the code.
pub fn sequence(
  input: Result(Traced(Variable), e),
  layer_fn: fn(Tape, Variable) -> Result(Traced(Variable), e),
) -> Result(Traced(Variable), e) {
  use Traced(var, tape) <- result.try(input)
  layer_fn(tape, var)
}

/// Traced addition: c = a + b
///
/// Gradient: dc/da = 1, dc/db = 1
/// With broadcasting, we sum over expanded dimensions.
/// This is trickier than it looks - broadcasting gradients must reduce back.
pub fn add(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.add_broadcast(a.data, b.data))

  let res_id = tape.next_id
  let a_shape = tensor.shape(a.data)
  let b_shape = tensor.shape(b.data)

  // Backward: y = a + b => dy/da = 1, dy/db = 1
  // But if we broadcast, grad_a might be larger than a.
  // We need to sum over the broadcast dimensions to match shapes.
  let backward = fn(grad: Tensor) {
    use grad_a <- result.try(sum_to_shape(grad, a_shape))
    use grad_b <- result.try(sum_to_shape(grad, b_shape))

    Ok([#(a.id, grad_a), #(b.id, grad_b)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced subtraction: c = a - b
///
/// Gradient: dc/da = 1, dc/db = -1
/// Subtraction is just addition with a sign flip. Simple, elegant.
pub fn sub(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.sub(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = a - b => dy/da = 1*grad, dy/db = -1*grad
  let backward = fn(grad: Tensor) {
    let neg_grad = ops.negate(grad)
    Ok([#(a.id, grad), #(b.id, neg_grad)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced element-wise multiplication: c = a * b (Hadamard product)
///
/// Gradient: dc/da = b, dc/db = a
/// The classic product rule: d(uv) = u*dv + v*du
pub fn mul(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.mul_broadcast(a.data, b.data))

  let res_id = tape.next_id
  let a_shape = tensor.shape(a.data)
  let b_shape = tensor.shape(b.data)

  // Backward: y = a * b => dy/da = b * grad, dy/db = a * grad
  // Product rule, meet chain rule. They get along well.
  let backward = fn(grad: Tensor) {
    use grad_a_full <- result.try(ops.mul_broadcast(grad, b.data))
    use grad_b_full <- result.try(ops.mul_broadcast(grad, a.data))
    use grad_a <- result.try(sum_to_shape(grad_a_full, a_shape))
    use grad_b <- result.try(sum_to_shape(grad_b_full, b_shape))
    Ok([#(a.id, grad_a), #(b.id, grad_b)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced mean reduction: y = mean(x)
///
/// Gradient: dy/dx_i = 1/n for all i
/// The gradient "fans out" uniformly to all inputs.
/// This is why mean loss converges more stably than sum loss.
pub fn mean(tape: Tape, a: Variable) -> Traced(Variable) {
  let val = ops.mean(a.data)
  let res_data = tensor.from_list([val])

  let res_id = tape.next_id
  let a_shape = tensor.shape(a.data)

  // Backward: y = sum(x) / n => dy/dx = (1/n) * grad
  // grad is scalar, we expand it to input shape divided by n
  let backward = fn(grad: Tensor) {
    let n = tensor.size(a.data) |> int.to_float
    let grad_val = tensor.to_list(grad) |> list.first |> result.unwrap(1.0)
    let scaled_grad_val = grad_val /. n

    let grad_input = tensor.fill(a_shape, scaled_grad_val)
    Ok([#(a.id, grad_input)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

/// Traced matrix multiplication: C = A @ B
///
/// Gradients (the beautiful part):
///   dL/dA = dL/dC @ B^T
///   dL/dB = A^T @ dL/dC
///
/// This is why linear algebra and calculus are best friends.
/// The transpose "reverses" the dimension matching from the forward pass.
pub fn matmul(
  tape: Tape,
  a: Variable,
  b: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.matmul_auto(a.data, b.data))

  let res_id = tape.next_id

  // Backward: y = A @ B
  // dy/dA = grad @ B^T  (dims: [m,n] @ [n,k]^T = [m,k] @ [k,n] = [m,n])
  // dy/dB = A^T @ grad  (dims: [m,k]^T @ [m,n] = [k,m] @ [m,n] = [k,n])
  let backward = fn(grad: Tensor) {
    use bt <- result.try(ops.transpose(b.data))
    use at <- result.try(ops.transpose(a.data))

    use grad_a <- result.try(ops.matmul_auto(grad, bt))
    use grad_b <- result.try(ops.matmul_auto(at, grad))

    Ok([#(a.id, grad_a), #(b.id, grad_b)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced transpose: B = A^T
///
/// Gradient: dL/dA = (dL/dB)^T
/// Transpose is its own inverse. Elegant symmetry.
pub fn transpose(
  tape: Tape,
  a: Variable,
) -> Result(Traced(Variable), TensorError) {
  use res_data <- result.try(ops.transpose(a.data))

  let res_id = tape.next_id

  // Backward: y = A^T => dy/dA = grad^T
  // The Jacobian of transpose is... transpose. Beautiful.
  let backward = fn(grad: Tensor) {
    use grad_t <- result.try(ops.transpose(grad))
    Ok([#(a.id, grad_t)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Traced ReLU activation: y = max(0, x)
///
/// Gradient: dy/dx = 1 if x > 0, else 0
///
/// The "dying ReLU" problem lives here: once a neuron outputs 0,
/// its gradient is 0, so it never learns again. RIP that neuron.
/// Leaky ReLU fixes this, but plain ReLU is still surprisingly effective.
pub fn relu(tape: Tape, a: Variable) -> Traced(Variable) {
  let res_data =
    ops.map(a.data, fn(x) {
      case x >. 0.0 {
        True -> x
        False -> 0.0
      }
    })

  let res_id = tape.next_id

  // Backward: y = relu(x) => dy/dx = indicator(x > 0)
  // This is a subgradient at x=0, but who's counting?
  let backward = fn(grad: Tensor) {
    let mask =
      ops.map(a.data, fn(x) {
        case x >. 0.0 {
          True -> 1.0
          False -> 0.0
        }
      })
    use grad_a <- result.try(ops.mul_auto(grad, mask))
    Ok([#(a.id, grad_a)])
  }

  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)

  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

// -------------------------------------------------------------------------
// Backpropagation Engine - Where Gradients Flow Uphill
// -------------------------------------------------------------------------
//
// "Backprop is just the chain rule applied recursively."
//   — Everyone who's ever explained backprop
//
// We traverse the graph in reverse topological order (newest to oldest).
// For each node, we compute dL/d(node) and propagate to its parents.
// Gradients accumulate when a node has multiple children (sum rule).

/// Executes backpropagation starting from a loss variable.
/// Returns gradients for all variables: Map(NodeId -> Tensor).
/// Instrumented: records graph size and backward pass latency.
///
/// The loss should be a scalar (or we treat it as sum of elements).
/// Multi-output differentiation is possible but rarely needed in ML.
pub fn backward(
  tape: Tape,
  loss: Variable,
) -> Result(Dict(NodeId, Tensor), TensorError) {
  let t0 = ffi.now_microseconds()
  // Seed gradient: dL/dL = 1.0 (the journey begins)
  let loss_shape = tensor.shape(loss.data)
  let initial_grad = tensor.ones(loss_shape)
  let initial_grads = dict.from_list([#(loss.id, initial_grad)])

  // Process nodes in reverse creation order.
  // Since IDs are sequential, this IS topological order.
  // No need for Kahn's algorithm or DFS - the tape gives it to us free.
  let all_ids = list.range(tape.next_id - 1, 0)

  use final_grads <- result.try(
    list.fold(all_ids, Ok(initial_grads), fn(grads_result, current_id) {
      use grads <- result.try(grads_result)
      case dict.get(grads, current_id) {
        // Node doesn't contribute to loss (not on any path to loss)
        Error(_) -> Ok(grads)
        Ok(current_grad) -> {
          case dict.get(tape.operations, current_id) {
            // Leaf node: no parents, gradient just accumulates here
            Error(_) -> Ok(grads)
            // Interior node: propagate gradient to parents via chain rule
            Ok(back_fn) -> {
              use parent_grads <- result.try(back_fn(current_grad))

              // Accumulate gradients (multivariate chain rule: sum contributions)
              list.fold(parent_grads, Ok(grads), fn(acc_result, pair) {
                use acc_grads <- result.try(acc_result)
                let #(pid, pgrad) = pair
                case dict.get(acc_grads, pid) {
                  Error(_) -> Ok(dict.insert(acc_grads, pid, pgrad))
                  Ok(existing) -> {
                    // Shape mismatch here means we have a bug in backward functions
                    let existing_shape = tensor.shape(existing)
                    let pgrad_shape = tensor.shape(pgrad)
                    case existing_shape == pgrad_shape {
                      True ->
                        ops.add_auto(existing, pgrad)
                        |> result.map(fn(sum) {
                          dict.insert(acc_grads, pid, sum)
                        })
                      False -> {
                        let msg =
                          "Gradient shape mismatch at node "
                          <> int.to_string(pid)
                          <> ": existing="
                          <> string_shape(existing_shape)
                          <> ", incoming="
                          <> string_shape(pgrad_shape)
                          <> ". This is a bug in the backward function."
                        Error(DimensionError(msg))
                      }
                    }
                  }
                }
              })
            }
          }
        }
      }
    }),
  )

  telemetry.record_backward(tape.next_id, ffi.now_microseconds() - t0)
  Ok(final_grads)
}

// -------------------------------------------------------------------------
// Internal Helpers - The Unglamorous but Necessary Parts
// -------------------------------------------------------------------------

fn string_shape(shape: List(Int)) -> String {
  "[" <> string.join(list.map(shape, int.to_string), with: ", ") <> "]"
}

/// Sum tensor to match target shape (for broadcast gradient reduction).
///
/// When we broadcast [3] to [2,3] in the forward pass,
/// we must sum [2,3] gradients back to [3] in backward.
/// This is the "reverse of broadcasting."
fn sum_to_shape(
  t: Tensor,
  target_shape: List(Int),
) -> Result(Tensor, TensorError) {
  let t_shape = tensor.shape(t)

  case t_shape == target_shape {
    True -> Ok(t)
    False -> {
      let rank_diff = list.length(t_shape) - list.length(target_shape)
      case rank_diff < 0 || !ops.can_broadcast(target_shape, t_shape) {
        True ->
          Error(DimensionError(
            "Cannot reduce gradient shape "
            <> string_shape(t_shape)
            <> " to broadcast source shape "
            <> string_shape(target_shape),
          ))
        False -> {
          let t_data = tensor.to_list(t)
          let t_size = tensor.size(t)
          let target_size = list.fold(target_shape, 1, fn(acc, d) { acc * d })
          let padded_target_shape =
            list.append(list.repeat(1, rank_diff), target_shape)

          let accum =
            indices(t_size)
            |> list.fold(dict.new(), fn(acc, source_flat) {
              let source_indices = flat_to_multi(source_flat, t_shape)
              let target_indices =
                source_to_target_indices(source_indices, padded_target_shape)
                |> list.drop(rank_diff)
              let target_flat = multi_to_flat(target_indices, target_shape)
              let source_value = value_at(t_data, source_flat)

              case dict.get(acc, target_flat) {
                Ok(current) ->
                  dict.insert(acc, target_flat, current +. source_value)
                Error(_) -> dict.insert(acc, target_flat, source_value)
              }
            })

          let reduced =
            indices(target_size)
            |> list.map(fn(target_flat) {
              dict.get(accum, target_flat)
              |> result.unwrap(0.0)
            })

          tensor.new(reduced, target_shape)
        }
      }
    }
  }
}

fn source_to_target_indices(
  source_indices: List(Int),
  target_shape: List(Int),
) -> List(Int) {
  list.zip(source_indices, target_shape)
  |> list.map(fn(pair) {
    let #(source_index, target_dim) = pair
    case target_dim == 1 {
      True -> 0
      False -> source_index
    }
  })
}

fn flat_to_multi(flat: Int, shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(indices, _) =
    list.fold(reversed, #([], flat), fn(acc, dim) {
      let #(idxs, remaining) = acc
      let idx = remaining % dim
      let next = remaining / dim
      #([idx, ..idxs], next)
    })
  indices
}

fn multi_to_flat(indices: List(Int), shape: List(Int)) -> Int {
  let strides = compute_strides(shape)
  list.zip(indices, strides)
  |> list.fold(0, fn(acc, pair) {
    let #(idx, stride) = pair
    acc + idx * stride
  })
}

fn compute_strides(shape: List(Int)) -> List(Int) {
  let reversed = list.reverse(shape)
  let #(strides, _) =
    list.fold(reversed, #([], 1), fn(acc, dim) {
      let #(s, running) = acc
      #([running, ..s], running * dim)
    })
  strides
}

fn indices(size: Int) -> List(Int) {
  case size <= 0 {
    True -> []
    False -> list.range(0, size - 1)
  }
}

fn value_at(values: List(Float), index: Int) -> Float {
  values
  |> list.drop(index)
  |> list.first
  |> result.unwrap(0.0)
}
