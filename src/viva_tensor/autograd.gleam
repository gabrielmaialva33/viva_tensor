import gleam/dict.{type Dict}
import gleam/list
import gleam/result
import gleam/int
import viva_tensor/tensor.{type Tensor}

/// Identificador único para cada nó no grafo computacional
pub type NodeId =
  Int

/// Função que calcula o gradiente dos pais com base no gradiente deste nó
pub type BackwardFn =
  fn(Tensor) -> List(#(NodeId, Tensor))

/// A Fita (Tape) mantém o histórico de operações do grafo computacional.
/// Diferente do PyTorch (global/mutável), aqui a Tape é um valor imutável explícito.
pub type Tape {
  Tape(
    next_id: NodeId,
    // Mapeia ID do nó resultante -> Closure que calcula gradientes dos pais
    operations: Dict(NodeId, BackwardFn),
  )
}

/// Uma variável rastreada no sistema de autograd
pub type Variable {
  Variable(id: NodeId, data: Tensor)
}

/// O resultado de uma operação rastreada.
/// Encapsula o valor produzido (ex: Variable) e o novo estado da fita.
/// É análogo a uma State Monad.
pub type Traced(a) {
  Traced(value: a, tape: Tape)
}

/// Cria uma nova fita vazia
pub fn new_tape() -> Tape {
  Tape(next_id: 0, operations: dict.new())
}

/// Registra uma nova variável (nó folha) no grafo
pub fn new_variable(tape: Tape, data: Tensor) -> Traced(Variable) {
  let id = tape.next_id
  let var = Variable(id: id, data: data)
  let new_tape = Tape(..tape, next_id: id + 1)
  Traced(value: var, tape: new_tape)
}

// =============================================================================
// OPERAÇÕES RASTREADAS (Traced Operations)
// =============================================================================

/// Sequenciamento de operações (Pipe Monádico)
/// Permite encadear camadas: x |> sequence(layer1) |> sequence(layer2)
pub fn sequence(
  input: Result(Traced(Variable), e),
  layer_fn: fn(Tape, Variable) -> Result(Traced(Variable), e),
) -> Result(Traced(Variable), e) {
  use Traced(var, tape) <- result.try(input)
  layer_fn(tape, var)
}

/// Adição rastreada: c = a + b
pub fn add(tape: Tape, a: Variable, b: Variable) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.add(a.data, b.data))
  
  let res_id = tape.next_id
  
  // Backward: y = a + b => dy/da = 1, dy/db = 1
  let backward = fn(grad: Tensor) {
    [#(a.id, grad), #(b.id, grad)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Subtração rastreada: c = a - b
pub fn sub(tape: Tape, a: Variable, b: Variable) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.sub(a.data, b.data))
  
  let res_id = tape.next_id
  
  // Backward: y = a - b 
  // dy/da = 1 * grad
  // dy/db = -1 * grad
  let backward = fn(grad: Tensor) {
    let neg_grad = tensor.negate(grad)
    [#(a.id, grad), #(b.id, neg_grad)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Multiplicação Element-wise rastreada: c = a * b
pub fn mul(tape: Tape, a: Variable, b: Variable) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.mul(a.data, b.data))
  
  let res_id = tape.next_id
  
  // Backward: y = a * b
  // dy/da = b * grad
  // dy/db = a * grad
  let backward = fn(grad: Tensor) {
    // TODO: Tratar erros
    let assert Ok(grad_a) = tensor.mul(grad, b.data)
    let assert Ok(grad_b) = tensor.mul(grad, a.data)
    [#(a.id, grad_a), #(b.id, grad_b)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Média (Reduce Mean) rastreada: y = mean(x)
/// Retorna um Tensor escalar (rank 0 ou 1 dependendo da implementação base, aqui forçamos 1)
pub fn mean(tape: Tape, a: Variable) -> Traced(Variable) {
  let val = tensor.mean(a.data)
  let res_data = tensor.from_list([val]) // Wrap scalar in tensor
  
  let res_id = tape.next_id
  
  // Backward: y = sum(x) / n
  // dy/dx = (1/n) * grad
  // O gradiente de entrada (grad) é um escalar (ou tensor de 1 elemento)
  // Precisamos expandi-lo para o shape de x e dividir por n
  let backward = fn(grad: Tensor) {
    let n = tensor.size(a.data) |> int.to_float
    let grad_val = tensor.to_list(grad) |> list.first |> result.unwrap(1.0)
    let scaled_grad_val = grad_val /. n
    
    // Cria um tensor cheio com o valor do gradiente escalado
    let grad_input = tensor.fill(a.data.shape, scaled_grad_val)
    [#(a.id, grad_input)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

/// Multiplicação matricial rastreada: c = a @ b
pub fn matmul(tape: Tape, a: Variable, b: Variable) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.matmul(a.data, b.data))
  
  let res_id = tape.next_id
  
  // Backward: y = a @ b
  // dy/da = grad @ b.T
  // dy/db = a.T @ grad
  let backward = fn(grad: Tensor) {
    // TODO: Tratamento de erro robusto dentro do backward
    let assert Ok(bt) = tensor.transpose(b.data)
    let assert Ok(at) = tensor.transpose(a.data)
    
    let assert Ok(grad_a) = tensor.matmul(grad, bt)
    let assert Ok(grad_b) = tensor.matmul(at, grad)
    
    [#(a.id, grad_a), #(b.id, grad_b)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// Transposição rastreada: c = a.T
pub fn transpose(tape: Tape, a: Variable) -> Result(Traced(Variable), tensor.TensorError) {
  use res_data <- result.try(tensor.transpose(a.data))
  
  let res_id = tape.next_id
  
  // Backward: y = a.T => dy/da = grad.T
  let backward = fn(grad: Tensor) {
    let assert Ok(grad_t) = tensor.transpose(grad)
    [#(a.id, grad_t)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Ok(Traced(value: Variable(id: res_id, data: res_data), tape: new_tape))
}

/// ReLU rastreada
pub fn relu(tape: Tape, a: Variable) -> Traced(Variable) {
  let res_data = tensor.map(a.data, fn(x) { 
    case x >. 0.0 {
      True -> x
      False -> 0.0
    }
  })
  
  let res_id = tape.next_id
  
  // Backward: y = relu(a) => dy/da = 1 se a > 0 senão 0
  let backward = fn(grad: Tensor) {
    let mask = tensor.map(a.data, fn(x) {
      case x >. 0.0 {
        True -> 1.0
        False -> 0.0
      }
    })
    let assert Ok(grad_a) = tensor.mul(grad, mask)
    [#(a.id, grad_a)]
  }
  
  let new_ops = dict.insert(tape.operations, res_id, backward)
  let new_tape = Tape(next_id: res_id + 1, operations: new_ops)
  
  Traced(value: Variable(id: res_id, data: res_data), tape: new_tape)
}

// =============================================================================
// MOTOR DE BACKPROPAGATION
// =============================================================================

/// Executa o backpropagation a partir de uma variável escalar (loss).
/// Retorna um Mapa de NodeId -> Gradiente (Tensor)
pub fn backward(tape: Tape, loss: Variable) -> Result(Dict(NodeId, Tensor), String) {
  // Gradiente inicial dLoss/dLoss = 1.0
  let initial_grad = tensor.ones(loss.data.shape)
  let initial_grads = dict.from_list([#(loss.id, initial_grad)])
  
  // Processar nós na ordem reversa de criação (Topological Sort implícito)
  // IDs são sequenciais, então (next_id - 1) até 0 garante a ordem correta.
  let all_ids = list.range(tape.next_id - 1, 0)
  
  let final_grads = list.fold(all_ids, initial_grads, fn(grads, current_id) {
    case dict.get(grads, current_id) {
      Error(_) -> grads // Nó não contribui para a loss ou não foi computado
      Ok(current_grad) -> {
        // Se este nó tem uma operação registrada na fita, expande o gradiente
        case dict.get(tape.operations, current_id) {
          Error(_) -> grads // Nó folha (input), sem pais para propagar
          Ok(back_fn) -> {
            let parent_grads = back_fn(current_grad)
            
            // Acumula gradientes nos pais (soma se já houver gradiente)
            list.fold(parent_grads, grads, fn(acc_grads, pair) {
              let #(pid, pgrad) = pair
              case dict.get(acc_grads, pid) {
                Error(_) -> dict.insert(acc_grads, pid, pgrad)
                Ok(existing) -> {
                  let assert Ok(sum) = tensor.add(existing, pgrad)
                  dict.insert(acc_grads, pid, sum)
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