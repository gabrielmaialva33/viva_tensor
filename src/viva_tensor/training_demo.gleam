import viva_tensor/autograd.{type Tape, type Variable, Traced}
import viva_tensor/nn
import viva_tensor/tensor
import gleam/list
import gleam/result
import gleam/dict
import gleam/io
import gleam/int
import gleam/float

// Configurações do treinamento
const learning_rate = 0.01
const epochs = 500
const input_features = 1
const hidden_features = 4
const output_features = 1

pub type TrainingState {
  TrainingState(
    tape: Tape,
    layer1: nn.Linear,
    layer2: nn.Linear,
  )
}

pub fn main() {
  io.println("🚀 Iniciando Demonstração de Treinamento Mycelial...")
  
  // 1. Criar Dados de Treinamento
  let tape = autograd.new_tape()
  
  // x = [1, 2, 3, 4, 5]
  // y = [2, 4, 6, 8, 10] (aproximado)
  let x_data = tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let y_data = tensor.from_list([2.1, 3.9, 6.2, 8.1, 10.3])

  let Traced(x, tape1) = autograd.new_variable(tape, x_data)
  let Traced(y, tape2) = autograd.new_variable(tape1, y_data)

  // 2. Inicializar Camadas
  let Traced(layer1, tape3) = nn.linear(tape2, input_features, hidden_features)
  let Traced(layer2, tape4) = nn.linear(tape3, hidden_features, output_features)

  let state = TrainingState(tape: tape4, layer1: layer1, layer2: layer2)

  // 3. Rodar Loop de Treinamento
  let final_state = list.fold(list.range(0, epochs - 1), state, fn(acc_state, epoch) {
    train_step(acc_state, epoch, x, y)
  })

  io.println("✅ Treinamento finalizado!")
}

fn train_step(state: TrainingState, epoch: Int, x: Variable, target: Variable) -> TrainingState {
  // Forward Pass
  let assert Ok(Traced(l1_out, tape1)) = nn.linear_forward(state.tape, state.layer1, x)
  let Traced(hidden_act, tape2) = nn.relu(tape1, l1_out)
  let assert Ok(Traced(output, tape3)) = nn.linear_forward(tape2, state.layer2, hidden_act)

  // Calcular Loss
  let assert Ok(Traced(loss_var, tape4)) = nn.mse_loss(tape3, output, target)

  // Backward Pass
  let assert Ok(grads) = autograd.backward(tape4, loss_var)

  // Log progress
  case epoch % 50 == 0 {
    True -> {
      let loss_val = tensor.to_list(loss_var.data) |> list.first |> result.unwrap(0.0)
      io.println("Época " <> int.to_string(epoch) <> " | Loss: " <> float.to_string(loss_val))
    }
    False -> Nil
  }

  // Atualizar Parâmetros (Gradient Descent Manual)
  // Precisamos criar novas variáveis para os pesos atualizados
  
  // Update Layer 1
  let assert Ok(gw1) = dict.get(grads, state.layer1.w.id)
  let assert Ok(gb1) = dict.get(grads, state.layer1.b.id)
  let assert Ok(new_w1_data) = tensor.sub(state.layer1.w.data, tensor.scale(gw1, learning_rate))
  let assert Ok(new_b1_data) = tensor.sub(state.layer1.b.data, tensor.scale(gb1, learning_rate))
  
  // Update Layer 2
  let assert Ok(gw2) = dict.get(grads, state.layer2.w.id)
  let assert Ok(gb2) = dict.get(grads, state.layer2.b.id)
  let assert Ok(new_w2_data) = tensor.sub(state.layer2.w.data, tensor.scale(gw2, learning_rate))
  let assert Ok(new_b2_data) = tensor.sub(state.layer2.b.data, tensor.scale(gb2, learning_rate))

  // Criar nova fita limpa para a próxima iteração (senão a tape cresce infinitamente!)
  // IMPORTANTE: Em treinamento real, limpamos a tape a cada iteração.
  let next_tape = autograd.new_tape()
  
  // Registrar novos pesos na fita limpa
  let Traced(nw1, nt1) = autograd.new_variable(next_tape, new_w1_data)
  let Traced(nb1, nt2) = autograd.new_variable(nt1, new_b1_data)
  let Traced(nw2, nt3) = autograd.new_variable(nt2, new_w2_data)
  let Traced(nb2, nt4) = autograd.new_variable(nt3, new_b2_data)

  TrainingState(
    tape: nt4,
    layer1: nn.Linear(w: nw1, b: nb1),
    layer2: nn.Linear(w: nw2, b: nb2)
  )
}