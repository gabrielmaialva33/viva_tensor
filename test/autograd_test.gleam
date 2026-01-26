import gleam/dict
import gleam/list
import gleam/result
import gleeunit
import gleeunit/should
import viva_tensor/autograd.{type Tape, Traced}
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

// -----------------------------------------------------------------------------
// Basic Autograd Tests
// -----------------------------------------------------------------------------

pub fn add_test() {
  let tape = autograd.new_tape()

  // x = [2.0]
  // y = [3.0]
  let x_data = tensor.from_list([2.0])
  let y_data = tensor.from_list([3.0])

  let Traced(x, tape1) = autograd.new_variable(tape, x_data)
  let Traced(y, tape2) = autograd.new_variable(tape1, y_data)

  // z = x + y = [5.0]
  let assert Ok(Traced(z, tape3)) = autograd.add(tape2, x, y)

  tensor.to_list(z.data)
  |> should.equal([5.0])

  // Backward
  // dz/dx = 1
  // dz/dy = 1
  let assert Ok(grads) = autograd.backward(tape3, z)

  let assert Ok(dx) = dict.get(grads, x.id)
  let assert Ok(dy) = dict.get(grads, y.id)

  tensor.to_list(dx) |> should.equal([1.0])
  tensor.to_list(dy) |> should.equal([1.0])
}

pub fn mul_test() {
  let tape = autograd.new_tape()

  // x = [2.0]
  // y = [3.0]
  let Traced(x, tape1) = autograd.new_variable(tape, tensor.from_list([2.0]))
  let Traced(y, tape2) = autograd.new_variable(tape1, tensor.from_list([3.0]))

  // z = x * y = [6.0]
  let assert Ok(Traced(z, tape3)) = autograd.mul(tape2, x, y)

  tensor.to_list(z.data)
  |> should.equal([6.0])

  // Backward
  // dz/dx = y = 3
  // dz/dy = x = 2
  let assert Ok(grads) = autograd.backward(tape3, z)

  let assert Ok(dx) = dict.get(grads, x.id)
  let assert Ok(dy) = dict.get(grads, y.id)

  tensor.to_list(dx) |> should.equal([3.0])
  tensor.to_list(dy) |> should.equal([2.0])
}

pub fn mean_test() {
  let tape = autograd.new_tape()

  // x = [2.0, 4.0]
  let Traced(x, tape1) =
    autograd.new_variable(tape, tensor.from_list([2.0, 4.0]))

  // z = mean(x) = 3.0
  let Traced(z, tape2) = autograd.mean(tape1, x)

  tensor.to_list(z.data)
  |> should.equal([3.0])

  // Backward
  // dz/dx = [1/2, 1/2] = [0.5, 0.5]
  let assert Ok(grads) = autograd.backward(tape2, z)

  let assert Ok(dx) = dict.get(grads, x.id)

  tensor.to_list(dx) |> should.equal([0.5, 0.5])
}

pub fn composite_test() {
  // z = (x + y) * x
  // dz/dx = (1 * x) + (x+y) * 1 = x + x + y = 2x + y
  // dz/dy = x

  let tape = autograd.new_tape()
  let Traced(x, tape1) = autograd.new_variable(tape, tensor.from_list([2.0]))
  let Traced(y, tape2) = autograd.new_variable(tape1, tensor.from_list([3.0]))

  // sum = x + y = 5
  let assert Ok(Traced(sum, tape3)) = autograd.add(tape2, x, y)

  // z = sum * x = 5 * 2 = 10
  let assert Ok(Traced(z, tape4)) = autograd.mul(tape3, sum, x)

  tensor.to_list(z.data) |> should.equal([10.0])

  let assert Ok(grads) = autograd.backward(tape4, z)

  let assert Ok(dx) = dict.get(grads, x.id)
  let assert Ok(dy) = dict.get(grads, y.id)

  // dz/dy = x = 2
  tensor.to_list(dy) |> should.equal([2.0])

  // dz/dx = 2x + y = 2(2) + 3 = 7
  tensor.to_list(dx) |> should.equal([7.0])
}
