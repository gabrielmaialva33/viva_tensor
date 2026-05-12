import gleam/dict
import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor/core/tensor
import viva_tensor/nn/autograd.{Traced}

pub fn main() {
  gleeunit.main()
}

fn assert_close(actual: Float, expected: Float, tolerance: Float) -> Bool {
  float.absolute_value(actual -. expected) <. tolerance
}

fn assert_list_close(
  actual: List(Float),
  expected: List(Float),
  tolerance: Float,
) -> Bool {
  list.length(actual) == list.length(expected)
  && list.zip(actual, expected)
  |> list.all(fn(pair) {
    let #(actual, expected) = pair
    assert_close(actual, expected, tolerance)
  })
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

pub fn broadcast_add_gradient_sums_over_expanded_axes_test() {
  let tape = autograd.new_tape()
  let assert Ok(x_data) =
    tensor.new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3])
  let bias_data = tensor.from_list([1.0, 2.0, 3.0])
  let assert Ok(weight_data) =
    tensor.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

  let Traced(x, tape1) = autograd.new_variable(tape, x_data)
  let Traced(bias, tape2) = autograd.new_variable(tape1, bias_data)
  let Traced(weight, tape3) = autograd.new_variable(tape2, weight_data)

  let assert Ok(Traced(shifted, tape4)) = autograd.add(tape3, x, bias)
  let assert Ok(Traced(weighted, tape5)) = autograd.mul(tape4, shifted, weight)
  let Traced(loss, tape6) = autograd.mean(tape5, weighted)

  let assert Ok(grads) = autograd.backward(tape6, loss)
  let assert Ok(dbias) = dict.get(grads, bias.id)

  tensor.shape(dbias) |> should.equal([3])
  assert_list_close(
    tensor.to_list(dbias),
    [5.0 /. 6.0, 7.0 /. 6.0, 9.0 /. 6.0],
    0.000001,
  )
  |> should.be_true()
}

pub fn broadcast_mul_gradient_sums_over_expanded_axes_test() {
  let tape = autograd.new_tape()
  let assert Ok(x_data) =
    tensor.new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3])
  let scale_data = tensor.from_list([2.0, 3.0, 4.0])

  let Traced(x, tape1) = autograd.new_variable(tape, x_data)
  let Traced(scale, tape2) = autograd.new_variable(tape1, scale_data)

  let assert Ok(Traced(scaled, tape3)) = autograd.mul(tape2, x, scale)
  let Traced(loss, tape4) = autograd.mean(tape3, scaled)

  let assert Ok(grads) = autograd.backward(tape4, loss)
  let assert Ok(dx) = dict.get(grads, x.id)
  let assert Ok(dscale) = dict.get(grads, scale.id)

  tensor.shape(dx) |> should.equal([2, 3])
  assert_list_close(
    tensor.to_list(dx),
    [2.0 /. 6.0, 3.0 /. 6.0, 4.0 /. 6.0, 2.0 /. 6.0, 3.0 /. 6.0, 4.0 /. 6.0],
    0.000001,
  )
  |> should.be_true()

  tensor.shape(dscale) |> should.equal([3])
  assert_list_close(
    tensor.to_list(dscale),
    [50.0 /. 6.0, 70.0 /. 6.0, 90.0 /. 6.0],
    0.000001,
  )
  |> should.be_true()
}
