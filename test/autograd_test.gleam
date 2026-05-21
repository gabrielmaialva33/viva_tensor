import gleam/dict
import gleam/float
import gleam/list
import gleam/result
import gleeunit
import gleeunit/should
import viva_tensor/core/ops
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

fn tensor_from(data: List(Float), shape: List(Int)) {
  let assert Ok(t) = tensor.new(data, shape)
  t
}

fn replace_at(values: List(Float), index: Int, value: Float) -> List(Float) {
  values
  |> list.index_map(fn(item, i) {
    case i == index {
      True -> value
      False -> item
    }
  })
}

fn finite_difference(
  values: List(Float),
  shape: List(Int),
  index: Int,
  epsilon: Float,
  loss: fn(tensor.Tensor) -> Float,
) -> Float {
  let current =
    values
    |> list.drop(index)
    |> list.first
    |> result.unwrap(0.0)
  let plus = replace_at(values, index, current +. epsilon)
  let minus = replace_at(values, index, current -. epsilon)

  { loss(tensor_from(plus, shape)) -. loss(tensor_from(minus, shape)) }
  /. { 2.0 *. epsilon }
}

fn numerical_gradient(
  values: List(Float),
  shape: List(Int),
  epsilon: Float,
  loss: fn(tensor.Tensor) -> Float,
) -> List(Float) {
  case values == [] {
    True -> []
    False ->
      range_int(0, list.length(values) - 1)
      |> list.map(fn(index) {
        finite_difference(values, shape, index, epsilon, loss)
      })
  }
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

pub fn broadcast_sub_gradient_sums_over_expanded_axes_test() {
  let tape = autograd.new_tape()
  let assert Ok(x_data) =
    tensor.new([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [2, 3])
  let bias_data = tensor.from_list([1.0, 2.0, 3.0])
  let assert Ok(weight_data) =
    tensor.new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

  let Traced(x, tape1) = autograd.new_variable(tape, x_data)
  let Traced(bias, tape2) = autograd.new_variable(tape1, bias_data)
  let Traced(weight, tape3) = autograd.new_variable(tape2, weight_data)

  let assert Ok(Traced(shifted, tape4)) = autograd.sub(tape3, x, bias)
  let assert Ok(Traced(weighted, tape5)) = autograd.mul(tape4, shifted, weight)
  let Traced(loss, tape6) = autograd.mean(tape5, weighted)

  let assert Ok(grads) = autograd.backward(tape6, loss)
  let assert Ok(dx) = dict.get(grads, x.id)
  let assert Ok(dbias) = dict.get(grads, bias.id)

  tensor.shape(dx) |> should.equal([2, 3])
  assert_list_close(
    tensor.to_list(dx),
    [1.0 /. 6.0, 2.0 /. 6.0, 3.0 /. 6.0, 4.0 /. 6.0, 5.0 /. 6.0, 6.0 /. 6.0],
    0.000001,
  )
  |> should.be_true()

  tensor.shape(dbias) |> should.equal([3])
  assert_list_close(
    tensor.to_list(dbias),
    [-5.0 /. 6.0, -7.0 /. 6.0, -9.0 /. 6.0],
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

pub fn gradient_check_broadcast_chain_test() {
  let epsilon = 0.00001
  let x_data = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
  let scale_data = [2.0, 3.0, 4.0]
  let bias_data = [1.0, 2.0, 3.0]

  let tape = autograd.new_tape()
  let Traced(x, tape1) =
    autograd.new_variable(tape, tensor_from(x_data, [2, 3]))
  let Traced(scale, tape2) =
    autograd.new_variable(tape1, tensor.from_list(scale_data))
  let Traced(bias, tape3) =
    autograd.new_variable(tape2, tensor.from_list(bias_data))

  let assert Ok(Traced(scaled, tape4)) = autograd.mul(tape3, x, scale)
  let assert Ok(Traced(shifted, tape5)) = autograd.add(tape4, scaled, bias)
  let Traced(loss, tape6) = autograd.mean(tape5, shifted)
  let assert Ok(grads) = autograd.backward(tape6, loss)
  let assert Ok(dscale) = dict.get(grads, scale.id)
  let assert Ok(dbias) = dict.get(grads, bias.id)

  let x_tensor = tensor_from(x_data, [2, 3])
  let bias_tensor = tensor.from_list(bias_data)
  let scale_loss = fn(scale_tensor) {
    broadcast_chain_loss(x_tensor, scale_tensor, bias_tensor)
  }
  let scale_numeric = numerical_gradient(scale_data, [3], epsilon, scale_loss)

  let scale_tensor = tensor.from_list(scale_data)
  let bias_loss = fn(bias_tensor) {
    broadcast_chain_loss(x_tensor, scale_tensor, bias_tensor)
  }
  let bias_numeric = numerical_gradient(bias_data, [3], epsilon, bias_loss)

  assert_list_close(tensor.to_list(dscale), scale_numeric, 0.0001)
  |> should.be_true()
  assert_list_close(tensor.to_list(dbias), bias_numeric, 0.0001)
  |> should.be_true()
}

pub fn gradient_check_matmul_mean_test() {
  let epsilon = 0.00001
  let a_data = [1.0, -2.0, 0.5, 3.0]
  let b_data = [2.0, 1.5, -1.0, 0.25]

  let tape = autograd.new_tape()
  let Traced(a, tape1) =
    autograd.new_variable(tape, tensor_from(a_data, [2, 2]))
  let Traced(b, tape2) =
    autograd.new_variable(tape1, tensor_from(b_data, [2, 2]))

  let assert Ok(Traced(product, tape3)) = autograd.matmul(tape2, a, b)
  let Traced(loss, tape4) = autograd.mean(tape3, product)
  let assert Ok(grads) = autograd.backward(tape4, loss)
  let assert Ok(da) = dict.get(grads, a.id)
  let assert Ok(db) = dict.get(grads, b.id)

  let b_tensor = tensor_from(b_data, [2, 2])
  let a_loss = fn(a_tensor) { matmul_mean_loss(a_tensor, b_tensor) }
  let a_numeric = numerical_gradient(a_data, [2, 2], epsilon, a_loss)

  let a_tensor = tensor_from(a_data, [2, 2])
  let b_loss = fn(b_tensor) { matmul_mean_loss(a_tensor, b_tensor) }
  let b_numeric = numerical_gradient(b_data, [2, 2], epsilon, b_loss)

  assert_list_close(tensor.to_list(da), a_numeric, 0.0001)
  |> should.be_true()
  assert_list_close(tensor.to_list(db), b_numeric, 0.0001)
  |> should.be_true()
}

pub fn gradient_check_relu_mean_test() {
  let epsilon = 0.00001
  let x_data = [-2.0, -0.5, 0.7, 3.0]

  let tape = autograd.new_tape()
  let Traced(x, tape1) = autograd.new_variable(tape, tensor.from_list(x_data))
  let Traced(activated, tape2) = autograd.relu(tape1, x)
  let Traced(loss, tape3) = autograd.mean(tape2, activated)
  let assert Ok(grads) = autograd.backward(tape3, loss)
  let assert Ok(dx) = dict.get(grads, x.id)

  let relu_numeric =
    numerical_gradient(x_data, [4], epsilon, fn(x_tensor) {
      let activated = ops.relu(x_tensor)
      ops.mean(activated)
    })

  assert_list_close(tensor.to_list(dx), relu_numeric, 0.0001)
  |> should.be_true()
}

fn broadcast_chain_loss(
  x: tensor.Tensor,
  scale: tensor.Tensor,
  bias: tensor.Tensor,
) -> Float {
  let assert Ok(scaled) = ops.mul_broadcast(x, scale)
  let assert Ok(shifted) = ops.add_broadcast(scaled, bias)
  ops.mean(shifted)
}

fn matmul_mean_loss(a: tensor.Tensor, b: tensor.Tensor) -> Float {
  let assert Ok(product) = ops.matmul_auto(a, b)
  ops.mean(product)
}

fn range_int(from: Int, to: Int) -> List(Int) {
  range_loop(from, to, [])
}

fn range_loop(from: Int, to: Int, acc: List(Int)) -> List(Int) {
  case from > to {
    True -> list.reverse(acc)
    False -> range_loop(from + 1, to, [from, ..acc])
  }
}
