import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/axis
import viva_tensor/core/ops
import viva_tensor/core/tensor as core_tensor
import viva_tensor/named
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// TENSOR CONSTRUCTORS
// =============================================================================

pub fn zeros_test() {
  let z = t.zeros([2, 3])
  t.shape(z) |> should.equal([2, 3])
  t.size(z) |> should.equal(6)
  t.to_list(z) |> should.equal([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
}

pub fn ones_test() {
  let o = t.ones([3])
  t.shape(o) |> should.equal([3])
  t.to_list(o) |> should.equal([1.0, 1.0, 1.0])
}

pub fn fill_test() {
  let f = t.fill([2, 2], 5.0)
  t.to_list(f) |> should.equal([5.0, 5.0, 5.0, 5.0])
}

pub fn from_list_test() {
  let v = t.from_list([1.0, 2.0, 3.0])
  t.shape(v) |> should.equal([3])
  t.to_list(v) |> should.equal([1.0, 2.0, 3.0])
}

pub fn vector_test() {
  let v = t.vector([1.0, 2.0])
  t.rank(v) |> should.equal(1)
}

pub fn matrix_test() {
  let m = t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  m |> should.be_ok()
  case m {
    Ok(mat) -> t.shape(mat) |> should.equal([2, 2])
    Error(_) -> should.fail()
  }
}

pub fn from_list2d_test() {
  let m = t.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  m |> should.be_ok()
  case m {
    Ok(mat) -> {
      t.shape(mat) |> should.equal([2, 2])
      t.to_list(mat) |> should.equal([1.0, 2.0, 3.0, 4.0])
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// ELEMENT-WISE OPERATIONS
// =============================================================================

pub fn add_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([4.0, 5.0, 6.0])
  case t.add(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([5.0, 7.0, 9.0])
    Error(_) -> should.fail()
  }
}

pub fn sub_test() {
  let a = t.from_list([5.0, 5.0])
  let b = t.from_list([2.0, 3.0])
  case t.sub(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([3.0, 2.0])
    Error(_) -> should.fail()
  }
}

pub fn mul_test() {
  let a = t.from_list([2.0, 3.0])
  let b = t.from_list([4.0, 5.0])
  case t.mul(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([8.0, 15.0])
    Error(_) -> should.fail()
  }
}

pub fn div_test() {
  let a = t.from_list([10.0, 20.0])
  let b = t.from_list([2.0, 4.0])
  case t.div(a, b) {
    Ok(c) -> t.to_list(c) |> should.equal([5.0, 5.0])
    Error(_) -> should.fail()
  }
}

pub fn scale_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let s = t.scale(a, 2.0)
  t.to_list(s) |> should.equal([2.0, 4.0, 6.0])
}

pub fn map_test() {
  let a = t.from_list([1.0, 4.0, 9.0])
  let b = t.map(a, fn(x) { x *. x })
  t.to_list(b) |> should.equal([1.0, 16.0, 81.0])
}

// =============================================================================
// REDUCTIONS
// =============================================================================

pub fn sum_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0])
  t.sum(a) |> should.equal(10.0)
}

pub fn mean_test() {
  let a = t.from_list([2.0, 4.0, 6.0, 8.0])
  t.mean(a) |> should.equal(5.0)
}

pub fn max_test() {
  let a = t.from_list([1.0, 5.0, 3.0, 2.0])
  t.max(a) |> should.equal(5.0)
}

pub fn min_test() {
  let a = t.from_list([1.0, 5.0, 3.0, 2.0])
  t.min(a) |> should.equal(1.0)
}

pub fn argmax_test() {
  let a = t.from_list([1.0, 5.0, 3.0])
  t.argmax(a) |> should.equal(1)
}

pub fn argmin_test() {
  let a = t.from_list([3.0, 1.0, 5.0])
  t.argmin(a) |> should.equal(1)
}

// =============================================================================
// DOT PRODUCT & MATMUL
// =============================================================================

pub fn dot_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([4.0, 5.0, 6.0])
  case t.dot(a, b) {
    Ok(d) -> d |> should.equal(32.0)
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    Error(_) -> should.fail()
  }
}

pub fn matmul_test() {
  // [1, 2]   [5, 6]   [1*5+2*7, 1*6+2*8]   [19, 22]
  // [3, 4] x [7, 8] = [3*5+4*7, 3*6+4*8] = [43, 50]
  case t.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case t.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case t.matmul(a, b) {
            Ok(c) -> {
              t.shape(c) |> should.equal([2, 2])
              t.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// OPTIMIZED OPERATIONS (Erlang Array Backend)
// =============================================================================

pub fn dot_fast_test() {
  let a = core_tensor.from_list([1.0, 2.0, 3.0])
  let b = core_tensor.from_list([4.0, 5.0, 6.0])
  case ops.dot_fast(a, b) {
    Ok(d) -> d |> should.equal(32.0)
    // Same result as dot: 1*4 + 2*5 + 3*6 = 32
    Error(_) -> should.fail()
  }
}

pub fn matmul_fast_test() {
  // Same test as matmul_test but using optimized version
  case core_tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case core_tensor.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case ops.matmul_fast(a, b) {
            Ok(c) -> {
              core_tensor.shape(c) |> should.equal([2, 2])
              core_tensor.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn dot_auto_test() {
  // Auto-selecting dot (uses NIF if available)
  let a = core_tensor.from_list([1.0, 2.0, 3.0])
  let b = core_tensor.from_list([4.0, 5.0, 6.0])
  case ops.dot_auto(a, b) {
    Ok(d) -> d |> should.equal(32.0)
    Error(_) -> should.fail()
  }
}

pub fn matmul_auto_test() {
  // Auto-selecting matmul (uses NIF if available)
  case core_tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0]) {
    Ok(a) -> {
      case core_tensor.matrix(2, 2, [5.0, 6.0, 7.0, 8.0]) {
        Ok(b) -> {
          case ops.matmul_auto(a, b) {
            Ok(c) -> {
              core_tensor.shape(c) |> should.equal([2, 2])
              core_tensor.to_list(c) |> should.equal([19.0, 22.0, 43.0, 50.0])
            }
            Error(_) -> should.fail()
          }
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

pub fn transpose_test() {
  // [1, 2, 3]      [1, 4]
  // [4, 5, 6]  ->  [2, 5]
  //                [3, 6]
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      case t.transpose(m) {
        Ok(mt) -> {
          t.shape(mt) |> should.equal([3, 2])
          t.to_list(mt) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// SHAPE OPERATIONS
// =============================================================================

pub fn reshape_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  case t.reshape(a, [2, 3]) {
    Ok(r) -> t.shape(r) |> should.equal([2, 3])
    Error(_) -> should.fail()
  }
}

pub fn flatten_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      let f = t.flatten(m)
      t.shape(f) |> should.equal([6])
    }
    Error(_) -> should.fail()
  }
}

pub fn squeeze_test() {
  let a = t.zeros([1, 3, 1])
  let s = t.squeeze(a)
  t.shape(s) |> should.equal([3])
}

pub fn unsqueeze_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let u = t.unsqueeze(a, 0)
  t.shape(u) |> should.equal([1, 3])
}

// =============================================================================
// BROADCASTING
// =============================================================================

pub fn can_broadcast_test() {
  // Same shape
  t.can_broadcast([2, 3], [2, 3]) |> should.be_true()

  // Scalar broadcast
  t.can_broadcast([2, 3], [1]) |> should.be_true()

  // Different ranks
  t.can_broadcast([2, 3], [3]) |> should.be_true()

  // Incompatible
  t.can_broadcast([2, 3], [4]) |> should.be_false()
}

pub fn add_broadcast_test() {
  let a = t.zeros([2, 3])
  let b = t.from_list([1.0, 2.0, 3.0])
  case t.add_broadcast(a, b) {
    Ok(c) -> {
      t.shape(c) |> should.equal([2, 3])
      t.to_list(c) |> should.equal([1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// STRIDED TENSORS
// =============================================================================

pub fn to_strided_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0])
  let s = t.to_strided(a)
  t.is_contiguous(s) |> should.be_true()
  t.to_list(s) |> should.equal([1.0, 2.0, 3.0, 4.0])
}

pub fn strided_transpose_test() {
  case t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) {
    Ok(m) -> {
      let s = t.to_strided(m)
      case t.transpose_strided(s) {
        Ok(st) -> {
          t.shape(st) |> should.equal([3, 2])
          // Zero-copy: not contiguous after transpose
          t.is_contiguous(st) |> should.be_false()
          // But converting back gives correct data
          let c = t.to_contiguous(st)
          t.to_list(c) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// RANDOM
// =============================================================================

pub fn random_uniform_test() {
  let r = t.random_uniform([100])
  t.shape(r) |> should.equal([100])
  // All values should be in [0, 1)
  let vals = t.to_list(r)
  let in_range =
    vals
    |> list.all(fn(v) { v >=. 0.0 && v <. 1.0 })
  in_range |> should.be_true()
}

pub fn xavier_init_test() {
  let w = t.xavier_init(128, 64)
  t.shape(w) |> should.equal([64, 128])
  // Xavier: std ≈ sqrt(2 / (fan_in + fan_out))
  let std_val = t.std(w)
  // Should be around 0.1 for these dimensions
  { std_val >. 0.05 && std_val <. 0.2 } |> should.be_true()
}

// =============================================================================
// AXIS MODULE
// =============================================================================

pub fn axis_constructors_test() {
  let b = axis.batch(32)
  b.size |> should.equal(32)

  let f = axis.feature(128)
  f.size |> should.equal(128)

  let s = axis.seq(10)
  s.size |> should.equal(10)
}

pub fn axis_equals_test() {
  axis.equals(axis.Batch, axis.Batch) |> should.be_true()
  axis.equals(axis.Batch, axis.Seq) |> should.be_false()
  axis.equals(axis.Named("foo"), axis.Named("foo")) |> should.be_true()
  axis.equals(axis.Named("foo"), axis.Named("bar")) |> should.be_false()
}

// =============================================================================
// NAMED TENSORS
// =============================================================================

pub fn named_zeros_test() {
  let nt = named.zeros([axis.batch(2), axis.feature(3)])
  named.shape(nt) |> should.equal([2, 3])
  named.rank(nt) |> should.equal(2)
}

pub fn named_find_axis_test() {
  let nt = named.zeros([axis.batch(4), axis.seq(10), axis.feature(64)])
  case named.find_axis(nt, axis.Seq) {
    Ok(idx) -> idx |> should.equal(1)
    Error(_) -> should.fail()
  }
  case named.find_axis(nt, axis.Channel) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn named_sum_along_test() {
  let nt = named.ones([axis.batch(2), axis.feature(3)])
  // Sum along batch: [2, 3] -> [3] (each feature summed over batch)
  case named.sum_along(nt, axis.Batch) {
    Ok(result) -> {
      named.shape(result) |> should.equal([3])
      // Each feature had 2 ones, so sum = 2
      tensor.to_list(named.to_tensor(result))
      |> should.equal([2.0, 2.0, 2.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn named_add_test() {
  let a = named.ones([axis.batch(2), axis.feature(3)])
  let b = named.ones([axis.batch(2), axis.feature(3)])
  case named.add(a, b) {
    Ok(c) -> {
      tensor.to_list(named.to_tensor(c))
      |> list.all(fn(v) { v == 2.0 })
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn named_scale_test() {
  let nt = named.ones([axis.batch(2)])
  let scaled = named.scale(nt, 5.0)
  tensor.to_list(named.to_tensor(scaled)) |> should.equal([5.0, 5.0])
}

pub fn named_describe_test() {
  let nt = named.zeros([axis.batch(32), axis.feature(128)])
  let desc = named.describe(nt)
  // Should contain axis names and sizes
  { string.contains(desc, "batch") && string.contains(desc, "32") }
  |> should.be_true()
}

// =============================================================================
// CONVOLUTION TESTS
// =============================================================================

pub fn pad2d_test() {
  // 2x2 input
  let input = tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [2, 2])

  // Pad by 1 on each side -> 4x4
  let result = tensor.pad2d(input, 1, 1)
  result |> should.be_ok()

  let padded = case result {
    Ok(p) -> p
    Error(_) -> tensor.zeros([1])
  }

  tensor.shape(padded) |> should.equal([4, 4])

  // Check corners are zeros
  let data = tensor.get_data(padded)
  list.first(data) |> should.equal(Ok(0.0))
}

pub fn conv2d_simple_test() {
  // 3x3 input
  let input =
    tensor.Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape: [
      3,
      3,
    ])

  // 2x2 kernel (all ones = sum)
  let kernel = tensor.Tensor(data: [1.0, 1.0, 1.0, 1.0], shape: [2, 2])

  let config =
    tensor.Conv2dConfig(
      kernel_h: 2,
      kernel_w: 2,
      stride_h: 1,
      stride_w: 1,
      padding_h: 0,
      padding_w: 0,
    )

  let result = tensor.conv2d(input, kernel, config)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // Output should be 2x2
  tensor.shape(output) |> should.equal([2, 2])

  // Top-left: 1+2+4+5 = 12
  let data = tensor.get_data(output)
  list.first(data) |> should.equal(Ok(12.0))
}

pub fn conv2d_same_padding_test() {
  // 4x4 input
  let input = tensor.ones([4, 4])

  // 3x3 kernel with "same" padding
  let kernel = tensor.fill([3, 3], 1.0)
  let config = tensor.conv2d_same(3, 3)

  let result = tensor.conv2d(input, kernel, config)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // With same padding, output = input size
  tensor.shape(output) |> should.equal([4, 4])
}

pub fn max_pool2d_test() {
  // 4x4 input
  let input =
    tensor.Tensor(
      data: [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
        14.0, 15.0, 16.0,
      ],
      shape: [4, 4],
    )

  // 2x2 pool, stride 2
  let result = tensor.max_pool2d(input, 2, 2, 2, 2)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // Output should be 2x2
  tensor.shape(output) |> should.equal([2, 2])

  // Max of [1,2,5,6] = 6, [3,4,7,8] = 8, etc.
  let data = tensor.get_data(output)
  data |> should.equal([6.0, 8.0, 14.0, 16.0])
}

pub fn avg_pool2d_test() {
  // 4x4 input (all 4s)
  let input = tensor.fill([4, 4], 4.0)

  // 2x2 pool, stride 2
  let result = tensor.avg_pool2d(input, 2, 2, 2, 2)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // Output should be 2x2, all 4.0
  tensor.shape(output) |> should.equal([2, 2])
  tensor.get_data(output) |> should.equal([4.0, 4.0, 4.0, 4.0])
}

pub fn conv2d_batch_test() {
  // Batch of 2, 1 channel, 3x3
  let input = tensor.ones([2, 1, 3, 3])

  // 1 output channel, 1 input channel, 2x2 kernel
  let kernel = tensor.fill([1, 1, 2, 2], 1.0)

  let config =
    tensor.Conv2dConfig(
      kernel_h: 2,
      kernel_w: 2,
      stride_h: 1,
      stride_w: 1,
      padding_h: 0,
      padding_w: 0,
    )

  let result = tensor.conv2d(input, kernel, config)
  result |> should.be_ok()

  let output = case result {
    Ok(o) -> o
    Error(_) -> tensor.zeros([1])
  }

  // [2, 1, 2, 2]
  tensor.shape(output) |> should.equal([2, 1, 2, 2])

  // Each position = sum of 4 ones = 4.0
  let data = tensor.get_data(output)
  list.first(data) |> should.equal(Ok(4.0))
}

// =============================================================================
// IMPORTS FOR TESTS
// =============================================================================

import gleam/list
import gleam/string
