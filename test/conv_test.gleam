//// Tests for `viva_tensor/nn/conv` — Conv1d, Conv3d, ConvTranspose2d
//// forward passes. Reference values cross-checked against PyTorch.

import gleam/option.{Some}
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/nn/conv
import viva_tensor/tensor

pub fn main() {
  gleeunit.main()
}

fn tensor_with_shape(data: List(Float), shape: List(Int)) -> tensor.Tensor {
  let assert Ok(t) = tensor.reshape(tensor.from_list(data), shape)
  t
}

// ---------------------------------------------------------------------------
// Conv1d
// ---------------------------------------------------------------------------

pub fn conv1d_forward_test() {
  // Sliding sum: weight = [1, 1, 1], stride 1, padding 0
  let cfg =
    conv.Conv1dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: 3,
      stride: 1,
      padding: 0,
      weight: tensor_with_shape([1.0, 1.0, 1.0], [1, 1, 3]),
      bias: Some(tensor.zeros([1])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])
  let assert Ok(out) = conv.conv1d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 3])
  numerics.lists_close(tensor.to_list(out), [6.0, 9.0, 12.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn conv1d_with_stride_test() {
  let cfg =
    conv.Conv1dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: 3,
      stride: 2,
      padding: 0,
      weight: tensor_with_shape([1.0, 1.0, 1.0], [1, 1, 3]),
      bias: Some(tensor.zeros([1])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])
  let assert Ok(out) = conv.conv1d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 2])
  numerics.lists_close(tensor.to_list(out), [6.0, 12.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn conv1d_with_padding_test() {
  let cfg =
    conv.Conv1dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: 3,
      stride: 1,
      padding: 1,
      weight: tensor_with_shape([1.0, 1.0, 1.0], [1, 1, 3]),
      bias: Some(tensor.zeros([1])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])
  let assert Ok(out) = conv.conv1d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 5])
  numerics.lists_close(
    tensor.to_list(out),
    [3.0, 6.0, 9.0, 12.0, 9.0],
    1.0e-9,
    1.0e-9,
  )
  |> should.be_true
}

pub fn conv1d_with_bias_test() {
  let cfg =
    conv.Conv1dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: 3,
      stride: 1,
      padding: 0,
      weight: tensor_with_shape([1.0, 1.0, 1.0], [1, 1, 3]),
      bias: Some(tensor_with_shape([10.0], [1])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0], [1, 1, 5])
  let assert Ok(out) = conv.conv1d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 3])
  numerics.lists_close(tensor.to_list(out), [16.0, 19.0, 22.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn conv1d_multi_channel_test() {
  // 2 input channels, 3 output channels, kernel 2
  // Input: [1, 2, 4] -> two channels of length 4
  //   ch0 = [1, 2, 3, 4], ch1 = [5, 6, 7, 8]
  // Weight: [3, 2, 2] - per output channel, sums all input channels.
  //   out_ch 0: w = [[1, 1], [1, 1]]   sum-all
  //   out_ch 1: w = [[1, 0], [0, 1]]   diagonal
  //   out_ch 2: w = [[0, 1], [1, 0]]   anti-diag
  let weight_data = [
    // oc=0
    1.0, 1.0, 1.0, 1.0,
    // oc=1
    1.0, 0.0, 0.0, 1.0,
    // oc=2
    0.0, 1.0, 1.0, 0.0,
  ]
  let cfg =
    conv.Conv1dConfig(
      in_channels: 2,
      out_channels: 3,
      kernel_size: 2,
      stride: 1,
      padding: 0,
      weight: tensor_with_shape(weight_data, [3, 2, 2]),
      bias: Some(tensor.zeros([3])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4])
  let assert Ok(out) = conv.conv1d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 3, 3])

  // out_length = 4 - 2 + 1 = 3
  // oc=0: sum both channels + sliding sum kernel 2
  //   pos 0: 1+2 + 5+6 = 14
  //   pos 1: 2+3 + 6+7 = 18
  //   pos 2: 3+4 + 7+8 = 22
  // oc=1: ch0 first elem + ch1 second elem (diagonal kernel)
  //   pos 0: 1 + 6 = 7
  //   pos 1: 2 + 7 = 9
  //   pos 2: 3 + 8 = 11
  // oc=2: ch0 second elem + ch1 first elem (anti-diag kernel)
  //   pos 0: 2 + 5 = 7
  //   pos 1: 3 + 6 = 9
  //   pos 2: 4 + 7 = 11
  let expected = [14.0, 18.0, 22.0, 7.0, 9.0, 11.0, 7.0, 9.0, 11.0]
  numerics.lists_close(tensor.to_list(out), expected, 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn conv1d_shape_error_test() {
  let cfg = conv.conv1d_init(1, 1, 3, 1, 0)
  // Rank-2 input — should fail
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0], [1, 5])
  case conv.conv1d_forward(cfg, x) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

pub fn conv3d_forward_basic_test() {
  // 2x2x2 input, kernel of ones = sum all 8 elements
  let cfg =
    conv.Conv3dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: #(2, 2, 2),
      stride: #(1, 1, 1),
      padding: #(0, 0, 0),
      weight: tensor_with_shape([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [
        1,
        1,
        2,
        2,
        2,
      ]),
      bias: Some(tensor.zeros([1])),
    )
  let x =
    tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 2, 2, 2])
  let assert Ok(out) = conv.conv3d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 1, 1, 1])
  numerics.lists_close(tensor.to_list(out), [36.0], 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn conv3d_shape_error_test() {
  let cfg = conv.conv3d_init(1, 1, #(2, 2, 2), #(1, 1, 1), #(0, 0, 0))
  // Rank-4 input — should fail (expects rank-5)
  let x =
    tensor_with_shape([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 1, 2, 4])
  case conv.conv3d_forward(cfg, x) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

// ---------------------------------------------------------------------------
// ConvTranspose2d
// ---------------------------------------------------------------------------

pub fn conv_transpose_2d_basic_test() {
  // 2x2 input, 2x2 kernel of ones, stride 1, padding 0 -> 3x3 output
  //  Input:  1 2     Kernel: 1 1
  //          3 4             1 1
  // Output positions:
  //  (0,0) = 1
  //  (0,1) = 1+2 = 3
  //  (0,2) = 2
  //  (1,0) = 1+3 = 4
  //  (1,1) = 1+2+3+4 = 10
  //  (1,2) = 2+4 = 6
  //  (2,0) = 3
  //  (2,1) = 3+4 = 7
  //  (2,2) = 4
  let cfg =
    conv.ConvTranspose2dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: #(2, 2),
      stride: #(1, 1),
      padding: #(0, 0),
      output_padding: #(0, 0),
      weight: tensor_with_shape([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]),
      bias: Some(tensor.zeros([1])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
  let assert Ok(out) = conv.conv_transpose_2d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 3, 3])
  let expected = [1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]
  numerics.lists_close(tensor.to_list(out), expected, 1.0e-9, 1.0e-9)
  |> should.be_true
}

pub fn conv_transpose_2d_stride_2_test() {
  // 2x2 input, 2x2 kernel of ones, stride 2 -> 4x4 output (each input cell
  // becomes its own 2x2 block since stride spreads them apart).
  let cfg =
    conv.ConvTranspose2dConfig(
      in_channels: 1,
      out_channels: 1,
      kernel_size: #(2, 2),
      stride: #(2, 2),
      padding: #(0, 0),
      output_padding: #(0, 0),
      weight: tensor_with_shape([1.0, 1.0, 1.0, 1.0], [1, 1, 2, 2]),
      bias: Some(tensor.zeros([1])),
    )
  let x = tensor_with_shape([1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2])
  let assert Ok(out) = conv.conv_transpose_2d_forward(cfg, x)

  tensor.shape(out) |> should.equal([1, 1, 4, 4])
  //   1 1 2 2
  //   1 1 2 2
  //   3 3 4 4
  //   3 3 4 4
  let expected = [
    1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0,
    4.0,
  ]
  numerics.lists_close(tensor.to_list(out), expected, 1.0e-9, 1.0e-9)
  |> should.be_true
}
