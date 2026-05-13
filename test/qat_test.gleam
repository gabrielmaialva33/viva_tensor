//// Tests for QAT (Quantization-Aware Training) primitives.
////
//// Covers symmetric / asymmetric / per-channel observe, fake-quant
//// round-trip at high and low bit-widths, the STE backward (pass-through
//// inside clip range, zero outside), and the QatLinear helper layer.

import gleam/list
import gleam/option.{None, Some}
import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/quant/qat.{
  type QuantConfig, type QuantStats, QatLinear, QuantConfig, QuantStats,
}
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// --- helpers --------------------------------------------------------------

fn sym8() -> QuantConfig {
  QuantConfig(num_bits: 8, symmetric: True, per_channel: False, channel_axis: 0)
}

fn asym8() -> QuantConfig {
  QuantConfig(
    num_bits: 8,
    symmetric: False,
    per_channel: False,
    channel_axis: 0,
  )
}

fn sym_pc8() -> QuantConfig {
  QuantConfig(num_bits: 8, symmetric: True, per_channel: True, channel_axis: 0)
}

// --- observe --------------------------------------------------------------

pub fn observe_symmetric_test() {
  let input = t.from_list([-1.0, 0.5, 1.0])
  let assert Ok(stats) = qat.observe(input, sym8())
  let scales = tensor.to_list(stats.scale)
  let zps = tensor.to_list(stats.zero_point)
  let expected_scale = 1.0 /. 127.0
  case scales {
    [s] -> {
      numerics.floats_close(s, expected_scale, 1.0e-6, 1.0e-9)
      |> should.be_true
    }
    _ -> should.fail()
  }
  case zps {
    [z] -> {
      numerics.floats_close(z, 0.0, 1.0e-9, 0.0) |> should.be_true
    }
    _ -> should.fail()
  }
}

pub fn observe_asymmetric_test() {
  // Skewed positive distribution: min=0.0, max=2.0
  let input = t.from_list([0.0, 0.5, 1.0, 1.5, 2.0])
  let assert Ok(stats) = qat.observe(input, asym8())
  let scales = tensor.to_list(stats.scale)
  let zps = tensor.to_list(stats.zero_point)
  // qmin=0, qmax=255 -> scale = 2.0/255, zp = round(0 - 0/scale) = 0
  let expected_scale = 2.0 /. 255.0
  case scales {
    [s] ->
      numerics.floats_close(s, expected_scale, 1.0e-6, 1.0e-9)
      |> should.be_true
    _ -> should.fail()
  }
  case zps {
    [z] -> numerics.floats_close(z, 0.0, 1.0e-9, 0.0) |> should.be_true
    _ -> should.fail()
  }

  // Negative-biased distribution to exercise non-zero zero_point.
  let input2 = t.from_list([-3.0, -1.0, 1.0])
  let assert Ok(stats2) = qat.observe(input2, asym8())
  let scales2 = tensor.to_list(stats2.scale)
  let zps2 = tensor.to_list(stats2.zero_point)
  let expected_scale2 = 4.0 /. 255.0
  // zp = round(0 - (-3.0)/scale) = round(3.0/scale) = 191
  case scales2 {
    [s] ->
      numerics.floats_close(s, expected_scale2, 1.0e-6, 1.0e-9)
      |> should.be_true
    _ -> should.fail()
  }
  case zps2 {
    [z] -> numerics.floats_close(z, 191.0, 0.0, 1.5) |> should.be_true
    _ -> should.fail()
  }
}

pub fn observe_per_channel_test() {
  // [2, 3] matrix; per-channel along axis 0 -> two scales.
  let assert Ok(input) = t.from_list2d([[-1.0, 0.5, 1.0], [-2.0, 1.0, 2.0]])
  let assert Ok(stats) = qat.observe(input, sym_pc8())
  let scales = tensor.to_list(stats.scale)
  let zps = tensor.to_list(stats.zero_point)
  list.length(scales) |> should.equal(2)
  list.length(zps) |> should.equal(2)
  let expected = [1.0 /. 127.0, 2.0 /. 127.0]
  numerics.lists_close(scales, expected, 1.0e-6, 1.0e-9)
  |> should.be_true
  numerics.lists_close(zps, [0.0, 0.0], 1.0e-9, 0.0)
  |> should.be_true
}

// --- fake_quant_forward ---------------------------------------------------

pub fn fake_quant_roundtrip_8bit_test() {
  let input = t.from_list([-1.0, -0.5, 0.0, 0.5, 1.0])
  let config = sym8()
  let assert Ok(stats) = qat.observe(input, config)
  let assert Ok(out) = qat.fake_quant_forward(input, stats, config)
  let got = tensor.to_list(out)
  let expected = [-1.0, -0.5, 0.0, 0.5, 1.0]
  // 8-bit symmetric quant of values within [-1, 1] should be near-identity.
  numerics.lists_close(got, expected, 1.0e-2, 1.0e-2)
  |> should.be_true
}

pub fn fake_quant_low_bits_test() {
  // 2 bits symmetric: qmax = 1, so levels are {-1, 0, 1} * scale.
  // For input range [-1, 1]: scale = 1.0 (max(|x|)/qmax = 1/1),
  // and rounding produces {-1, 0, 1}. We also want to verify that
  // {-1, -0.33, 0.33, 1} pattern emerges with 3 bits where qmax = 3.
  let input3 = t.from_list([-1.0, -0.33, 0.33, 1.0])
  let cfg3 =
    QuantConfig(
      num_bits: 3,
      symmetric: True,
      per_channel: False,
      channel_axis: 0,
    )
  let assert Ok(stats3) = qat.observe(input3, cfg3)
  let assert Ok(out3) = qat.fake_quant_forward(input3, stats3, cfg3)
  // qmax=3, scale=1/3 -> quant of [-1, -0.33, 0.33, 1] = [-3, -1, 1, 3]
  // -> dequant ≈ [-1, -0.333..., 0.333..., 1]
  let got = tensor.to_list(out3)
  let expected = [-1.0, -1.0 /. 3.0, 1.0 /. 3.0, 1.0]
  numerics.lists_close(got, expected, 1.0e-2, 1.0e-2)
  |> should.be_true
}

// --- fake_quant_backward (STE) --------------------------------------------

pub fn fake_quant_backward_passes_test() {
  let input = t.from_list([-0.5, 0.0, 0.5])
  let grad = t.from_list([0.1, 0.2, 0.3])
  let config = sym8()
  let assert Ok(stats) = qat.observe(input, config)
  let assert Ok(g_in) = qat.fake_quant_backward(grad, input, stats, config)
  let got = tensor.to_list(g_in)
  // All inputs are inside the clip range -> gradient passes through.
  numerics.lists_close(got, [0.1, 0.2, 0.3], 1.0e-9, 1.0e-12)
  |> should.be_true
}

pub fn fake_quant_backward_clip_test() {
  // Calibrate on [-1, 1] then evaluate gradient at out-of-range inputs.
  let calib = t.from_list([-1.0, 1.0])
  let config = sym8()
  let assert Ok(stats) = qat.observe(calib, config)
  // Test inputs: -2.0 and 2.0 are well outside the clip range.
  let test_input = t.from_list([-2.0, 0.0, 2.0])
  let grad = t.from_list([1.0, 1.0, 1.0])
  let assert Ok(g_in) = qat.fake_quant_backward(grad, test_input, stats, config)
  let got = tensor.to_list(g_in)
  // Outside-range values should have gradient zeroed; inside passes through.
  numerics.lists_close(got, [0.0, 1.0, 0.0], 1.0e-9, 1.0e-12)
  |> should.be_true
}

// --- compute_per_channel_scales -------------------------------------------

pub fn compute_per_channel_scales_test() {
  let assert Ok(weight) = t.from_list2d([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
  let assert Ok(scales) = qat.compute_per_channel_scales(weight, 8, 0)
  let got = tensor.to_list(scales)
  let expected = [3.0 /. 127.0, 30.0 /. 127.0]
  list.length(got) |> should.equal(2)
  numerics.lists_close(got, expected, 1.0e-6, 1.0e-9)
  |> should.be_true
}

// --- QatLinear -------------------------------------------------------------

pub fn qat_linear_forward_test() {
  // 3-in, 2-out layer with deterministic weights/bias.
  let base = qat.qat_linear_init(3, 2, 8, 8)
  let weight_data = [1.0, 0.5, -0.5, 2.0, -1.0, 1.0]
  let weight = tensor.Tensor(data: weight_data, shape: [2, 3])
  let bias = t.from_list([0.0, 0.0])
  let assert Ok(weight_stats) = qat.observe(weight, base.weight_config)
  let layer =
    QatLinear(
      ..base,
      weight: weight,
      bias: Some(bias),
      weight_stats: weight_stats,
    )
  // Single sample input.
  let assert Ok(input) = t.from_list2d([[1.0, 1.0, 1.0]])
  let assert Ok(out) = qat.qat_linear_forward(layer, input)
  let shape = tensor.shape(out)
  shape |> should.equal([1, 2])
  // Expected output (no input quant since input_stats=None):
  //   row 0 = 1.0 + 0.5 - 0.5 = 1.0
  //   row 1 = 2.0 - 1.0 + 1.0 = 2.0
  // Weight quant adds small noise; rtol=5e-2 gives plenty of slack.
  let got = tensor.to_list(out)
  numerics.lists_close(got, [1.0, 2.0], 5.0e-2, 5.0e-2)
  |> should.be_true
}

pub fn qat_linear_calibrate_then_forward_test() {
  let base = qat.qat_linear_init(2, 1, 8, 8)
  let weight = tensor.Tensor(data: [1.0, 1.0], shape: [1, 2])
  let bias = t.from_list([0.0])
  let assert Ok(weight_stats) = qat.observe(weight, base.weight_config)
  let layer =
    QatLinear(
      ..base,
      weight: weight,
      bias: Some(bias),
      weight_stats: weight_stats,
    )
  // Calibrate input_stats with a batch of activations.
  let assert Ok(calib) = t.from_list2d([[1.0, -1.0], [0.5, -0.5]])
  let assert Ok(calibrated) = qat.qat_linear_calibrate(layer, calib)
  // input_stats must now be populated.
  case calibrated.input_stats {
    Some(stats) -> {
      let scales = tensor.to_list(stats.scale)
      list.length(scales) |> should.equal(1)
    }
    None -> should.fail()
  }
  // Forward must succeed using the calibrated stats.
  let assert Ok(input) = t.from_list2d([[1.0, -1.0]])
  let assert Ok(out) = qat.qat_linear_forward(calibrated, input)
  let shape = tensor.shape(out)
  shape |> should.equal([1, 1])
  // 1.0 + (-1.0) = 0.0 give or take quant noise.
  let got = tensor.to_list(out)
  numerics.lists_close(got, [0.0], 5.0e-2, 5.0e-2)
  |> should.be_true
}

// Touch re-exported types to suppress unused-import warnings.
pub fn quant_stats_construct_test() {
  let _: QuantStats =
    QuantStats(scale: t.from_list([1.0]), zero_point: t.from_list([0.0]))
  Nil
}
