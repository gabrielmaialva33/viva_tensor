//// Quantized tensor layout metadata.
////
//// This module models storage contracts separately from kernels so the public
//// API can describe NVFP4/INT2-ready tensors before hardware-specific kernels
//// exist in the NIF layer.

import gleam/int
import gleam/list
import viva_tensor/core/error.{type TensorError, InvalidShape}

pub type QuantFormat {
  QuantInt8
  QuantInt4
  QuantNf4
  QuantNvfp4
  QuantInt3
  QuantInt2
  QuantFp2
}

pub type ScaleGranularity {
  PerTensorScale
  PerBlockScale(block_size: Int)
  PerMicroBlockScale(block_size: Int)
  PerChannelScale(axis: Int)
}

pub type AccumulatorFormat {
  AccumulateFloat32
  AccumulateFloat16
  AccumulateInt32
}

pub type QuantLayout {
  QuantLayout(
    shape: List(Int),
    format: QuantFormat,
    scale_granularity: ScaleGranularity,
    accumulator: AccumulatorFormat,
    storage_bits_per_value: Int,
    has_zero_point: Bool,
    requires_hadamard: Bool,
    native_micro_block: Int,
    experimental: Bool,
  )
}

pub fn nvfp4_block_scaled(shape: List(Int)) -> QuantLayout {
  QuantLayout(
    shape: shape,
    format: QuantNvfp4,
    scale_granularity: PerMicroBlockScale(16),
    accumulator: AccumulateFloat32,
    storage_bits_per_value: 4,
    has_zero_point: False,
    requires_hadamard: False,
    native_micro_block: 16,
    experimental: True,
  )
}

pub fn int2_progressive(
  shape: List(Int),
  block_size: Int,
) -> Result(QuantLayout, TensorError) {
  case block_size > 0 {
    False ->
      Error(InvalidShape("INT2 progressive layout requires block_size > 0"))
    True ->
      Ok(QuantLayout(
        shape: shape,
        format: QuantInt2,
        scale_granularity: PerBlockScale(block_size),
        accumulator: AccumulateFloat32,
        storage_bits_per_value: 2,
        has_zero_point: True,
        requires_hadamard: True,
        native_micro_block: 16,
        experimental: True,
      ))
  }
}

pub fn int3_progressive(
  shape: List(Int),
  block_size: Int,
) -> Result(QuantLayout, TensorError) {
  case block_size > 0 {
    False ->
      Error(InvalidShape("INT3 progressive layout requires block_size > 0"))
    True ->
      Ok(QuantLayout(
        shape: shape,
        format: QuantInt3,
        scale_granularity: PerBlockScale(block_size),
        accumulator: AccumulateFloat32,
        storage_bits_per_value: 3,
        has_zero_point: True,
        requires_hadamard: True,
        native_micro_block: 16,
        experimental: True,
      ))
  }
}

pub fn element_count(shape: List(Int)) -> Int {
  list.fold(shape, 1, fn(total, dim) { total * dim })
}

pub fn memory_bits(layout: QuantLayout) -> Int {
  element_count(layout.shape) * layout.storage_bits_per_value
}

pub fn memory_bytes(layout: QuantLayout) -> Int {
  let bits = memory_bits(layout)
  case bits % 8 {
    0 -> bits / 8
    _ -> bits / 8 + 1
  }
}

pub fn compression_ratio_against(
  layout: QuantLayout,
  baseline_bits_per_value: Int,
) -> Float {
  case layout.storage_bits_per_value <= 0 {
    True -> 0.0
    False ->
      int.to_float(baseline_bits_per_value)
      /. int.to_float(layout.storage_bits_per_value)
  }
}

pub fn format_name(format: QuantFormat) -> String {
  case format {
    QuantInt8 -> "int8"
    QuantInt4 -> "int4"
    QuantNf4 -> "nf4"
    QuantNvfp4 -> "nvfp4"
    QuantInt3 -> "int3"
    QuantInt2 -> "int2"
    QuantFp2 -> "fp2"
  }
}

pub fn is_rubin_native_candidate(layout: QuantLayout) -> Bool {
  case layout.format, layout.native_micro_block {
    QuantNvfp4, 16 -> True
    QuantFp2, 16 -> True
    QuantInt2, 16 -> layout.requires_hadamard
    _, _ -> False
  }
}
