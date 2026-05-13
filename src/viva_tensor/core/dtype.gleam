//// Data types (dtype) for tensors using phantom types.
////
//// Phantom types exist only at compile time. They let the type checker track
//// tensor element types without adding runtime representation or allocation.
////
//// ## Historical Context
////
//// The dtype zoo we support reflects 40 years of numerical computing evolution:
////
//// - Float32: IEEE 754 (1985) - the standard that made portable floating point possible.
////   Before this, every CPU had its own idea of what a float should be.
////
//// - Float16: Micikevicius et al. (2018) "Mixed Precision Training" proved you can
////   train with half precision and lose almost nothing. NVIDIA's V100 made it fast.
////
//// - BFloat16: a 16-bit format with Float32-like exponent range and reduced
////   mantissa precision, commonly used for ML training and inference.
////
//// - INT8: Jacob et al. (2017) "Quantization and Training of Neural Networks for
////   Efficient Integer-Arithmetic-Only Inference" - 4x compression, <1% accuracy loss.
////
//// - NF4: Dettmers et al. (2023) "QLoRA: Efficient Finetuning of Quantized LLMs" -
////   the insight that weight distributions are Gaussian, so quantize accordingly.
////
//// - AWQ: Lin et al. (2023) "AWQ: Activation-aware Weight Quantization" -
////   not all weights are equally important during quantization.
////
//// ## Example
//// ```gleam
//// import viva_tensor/core/dtype
////
//// // These types are used as type parameters:
//// // Tensor(Float32) can only interact with Tensor(Float32)
//// ```

// --- Phantom Types ---
//
// These types have no constructors and no runtime representation.
// They exist purely to make the type checker work for us.
// If you try to construct one, Gleam will rightfully complain.

/// 32-bit floating point - the workhorse of numerical computing.
///
/// 23 bits mantissa, 8 bits exponent, 1 sign bit.
/// ~7 decimal digits of precision, range 1e-38 to 3e38.
///
/// For training: always start here. Optimize later.
pub type Float32

/// 16-bit floating point (IEEE 754-2008 "binary16")
///
/// 10 bits mantissa, 5 bits exponent = ~3 decimal digits precision.
/// Range is limited (65504 max), so watch for overflow in large activations.
///
/// Performance: 2x memory bandwidth, 2-8x compute on tensor cores.
/// Use for inference when you've validated your model doesn't overflow.
pub type Float16

/// 16-bit brain floating point.
///
/// Same exponent range as Float32 (8 bits), but only 7 bits mantissa.
/// Same dynamic range as Float32 with reduced precision.
///
/// Why it works: gradient descent is remarkably noise-tolerant.
/// The accumulated weight updates average out the quantization noise.
pub type BFloat16

/// 8-bit signed integer for quantized inference.
///
/// Jacob et al. (2017) showed INT8 inference loses <1% accuracy on ImageNet.
/// That's 4x memory reduction for basically free. Production ML loves this.
///
/// Key insight: activations need dynamic quantization (computed at runtime),
/// weights can use static quantization (computed once during calibration).
///
/// Compression: 4x vs Float32
/// Speed: Up to 4x on INT8-optimized hardware (VNNI, DP4A)
pub type Int8

/// 4-bit NormalFloat - the QLoRA revolution.
///
/// Dettmers et al. (2023) observed that trained weights follow a roughly
/// normal distribution. So instead of uniform quantization levels, use
/// levels optimized for N(0,1). Brilliant in its simplicity.
///
/// The 16 quantization levels are: [-1.0, -0.6962, -0.5251, -0.3949,
/// -0.2844, -0.1848, -0.0911, 0.0, 0.0796, 0.1609, 0.2461, 0.3379,
/// 0.4407, 0.5626, 0.7230, 1.0] (normalized to absmax)
///
/// Compression: 8x vs Float32 (with block scales adding ~0.5 bits/param)
/// Why "Memory x 8": NF4 turns your 24GB GPU into a 192GB monster.
pub type NF4

/// 4-bit Activation-Aware Quantization
///
/// Lin et al. (2023) key insight: 1% of weights cause 99% of quantization error.
/// These "salient" weights correspond to channels with large activation magnitudes.
/// Protect them with higher precision, aggressively quantize the rest.
///
/// The algorithm:
/// 1. Run calibration samples through the model
/// 2. Identify channels with high activation magnitudes
/// 3. Scale those channels up before quantization (effectively giving them more bits)
/// 4. Scale them back down during inference
///
/// Compression: 8x vs Float32 (slightly better quality than naive 4-bit)
/// Trade-off: Requires calibration data, but worth it for production.
pub type AWQ

// --- Dtype Runtime Info ---
//
// Sometimes you need to know dtype properties at runtime (serialization,
// memory allocation, debugging). This struct captures the essentials.

/// Runtime dtype information for when you need to inspect types dynamically.
///
/// Use cases:
/// - Memory allocation: bytes_per_element * num_elements = buffer size
/// - Serialization: need to know how to read/write the binary format
/// - Debugging: "why is this tensor 8GB?" -> check the dtype
pub type DtypeInfo {
  DtypeInfo(
    /// Human-readable name for debugging and serialization
    name: String,
    /// Memory footprint per element (for quantized types, this is approximate
    /// as block scales add overhead - typically 0.5 bits/element for NF4/AWQ)
    bytes_per_element: Int,
    /// True for INT8, NF4, AWQ - types that require dequantization before math
    is_quantized: Bool,
  )
}

/// Float32: 4 bytes, the gold standard.
pub fn float32_info() -> DtypeInfo {
  DtypeInfo(name: "float32", bytes_per_element: 4, is_quantized: False)
}

/// Float16: 2 bytes, half the memory, half the precision (roughly).
pub fn float16_info() -> DtypeInfo {
  DtypeInfo(name: "float16", bytes_per_element: 2, is_quantized: False)
}

/// BFloat16: 2 bytes, same range as float32, worse precision.
/// Google proved this is fine for training. Who are we to argue?
pub fn bfloat16_info() -> DtypeInfo {
  DtypeInfo(name: "bfloat16", bytes_per_element: 2, is_quantized: False)
}

/// INT8: 1 byte, 4x compression, the production ML workhorse.
pub fn int8_info() -> DtypeInfo {
  DtypeInfo(name: "int8", bytes_per_element: 1, is_quantized: True)
}

/// NF4: 0.5 bytes per weight + ~0.0625 bytes block scale overhead.
/// We round up to 1 byte for the info struct (actual savings depend on block size).
/// With block_size=64, effective compression is ~7.5x vs Float32.
pub fn nf4_info() -> DtypeInfo {
  DtypeInfo(name: "nf4", bytes_per_element: 1, is_quantized: True)
}

/// AWQ: Same storage as NF4, but smarter about which weights to protect.
/// The activation-aware scaling adds negligible overhead (one scale per channel).
pub fn awq_info() -> DtypeInfo {
  DtypeInfo(name: "awq", bytes_per_element: 1, is_quantized: True)
}
