//// Configuration Types with Builder Pattern
////
//// Provides fluent builder APIs for configuring tensor operations.
//// Uses labelled arguments and record updates for ergonomic configuration.
////
//// ## Example
//// ```gleam
//// import viva_tensor/core/config
////
//// // Create config with defaults then customize
//// let cfg = config.conv2d()
////   |> config.with_stride(2)
////   |> config.with_padding(1)
////
//// // Or use labelled arguments
//// let cfg = config.conv2d_new(
////   kernel_h: 5,
////   kernel_w: 5,
////   stride: 2,
////   padding: 1,
//// )
//// ```

// =============================================================================
// CONV2D CONFIGURATION
// =============================================================================

/// Conv2D operation configuration
pub type Conv2dConfig {
  Conv2dConfig(
    /// Kernel height
    kernel_h: Int,
    /// Kernel width
    kernel_w: Int,
    /// Stride height
    stride_h: Int,
    /// Stride width
    stride_w: Int,
    /// Padding height
    padding_h: Int,
    /// Padding width
    padding_w: Int,
    /// Dilation height (atrous convolution)
    dilation_h: Int,
    /// Dilation width
    dilation_w: Int,
    /// Number of groups (for grouped convolution)
    groups: Int,
  )
}

/// Create default Conv2d config (3x3 kernel, stride 1, no padding)
pub fn conv2d() -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: 3,
    kernel_w: 3,
    stride_h: 1,
    stride_w: 1,
    padding_h: 0,
    padding_w: 0,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
  )
}

/// Create Conv2d config with labelled arguments
pub fn conv2d_new(
  kernel_h kernel_h: Int,
  kernel_w kernel_w: Int,
  stride stride: Int,
  padding padding: Int,
) -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: stride,
    stride_w: stride,
    padding_h: padding,
    padding_w: padding,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
  )
}

/// Create Conv2d config with "same" padding (output same size as input)
pub fn conv2d_same(kernel_h: Int, kernel_w: Int) -> Conv2dConfig {
  Conv2dConfig(
    kernel_h: kernel_h,
    kernel_w: kernel_w,
    stride_h: 1,
    stride_w: 1,
    padding_h: kernel_h / 2,
    padding_w: kernel_w / 2,
    dilation_h: 1,
    dilation_w: 1,
    groups: 1,
  )
}

// Builder methods for Conv2dConfig
/// Set uniform stride
pub fn with_stride(config: Conv2dConfig, stride: Int) -> Conv2dConfig {
  Conv2dConfig(..config, stride_h: stride, stride_w: stride)
}

/// Set separate strides
pub fn with_stride_hw(
  config: Conv2dConfig,
  stride_h: Int,
  stride_w: Int,
) -> Conv2dConfig {
  Conv2dConfig(..config, stride_h: stride_h, stride_w: stride_w)
}

/// Set uniform padding
pub fn with_padding(config: Conv2dConfig, padding: Int) -> Conv2dConfig {
  Conv2dConfig(..config, padding_h: padding, padding_w: padding)
}

/// Set separate paddings
pub fn with_padding_hw(
  config: Conv2dConfig,
  padding_h: Int,
  padding_w: Int,
) -> Conv2dConfig {
  Conv2dConfig(..config, padding_h: padding_h, padding_w: padding_w)
}

/// Set uniform dilation
pub fn with_dilation(config: Conv2dConfig, dilation: Int) -> Conv2dConfig {
  Conv2dConfig(..config, dilation_h: dilation, dilation_w: dilation)
}

/// Set number of groups
pub fn with_groups(config: Conv2dConfig, groups: Int) -> Conv2dConfig {
  Conv2dConfig(..config, groups: groups)
}

/// Set kernel size
pub fn with_kernel(
  config: Conv2dConfig,
  kernel_h: Int,
  kernel_w: Int,
) -> Conv2dConfig {
  Conv2dConfig(..config, kernel_h: kernel_h, kernel_w: kernel_w)
}

// =============================================================================
// POOLING CONFIGURATION
// =============================================================================

/// Pooling operation configuration
pub type PoolConfig {
  PoolConfig(
    /// Pool height
    pool_h: Int,
    /// Pool width
    pool_w: Int,
    /// Stride height
    stride_h: Int,
    /// Stride width
    stride_w: Int,
    /// Padding height
    padding_h: Int,
    /// Padding width
    padding_w: Int,
  )
}

/// Create default pool config (2x2 pool, stride 2, no padding)
pub fn pool() -> PoolConfig {
  PoolConfig(
    pool_h: 2,
    pool_w: 2,
    stride_h: 2,
    stride_w: 2,
    padding_h: 0,
    padding_w: 0,
  )
}

/// Create pool config with labelled arguments
pub fn pool_new(pool_size pool_size: Int, stride stride: Int) -> PoolConfig {
  PoolConfig(
    pool_h: pool_size,
    pool_w: pool_size,
    stride_h: stride,
    stride_w: stride,
    padding_h: 0,
    padding_w: 0,
  )
}

/// Set pool size
pub fn pool_with_size(
  config: PoolConfig,
  pool_h: Int,
  pool_w: Int,
) -> PoolConfig {
  PoolConfig(..config, pool_h: pool_h, pool_w: pool_w)
}

/// Set pool stride
pub fn pool_with_stride(config: PoolConfig, stride: Int) -> PoolConfig {
  PoolConfig(..config, stride_h: stride, stride_w: stride)
}

/// Set pool padding
pub fn pool_with_padding(config: PoolConfig, padding: Int) -> PoolConfig {
  PoolConfig(..config, padding_h: padding, padding_w: padding)
}

// =============================================================================
// QUANTIZATION CONFIGURATION
// =============================================================================

/// NF4 quantization configuration
pub type NF4Config {
  NF4Config(
    /// Block size for quantization (typically 64)
    block_size: Int,
    /// Whether to use double quantization (QLoRA style)
    double_quant: Bool,
  )
}

/// Create default NF4 config
pub fn nf4() -> NF4Config {
  NF4Config(block_size: 64, double_quant: True)
}

/// Set NF4 block size
pub fn nf4_with_block_size(config: NF4Config, block_size: Int) -> NF4Config {
  NF4Config(..config, block_size: block_size)
}

/// Enable/disable double quantization
pub fn nf4_with_double_quant(config: NF4Config, enabled: Bool) -> NF4Config {
  NF4Config(..config, double_quant: enabled)
}

/// INT8 quantization configuration
pub type Int8Config {
  Int8Config(
    /// Block size for per-block quantization (0 = per-tensor)
    block_size: Int,
    /// Whether to use symmetric quantization
    symmetric: Bool,
  )
}

/// Create default INT8 config
pub fn int8() -> Int8Config {
  Int8Config(block_size: 0, symmetric: True)
}

/// Set INT8 block size
pub fn int8_with_block_size(config: Int8Config, block_size: Int) -> Int8Config {
  Int8Config(..config, block_size: block_size)
}

/// AWQ quantization configuration
pub type AWQConfig {
  AWQConfig(
    /// Block size
    block_size: Int,
    /// Number of calibration samples
    n_calibration: Int,
    /// Scaling factor for salient channels
    scale_factor: Float,
  )
}

/// Create default AWQ config
pub fn awq() -> AWQConfig {
  AWQConfig(block_size: 64, n_calibration: 128, scale_factor: 1.0)
}

/// Set AWQ block size
pub fn awq_with_block_size(config: AWQConfig, block_size: Int) -> AWQConfig {
  AWQConfig(..config, block_size: block_size)
}

/// Set AWQ calibration samples
pub fn awq_with_calibration(config: AWQConfig, n: Int) -> AWQConfig {
  AWQConfig(..config, n_calibration: n)
}

// =============================================================================
// ATTENTION CONFIGURATION
// =============================================================================

/// Flash Attention configuration
pub type AttentionConfig {
  AttentionConfig(
    /// Number of attention heads
    num_heads: Int,
    /// Head dimension (d_k)
    head_dim: Int,
    /// Dropout probability (0.0 = no dropout)
    dropout: Float,
    /// Whether to use causal (autoregressive) masking
    causal: Bool,
    /// Softmax scale factor (typically 1/sqrt(head_dim))
    scale: Float,
  )
}

/// Create default attention config
pub fn attention(
  num_heads num_heads: Int,
  head_dim head_dim: Int,
) -> AttentionConfig {
  let scale = 1.0 /. sqrt(int_to_float(head_dim))
  AttentionConfig(
    num_heads: num_heads,
    head_dim: head_dim,
    dropout: 0.0,
    causal: False,
    scale: scale,
  )
}

/// Enable causal masking
pub fn attention_causal(config: AttentionConfig) -> AttentionConfig {
  AttentionConfig(..config, causal: True)
}

/// Set dropout probability
pub fn attention_with_dropout(
  config: AttentionConfig,
  dropout: Float,
) -> AttentionConfig {
  AttentionConfig(..config, dropout: dropout)
}

/// Set custom scale factor
pub fn attention_with_scale(
  config: AttentionConfig,
  scale: Float,
) -> AttentionConfig {
  AttentionConfig(..config, scale: scale)
}

// =============================================================================
// HELPERS
// =============================================================================

@external(erlang, "math", "sqrt")
fn sqrt(x: Float) -> Float

fn int_to_float(i: Int) -> Float {
  case i >= 0 {
    True -> positive_int_to_float(i, 0.0)
    False -> 0.0 -. positive_int_to_float(0 - i, 0.0)
  }
}

fn positive_int_to_float(i: Int, acc: Float) -> Float {
  case i {
    0 -> acc
    _ -> positive_int_to_float(i - 1, acc +. 1.0)
  }
}
