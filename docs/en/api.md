# API Reference

## tensor

### Types

```gleam
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
}
```

### Creation

| Function | Description |
|:---------|:------------|
| `from_list(data, shape)` | Create from list |
| `zeros(shape)` | Zero tensor |
| `ones(shape)` | Ones tensor |
| `random_uniform(shape)` | Values [0,1] |

### Operations

| Function | Description |
|:---------|:------------|
| `add(a, b)` | Element-wise sum |
| `mul(a, b)` | Element-wise multiply |
| `scale(t, s)` | Scale by constant |
| `sum(t)` | Total sum |
| `mean(t)` | Mean |

---

## nf4

### Types

```gleam
pub type NF4Config {
  NF4Config(block_size: Int, double_quant: Bool)
}

pub type NF4Tensor {
  NF4Tensor(
    blocks: List(NF4Block),
    shape: List(Int),
    num_elements: Int,
    memory_bytes: Int,
    compression_ratio: Float,
  )
}
```

### Functions

| Function | Description |
|:---------|:------------|
| `default_config()` | Default config (block=64) |
| `quantize(t, config)` | Quantize tensor |
| `dequantize(nf4)` | Restore FP32 |
| `nf4_levels()` | List 16 levels |

---

## awq

### Types

```gleam
pub type AWQConfig {
  AWQConfig(
    bits: Int,
    group_size: Int,
    alpha: Float,
    zero_point: Bool,
  )
}
```

### Functions

| Function | Description |
|:---------|:------------|
| `default_config()` | Default config |
| `quantize_awq(w, cal, cfg)` | Quantize with calibration |
| `dequantize_awq(awq)` | Restore FP32 |
| `collect_activation_stats(data)` | Collect stats |

---

## flash_attention

### Types

```gleam
pub type FlashConfig {
  FlashConfig(block_size: Int, use_causal_mask: Bool)
}
```

### Functions

| Function | Description |
|:---------|:------------|
| `default_config()` | Default config |
| `flash_attention_forward(q, k, v, cfg)` | Forward pass |

---

## sparsity

### Functions

| Function | Description |
|:---------|:------------|
| `apply_2_4_sparsity(t)` | Apply 2:4 pattern |
| `compress_sparse(s)` | Remove zeros |
| `decompress_sparse(c)` | Restore |

---

## metrics

### Functions

| Function | Description |
|:---------|:------------|
| `compute_sqnr(orig, quant)` | SQNR in dB |
| `compute_mse(orig, quant)` | Mean squared error |
| `theoretical_sqnr(bits)` | Theoretical max SQNR |
