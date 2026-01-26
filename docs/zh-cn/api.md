# API 参考

## tensor

### 类型

```gleam
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
}
```

### 创建

| 函数 | 描述 |
|:-----|:-----|
| `from_list(data, shape)` | 从列表创建 |
| `zeros(shape)` | 零张量 |
| `ones(shape)` | 一张量 |
| `random_uniform(shape)` | [0,1] 随机值 |

### 操作

| 函数 | 描述 |
|:-----|:-----|
| `add(a, b)` | 逐元素加法 |
| `mul(a, b)` | 逐元素乘法 |
| `scale(t, s)` | 常数缩放 |
| `sum(t)` | 总和 |
| `mean(t)` | 平均值 |

---

## nf4

### 类型

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

### 函数

| 函数 | 描述 |
|:-----|:-----|
| `default_config()` | 默认配置 (block=64) |
| `quantize(t, config)` | 量化张量 |
| `dequantize(nf4)` | 恢复 FP32 |
| `nf4_levels()` | 列出 16 个级别 |

---

## awq

### 类型

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

### 函数

| 函数 | 描述 |
|:-----|:-----|
| `default_config()` | 默认配置 |
| `quantize_awq(w, cal, cfg)` | 带校准量化 |
| `dequantize_awq(awq)` | 恢复 FP32 |
| `collect_activation_stats(data)` | 收集统计 |

---

## flash_attention

### 类型

```gleam
pub type FlashConfig {
  FlashConfig(block_size: Int, use_causal_mask: Bool)
}
```

### 函数

| 函数 | 描述 |
|:-----|:-----|
| `default_config()` | 默认配置 |
| `flash_attention_forward(q, k, v, cfg)` | 前向传播 |

---

## sparsity

### 函数

| 函数 | 描述 |
|:-----|:-----|
| `apply_2_4_sparsity(t)` | 应用 2:4 模式 |
| `compress_sparse(s)` | 移除零 |
| `decompress_sparse(c)` | 恢复 |

---

## metrics

### 函数

| 函数 | 描述 |
|:-----|:-----|
| `compute_sqnr(orig, quant)` | SQNR（分贝） |
| `compute_mse(orig, quant)` | 均方误差 |
| `theoretical_sqnr(bits)` | 理论最大 SQNR |
