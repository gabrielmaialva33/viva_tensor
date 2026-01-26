# Referência API

## tensor

### Tipos

```gleam
pub type Tensor {
  Tensor(data: List(Float), shape: List(Int))
}
```

### Criação

| Função | Descrição |
|:-------|:----------|
| `from_list(data, shape)` | Cria de lista |
| `zeros(shape)` | Tensor de zeros |
| `ones(shape)` | Tensor de uns |
| `random_uniform(shape)` | Valores [0,1] |

### Operações

| Função | Descrição |
|:-------|:----------|
| `add(a, b)` | Soma element-wise |
| `mul(a, b)` | Multiplicação element-wise |
| `scale(t, s)` | Escala por constante |
| `sum(t)` | Soma total |
| `mean(t)` | Média |

---

## nf4

### Tipos

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

### Funções

| Função | Descrição |
|:-------|:----------|
| `default_config()` | Config padrão (block=64) |
| `quantize(t, config)` | Quantiza tensor |
| `dequantize(nf4)` | Restaura FP32 |
| `nf4_levels()` | Lista 16 níveis |

---

## awq

### Tipos

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

### Funções

| Função | Descrição |
|:-------|:----------|
| `default_config()` | Config padrão |
| `quantize_awq(w, cal, cfg)` | Quantiza com calibração |
| `dequantize_awq(awq)` | Restaura FP32 |
| `collect_activation_stats(data)` | Coleta estatísticas |

---

## flash_attention

### Tipos

```gleam
pub type FlashConfig {
  FlashConfig(block_size: Int, use_causal_mask: Bool)
}
```

### Funções

| Função | Descrição |
|:-------|:----------|
| `default_config()` | Config padrão |
| `flash_attention_forward(q, k, v, cfg)` | Forward pass |

---

## sparsity

### Funções

| Função | Descrição |
|:-------|:----------|
| `apply_2_4_sparsity(t)` | Aplica padrão 2:4 |
| `compress_sparse(s)` | Remove zeros |
| `decompress_sparse(c)` | Restaura |

---

## metrics

### Funções

| Função | Descrição |
|:-------|:----------|
| `compute_sqnr(orig, quant)` | SQNR em dB |
| `compute_mse(orig, quant)` | Erro quadrático médio |
| `theoretical_sqnr(bits)` | SQNR máximo teórico |
