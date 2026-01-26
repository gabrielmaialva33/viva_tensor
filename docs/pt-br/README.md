# viva_tensor

Biblioteca de tensors em Pure Gleam para compressão de memória.

**[English](../en/README.md)** | **[中文](../zh-cn/README.md)**

## Conceito Central

```mermaid
graph LR
    subgraph Entrada
        A[Tensor FP32]
    end

    subgraph Compressao["Compressão"]
        B[Quantização]
    end

    subgraph Saida["Saída"]
        C[INT8 4x]
        D[NF4 8x]
    end

    A --> B
    B --> C
    B --> D
```

**Multiplicação de Memória:**

| Formato | Compressão | 24GB VRAM |
|:--------|:----------:|:----------|
| FP32 | 1x | 24 GB |
| INT8 | 4x | 96 GB |
| NF4 | 8x | 192 GB |

## Arquitetura

```mermaid
graph TB
    subgraph Core
        T[tensor.gleam]
        S[shape]
        O[ops]
    end

    subgraph Quantization
        I[int8.gleam]
        N[nf4.gleam]
        A[awq.gleam]
    end

    subgraph Optimization
        F[flash_attention.gleam]
        P[sparsity.gleam]
    end

    Core --> Quantization
    Quantization --> Optimization
```

## Documentação

| Documento | Descrição |
|:----------|:----------|
| [Início Rápido](guia-inicio.md) | Instalação e primeiros passos |
| [Algoritmos](algoritmos.md) | INT8, NF4, AWQ, Flash Attention |
| [API](api.md) | Referência completa |

## Build

```bash
make build    # Compilar
make test     # Testes
make bench    # Benchmarks
```
