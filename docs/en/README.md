# viva_tensor

Pure Gleam tensor library focused on memory compression.

**[Português](../pt-br/README.md)** | **[中文](../zh-cn/README.md)**

## Core Concept

```mermaid
graph LR
    subgraph Input
        A[FP32 Tensor]
    end

    subgraph Compression
        B[Quantization]
    end

    subgraph Output
        C[INT8 4x]
        D[NF4 8x]
    end

    A --> B
    B --> C
    B --> D
```

**Memory Multiplication:**

| Format | Compression | 24GB VRAM |
|:-------|:----------:|:----------|
| FP32 | 1x | 24 GB |
| INT8 | 4x | 96 GB |
| NF4 | 8x | 192 GB |

## Architecture

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

## Documentation

| Document | Description |
|:---------|:------------|
| [Getting Started](getting-started.md) | Installation and first steps |
| [Algorithms](algorithms.md) | INT8, NF4, AWQ, Flash Attention |
| [API](api.md) | Complete reference |
| [Why Revolutionary](why-revolutionary.md) | Scientific benchmarks |

## Build

```bash
make build    # Compile
make test     # Tests
make bench    # Benchmarks
```
