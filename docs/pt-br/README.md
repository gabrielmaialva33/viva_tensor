# viva_tensor

**[English](../en/README.md)** | **[中文](../zh-cn/README.md)**

```mermaid
graph TB
    subgraph Transform["Transformação de Memória"]
        direction LR
        A["24 GB"] -->|"×8"| B["192 GB"]
    end

    subgraph Como
        direction TB
        C[FP32] -->|quantizar| D[NF4]
        D -->|"7.5x menor"| E[Mesma Info]
    end

    Transform ~~~ Como
```

## Conceito

```mermaid
flowchart LR
    subgraph In["Entrada"]
        T[Tensor]
    end

    subgraph Compress["Compressão"]
        Q{Quantizar}
        Q -->|4x| I8[INT8]
        Q -->|8x| N4[NF4]
        Q -->|8x| AW[AWQ]
    end

    subgraph Out["Saída"]
        M[Menos Memória]
    end

    In --> Compress --> Out
```

## Início Rápido

```gleam
import viva_tensor/nf4

let compressed = nf4.quantize(tensor, nf4.default_config())
// 8x menos memória, mesma informação
```

## Performance

| Método | Compressão | Eficiência |
|:-------|:----------:|:----------:|
| INT8 | 4x | 40% |
| NF4 | 7.5x | 77% |
| AWQ | 7.7x | 53% |

**[Referência API →](api.md)**
