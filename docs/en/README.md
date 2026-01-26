# viva_tensor

**[Português](../pt-br/README.md)** | **[中文](../zh-cn/README.md)**

```mermaid
graph TB
    subgraph Transform["Memory Transformation"]
        direction LR
        A["24 GB"] -->|"×8"| B["192 GB"]
    end

    subgraph How
        direction TB
        C[FP32] -->|quantize| D[NF4]
        D -->|"7.5x smaller"| E[Same Info]
    end

    Transform ~~~ How
```

## Concept

```mermaid
flowchart LR
    subgraph In["Input"]
        T[Tensor]
    end

    subgraph Compress
        Q{Quantize}
        Q -->|4x| I8[INT8]
        Q -->|8x| N4[NF4]
        Q -->|8x| AW[AWQ]
    end

    subgraph Out["Output"]
        M[Less Memory]
    end

    In --> Compress --> Out
```

## Quick Start

```gleam
import viva_tensor/nf4

let compressed = nf4.quantize(tensor, nf4.default_config())
// 8x less memory, same information
```

## Performance

| Method | Compression | Efficiency |
|:-------|:-----------:|:----------:|
| INT8 | 4x | 40% |
| NF4 | 7.5x | 77% |
| AWQ | 7.7x | 53% |

**[API Reference →](api.md)**
