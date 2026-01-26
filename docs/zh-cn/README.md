# viva_tensor

```mermaid
graph TB
    subgraph Transform["内存转换"]
        direction LR
        A["24 GB"] -->|"×8"| B["192 GB"]
    end

    subgraph How["原理"]
        direction TB
        C[FP32] -->|量化| D[NF4]
        D -->|"7.5x 更小"| E[相同信息]
    end

    Transform ~~~ How
```

## 概念

```mermaid
flowchart LR
    subgraph In["输入"]
        T[张量]
    end

    subgraph Compress["压缩"]
        Q{量化}
        Q -->|4x| I8[INT8]
        Q -->|8x| N4[NF4]
        Q -->|8x| AW[AWQ]
    end

    subgraph Out["输出"]
        M[更少内存]
    end

    In --> Compress --> Out
```

## 快速开始

```gleam
import viva_tensor/nf4

let compressed = nf4.quantize(tensor, nf4.default_config())
// 8x 更少内存，相同信息
```

## 性能

```mermaid
xychart-beta
    title "压缩比 vs 效率"
    x-axis [INT8, NF4, AWQ]
    y-axis "效率 %" 0 --> 100
    bar [40, 77, 53]
```

| 方法 | 压缩比 | 效率 |
|:-----|:------:|:----:|
| INT8 | 4x | 40% |
| NF4 | 7.5x | 77% |
| AWQ | 7.7x | 53% |

**[API 参考 →](api.md)**
