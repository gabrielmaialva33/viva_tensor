# Documentation

```mermaid
graph LR
    subgraph Languages
        PT[Português]
        EN[English]
        ZH[中文]
    end

    PT --> A[pt-br/]
    EN --> B[en/]
    ZH --> C[zh-cn/]
```

| | Docs | API |
|:--|:----:|:---:|
| **PT-BR** | [README](pt-br/README.md) | [api](pt-br/api.md) |
| **EN** | [README](en/README.md) | [api](en/api.md) |
| **中文** | [README](zh-cn/README.md) | [api](zh-cn/api.md) |

## Metrics

```json
{
  "int8": { "compression": "4x", "efficiency": "40%" },
  "nf4":  { "compression": "7.5x", "efficiency": "77%" },
  "awq":  { "compression": "7.7x", "efficiency": "53%" }
}
```
