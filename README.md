<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:8B0000,100:006400&height=180&section=header&text=viva_tensor&fontSize=60&fontColor=fff&animation=twinkling&fontAlignY=35&desc=Memory%20×%208&descSize=20&descAlignY=55" width="100%"/>

[![Gleam](https://img.shields.io/badge/Gleam-FFAFF3?style=for-the-badge&logo=gleam&logoColor=000)](https://gleam.run/)
[![CI](https://img.shields.io/github/actions/workflow/status/gabrielmaialva33/viva_tensor/ci.yml?style=for-the-badge&label=CI)](https://github.com/gabrielmaialva33/viva_tensor/actions)
[![License](https://img.shields.io/badge/MIT-2E8B57?style=for-the-badge)](./LICENSE)

</div>

---

```mermaid
graph LR
    A["24 GB"] -->|"×8"| B["192 GB"]
```

---

## Install

```bash
gleam add viva_tensor
```

## Use

```gleam
import viva_tensor/nf4

let small = nf4.quantize(big_tensor, nf4.default_config())
// 8x less memory
```

## Algorithms

```mermaid
flowchart LR
    T[Tensor] --> Q{Quantize}
    Q -->|4x| I[INT8]
    Q -->|8x| N[NF4]
    Q -->|8x| A[AWQ]
```

| | Compression | Efficiency |
|:--|:-----------:|:----------:|
| **INT8** | 4x | 40% |
| **NF4** | 7.5x | 77% |
| **AWQ** | 7.7x | 53% |

## Build

```bash
make test
make bench
```

## Docs

[docs/](docs/) — PT-BR, EN, 中文

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:006400,100:8B0000&height=80&section=footer" width="100%"/>

</div>
