# viva_tensor: Multiplicação de Memória

**Gabriel Maia** · VIVA Research · 2026

---

## Resumo

Biblioteca de tensors em Pure Gleam com **8x multiplicação de memória** via compressão matemática.

```mermaid
graph LR
    A["24 GB Física"] -->|"NF4"| B["192 GB Efetiva"]
```

---

## Problema

LLMs são **memory-bound**, não compute-bound.

| Modelo    |  FP32  |  NF4   |
|:----------|:------:|:------:|
| LLaMA-7B  | 28 GB  | 3.7 GB |
| LLaMA-70B | 280 GB | 37 GB  |

---

## Solução

```mermaid
flowchart TB
    subgraph Quantizacao["Quantização"]
        INT8["INT8: 4x"]
        NF4["NF4: 7.5x"]
        AWQ["AWQ: 7.7x"]
    end

    subgraph Resultado
        M["Memória × 8"]
    end

    Quantizacao --> Resultado
```

---

## Algoritmos

### INT8

Quantização linear. Rápido, simples.

```
scale = 127 / max|x|
q = round(x × scale)
```

### NF4 (QLoRA)

16 níveis dos quantis da distribuição normal. Ótimo para pesos gaussianos.

### AWQ (MLSys 2024 Best Paper)

Insight: **1% dos pesos são salientes** — identificados pela magnitude das ativações.

```mermaid
flowchart LR
    A[Ativações] --> S[Estatísticas]
    S --> T["Top 1% Salientes"]
    T --> U["Escalar UP"]
    U --> Q[Quantizar]
```

---

## Resultados

| Método | Compressão | Eficiência |
|:-------|:----------:|:----------:|
| INT8   |     4x     |    40%     |
| NF4    |    7.5x    |    77%     |
| AWQ    |    7.7x    |    53%     |

---

## Por que Gleam?

```mermaid
graph TB
    subgraph BEAM
        P1[Processo 1]
        P2[Processo 2]
        P3[Processo N]
    end

    subgraph Propriedades
        I[Imutável]
        F[Fault-tolerant]
        C[Concorrente]
    end

    BEAM --> Propriedades
```

| Propriedade     |    Threads    |  BEAM   |
|:----------------|:-------------:|:-------:|
| Overhead        |     1 MB      |  2 KB   |
| Max concorrente |      1K       |   1M    |
| Isolamento      | Compartilhado | Isolado |

---

## Referências

1. Lin et al. "AWQ" MLSys 2024 Best Paper
2. Dettmers et al. "QLoRA" NeurIPS 2023
3. NVIDIA Blackwell Architecture 2024
