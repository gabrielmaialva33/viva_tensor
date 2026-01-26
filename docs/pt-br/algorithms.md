# Algoritmos

## INT8 Quantização Simétrica

Reduz 32 bits para 8 bits usando escala linear.

```mermaid
graph LR
    subgraph Input
        F[Float32]
    end

    subgraph Process
        S["scale = max|x| / 127"]
        Q["q = round(x / scale)"]
    end

    subgraph Output
        I[Int8]
    end

    F --> S
    S --> Q
    Q --> I
```

**Fórmula:**

```
Quantização:  q = round(x / scale)
Restauração:  x' = q × scale
```

**Características:**

| Propriedade | Valor |
|:------------|:------|
| Compressão | 4x |
| Erro típico | < 0.5% |
| Velocidade | Muito rápida |

---

## NF4 (NormalFloat4)

Quantização 4-bit otimizada para distribuição gaussiana.

```mermaid
graph TB
    subgraph Normal["Distribuição Normal"]
        N["N(0,1)"]
    end

    subgraph Quantiles["16 Quantis"]
        Q0["-1.0"]
        Q7["0.0"]
        Q15["1.0"]
    end

    subgraph Mapping
        M["Valor → Nível mais próximo"]
    end

    N --> Quantiles
    Quantiles --> Mapping
```

**Por que funciona:**
- Pesos de NNs seguem distribuição gaussiana
- Mais níveis próximos de zero
- Matematicamente ótimo para dados normais

**Níveis NF4:**

| Índice | Valor |
|:------:|:------|
| 0 | -1.0000 |
| 7 | 0.0000 |
| 15 | 1.0000 |

---

## AWQ (Activation-aware Weight Quantization)

MLSys 2024 Best Paper. Usa ativações para guiar quantização.

```mermaid
flowchart TB
    subgraph Calibration["1. Calibração"]
        A[Ativações] --> S[Estatísticas por canal]
    end

    subgraph Analysis["2. Análise"]
        S --> T["Top 1% = Salientes"]
    end

    subgraph Transform["3. Transformação"]
        T --> U["Escalar canais salientes UP"]
        U --> Q[Quantizar]
    end

    subgraph Inference["4. Inferência"]
        Q --> D["Compensar na entrada"]
    end
```

**Insight principal:**
- ~1% dos pesos são "salientes"
- Identificados pela magnitude das ATIVAÇÕES
- Escalar UP antes de quantizar preserva informação

**Matemática:**

```
W × X = (s×W) × (X/s)
       ↑        ↑
    quantiza  compensa
```

---

## Flash Attention

Atenção com O(n) memória ao invés de O(n²).

```mermaid
graph TB
    subgraph Standard["Padrão: O(n²)"]
        S1["Q×K^T"] --> S2["n×n matriz"]
        S2 --> S3["Softmax"]
        S3 --> S4["×V"]
    end

    subgraph Flash["Flash: O(n)"]
        F1["Bloco Q"] --> F2["Online softmax"]
        F2 --> F3["Acumula resultado"]
        F3 --> F1
    end
```

**Truque: Online Softmax**

Computa max e sum incrementalmente sem materializar matriz completa.

---

## 2:4 Structured Sparsity

Padrão de esparsidade para NVIDIA Tensor Cores.

```mermaid
graph LR
    subgraph Input["Entrada"]
        I["[a, b, c, d]"]
    end

    subgraph Prune["Pruning"]
        P["Zerar 2 menores"]
    end

    subgraph Output["Saída"]
        O["[a, 0, c, 0]"]
    end

    I --> P
    P --> O
```

**Regra:** Em cada 4 elementos, exatamente 2 são zero.

```
Válido:   [a, 0, b, 0]  [0, a, 0, b]  [a, b, 0, 0]
Inválido: [a, 0, 0, 0]  [a, b, c, 0]
```

| Propriedade | Valor |
|:------------|:------|
| Compressão | 2x |
| Overhead HW | Zero |
| Aceleração | 2x em Tensor Cores |
