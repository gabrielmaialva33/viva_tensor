# Algorithms

## INT8 Symmetric Quantization

Reduces 32 bits to 8 bits using linear scaling.

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

**Formula:**

```
Quantization: q = round(x / scale)
Restoration:  x' = q × scale
```

| Property | Value |
|:---------|:------|
| Compression | 4x |
| Typical error | < 0.5% |
| Speed | Very fast |

---

## NF4 (NormalFloat4)

4-bit quantization optimized for Gaussian distribution.

```mermaid
graph TB
    subgraph Normal["Normal Distribution"]
        N["N(0,1)"]
    end

    subgraph Quantiles["16 Quantiles"]
        Q0["-1.0"]
        Q7["0.0"]
        Q15["1.0"]
    end

    subgraph Mapping
        M["Value → Nearest level"]
    end

    N --> Quantiles
    Quantiles --> Mapping
```

**Why it works:**
- NN weights follow Gaussian distribution
- More levels near zero
- Mathematically optimal for normal data

---

## AWQ (Activation-aware Weight Quantization)

MLSys 2024 Best Paper. Uses activations to guide quantization.

```mermaid
flowchart TB
    subgraph Calibration["1. Calibration"]
        A[Activations] --> S[Per-channel stats]
    end

    subgraph Analysis["2. Analysis"]
        S --> T["Top 1% = Salient"]
    end

    subgraph Transform["3. Transform"]
        T --> U["Scale salient UP"]
        U --> Q[Quantize]
    end

    subgraph Inference["4. Inference"]
        Q --> D["Compensate in input"]
    end
```

**Key insight:**
- ~1% of weights are "salient"
- Identified by ACTIVATION magnitude
- Scale UP before quantizing preserves info

---

## Flash Attention

O(n) memory attention instead of O(n²).

```mermaid
graph TB
    subgraph Standard["Standard: O(n²)"]
        S1["Q×K^T"] --> S2["n×n matrix"]
        S2 --> S3["Softmax"]
        S3 --> S4["×V"]
    end

    subgraph Flash["Flash: O(n)"]
        F1["Block Q"] --> F2["Online softmax"]
        F2 --> F3["Accumulate"]
        F3 --> F1
    end
```

**Trick: Online Softmax**

Compute max and sum incrementally without materializing full matrix.

---

## 2:4 Structured Sparsity

Sparsity pattern for NVIDIA Tensor Cores.

```mermaid
graph LR
    subgraph Input
        I["[a, b, c, d]"]
    end

    subgraph Prune
        P["Zero 2 smallest"]
    end

    subgraph Output
        O["[a, 0, c, 0]"]
    end

    I --> P
    P --> O
```

**Rule:** In every 4 elements, exactly 2 are zero.

| Property | Value |
|:---------|:------|
| Compression | 2x |
| HW overhead | Zero |
| Speedup | 2x on Tensor Cores |
