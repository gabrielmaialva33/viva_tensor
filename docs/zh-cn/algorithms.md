# 算法

## INT8 对称量化

使用线性缩放将 32 位减少到 8 位。

```mermaid
graph LR
    subgraph 输入
        F[Float32]
    end

    subgraph 处理
        S["scale = max|x| / 127"]
        Q["q = round(x / scale)"]
    end

    subgraph 输出
        I[Int8]
    end

    F --> S
    S --> Q
    Q --> I
```

**公式：**

```
量化：  q = round(x / scale)
恢复：  x' = q × scale
```

| 属性 | 值 |
|:-----|:---|
| 压缩比 | 4x |
| 典型误差 | < 0.5% |
| 速度 | 非常快 |

---

## NF4 (NormalFloat4)

针对高斯分布优化的 4 位量化。

```mermaid
graph TB
    subgraph 正态分布
        N["N(0,1)"]
    end

    subgraph 16个分位数
        Q0["-1.0"]
        Q7["0.0"]
        Q15["1.0"]
    end

    subgraph 映射
        M["值 → 最近级别"]
    end

    正态分布 --> 16个分位数
    16个分位数 --> 映射
```

**为什么有效：**
- 神经网络权重遵循高斯分布
- 零附近有更多级别
- 对正态数据数学最优

---

## AWQ（激活感知权重量化）

MLSys 2024 最佳论文。使用激活来指导量化。

```mermaid
flowchart TB
    subgraph 校准["1. 校准"]
        A[激活] --> S[每通道统计]
    end

    subgraph 分析["2. 分析"]
        S --> T["Top 1% = 显著"]
    end

    subgraph 变换["3. 变换"]
        T --> U["放大显著通道"]
        U --> Q[量化]
    end

    subgraph 推理["4. 推理"]
        Q --> D["在输入中补偿"]
    end
```

**关键洞察：**
- 约 1% 的权重是"显著的"
- 由激活幅度识别
- 量化前放大可保留信息

---

## Flash Attention

O(n) 内存注意力，而不是 O(n²)。

```mermaid
graph TB
    subgraph 标准["标准：O(n²)"]
        S1["Q×K^T"] --> S2["n×n 矩阵"]
        S2 --> S3["Softmax"]
        S3 --> S4["×V"]
    end

    subgraph Flash["Flash：O(n)"]
        F1["块 Q"] --> F2["在线 softmax"]
        F2 --> F3["累积"]
        F3 --> F1
    end
```

**技巧：在线 Softmax**

增量计算 max 和 sum，无需实现完整矩阵。

---

## 2:4 结构化稀疏

NVIDIA Tensor Cores 的稀疏模式。

```mermaid
graph LR
    subgraph 输入
        I["[a, b, c, d]"]
    end

    subgraph 剪枝
        P["将 2 个最小值置零"]
    end

    subgraph 输出
        O["[a, 0, c, 0]"]
    end

    输入 --> 剪枝
    剪枝 --> 输出
```

**规则：** 每 4 个元素中，恰好 2 个为零。

| 属性 | 值 |
|:-----|:---|
| 压缩比 | 2x |
| 硬件开销 | 零 |
| 加速 | Tensor Cores 上 2x |
