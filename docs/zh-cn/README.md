# viva_tensor

**[English](../en/README.md)** | **[Português](../pt-br/README.md)**

Gleam/BEAM 的张量库，配备纯 Gleam 公共 API 和可选的 CUDA + CUTLASS NIF，
用于高吞吐量推理。同样的 `import viva_tensor as t` 代码在没有 CUDA 的笔
记本和具有 FP8 Tensor Cores 的 RTX 4090 上都能运行。

> **规范语言**：完整文档位于 [`docs/en/`](../en/README.md)。这个文件夹
> 用中文涵盖核心内容 — 如果需要更多页面的翻译请提 issue 或 PR。

## 现已提供

| 子系统                                | 状态         | 亮点                                                                          |
| :----------------------------------- | :----------- | :---------------------------------------------------------------------------- |
| 纯 Gleam tensor API                   | 稳定         | shape、广播、基础 autograd、命名轴、纯 BEAM fallback。                          |
| FP8 密集（CUTLASS + cuBLASLt）         | 生产级       | RTX 4090 上约 588 TFLOPS。FP32 输出缓冲区（无 FP16 饱和）。                    |
| FP8 W8A16 + per-block-16 缩放         | 生产级       | 关闭与 HF transformers 的数值差距；argmax token 与 fp32 参考相同。              |
| Fused SwiGLU（NIF）                   | 生产级       | 单 kernel 完成 `silu(gate)·up` + per-channel dequant。                         |
| INT8 2:4 稀疏（cuSPARSELt）           | 生产级       | 约 1320 TOPS。通过 reorder_meta shim 实现字节精确元数据。                       |
| INT4 2:4 稀疏（CUTLASS Sm80）          | 生产级       | 约 1854 TOPS。kernel + reorder + 编码端到端字节精确自测。                       |
| SafeTensors 加载器                    | 可用         | bf16 → fp32，通过 NIF 转置（TinyLlama 加载从 3 分钟降到 25 秒）。                |
| BPE Tokenizer                         | 可用         | encode/decode 与 HuggingFace transformers 比特精确匹配。                       |
| Llama-1.1B 端到端 forward             | 可用         | 22 层 + RoPE + GQA + KV cache + LM head + argmax。argmax token 与 HF 一致。  |
| 高级采样（temp/top-k/p）              | 可用         | 多项式采样，支持可重现的 seed。                                                 |

## 快速开始

```bash
gleam add viva_tensor
```

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(a) = t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let assert Ok(b) = t.matrix(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
  let assert Ok(c) = t.matmul(a, b)
  c |> t.to_list
}
```

推理路径（需要 CUDA）：

```gleam
let assert Ok(packed) = t.prepack_fp8_weight(weight_tensor)
let assert Ok(logits) = t.linear_fp8(input, packed, None)
```

完整的安装 + 第一个程序见 [`guides/getting-started.md`](guides/getting-started.md)。

## 中文页面

| 页面                                                       | 内容                                                     |
| :--------------------------------------------------------- | :------------------------------------------------------- |
| [`guides/getting-started.md`](guides/getting-started.md)   | 安装、构建、运行第一个程序。                              |
| [`api.md`](api.md)                                         | 纯 Gleam tensor API 快速参考。                            |
| [`paper.md`](paper.md)                                     | 技术论文（中文版）。                                      |

未翻译的主题（推理端到端指南、FP8/INT4 参考、FFI 契约）请见
[`docs/en/`](../en/README.md)。
