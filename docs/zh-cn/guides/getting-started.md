# 入门指南

从零到运行的快速路径。纯 Gleam 路径只需要 `gleam`。CUDA 推理路径需要最近的
CUDA 工具包和 NVIDIA Ada 或更新的 GPU。

## 纯 Gleam 路径

```bash
gleam new my_app
cd my_app
gleam add viva_tensor
```

```gleam
// src/my_app.gleam
import gleam/io
import gleam/string
import viva_tensor as t

pub fn main() {
  let assert Ok(a) = t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  let assert Ok(b) = t.matrix(3, 2, [1.0, 0.0, 0.0, 1.0, 1.0, 0.0])
  let assert Ok(c) = t.matmul(a, b)
  io.println(string.inspect(t.to_list(c)))
}
```

```bash
gleam run
```

在 Gleam 支持的任何平台上都能运行。不需要 NIF。

## CUDA 路径

前置要求：

- NVIDIA GPU，建议 Ada SM89（RTX 4090）或更新
- CUDA 12.0+（`nvcc`）
- 驱动 555+
- `zig` 0.14+（构建系统）
- `g++` 14（GCC 16 在 `<functional>` 上有已知 nvcc 兼容问题；Makefile
  会自动检测 `g++-15` 作为 host 编译器）

构建：

```bash
git clone https://github.com/gabrielmaialva33/viva_tensor
cd viva_tensor
make cutlass-libs     # CUTLASS + cuSPARSELt 静态归档
make zig              # NIF .so
gleam test            # 789 个测试，NIF 加载后应全部通过
```

如果 NIF .so 不存在，`gleam test` 仍能运行 — 只是跳过原生路径。

## 运行 TinyLlama-1.1B 端到端

简短版本（完整步骤见
[`../../en/guides/inference.md`](../../en/guides/inference.md)）：

```bash
mkdir -p tmp/tinyllama && cd tmp/tinyllama
for f in model.safetensors config.json tokenizer.json tokenizer_config.json; do
  wget https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/resolve/main/$f
done
cd ../..

erlc -o /tmp dev/llama_forward.erl
erl -pa /tmp -pa build/dev/erlang/viva_tensor/ebin -noshell \
    -eval 'llama_forward:run_generate_w8a16(22, <<"Hello">>, 20, #{}, 16), halt(0).'
```

期望：模型以约 5.6 tok/s 的速度打印 "Hello" 的延续。

## 验证安装

```bash
# 检查 NIF 是否加载
erl -pa build/dev/erlang/viva_tensor/ebin -noshell -eval \
    'io:format("~p~n", [viva_tensor_zig:cuda_available()]), halt(0).'
# -> true   （CPU-only 构建为 false）

# 快速运行 CUDA matmul
gleam run -m viva_tensor/bench/peak
```

## 下一步

- [`api.md`](../api.md) — tensor API 参考（中文）。
- [`../../en/api/inference.md`](../../en/api/inference.md) — prepack /
  linear / sampling / tokenizer 参考（英文）。
- [`../../en/guides/inference.md`](../../en/guides/inference.md) —
  完整 Llama 端到端走查（英文）。
