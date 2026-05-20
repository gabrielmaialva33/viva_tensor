# 入门

从零到运行张量程序的快速路径。纯 Gleam 路径只需要 `gleam`。CUDA 推理路径需要较新的
CUDA toolkit 和 Ada 或更新的 NVIDIA GPU。

## 使用公共 API 运行模型

```bash
git clone https://github.com/gabrielmaialva33/viva_tensor
cd viva_tensor
make cutlass-libs     # CUTLASS + cuSPARSELt static archives
make zig              # the NIF .so
```

```gleam
import viva_tensor as t

pub fn main() {
  let assert Ok(model) = t.load_model("tmp/tinyllama/model.safetensors")
  let opts = t.default_generate_opts()
  let assert Ok(result) = t.generate(model, "Hello", opts)
  result.text
}
```

这是 Llama-family HF checkpoints 推荐的 v2.2.102 路径。同一个 API 已在 TinyLlama-1.1B
和 Llama-3.2-1B-Instruct 上验证。

## 纯 Gleam 张量路径

```bash
gleam new my_app
cd my_app
gleam add viva_tensor
```

```gleam
// src/my_app.gleam
import gleam/io
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

这可在任何 Gleam 支持的平台上工作。不需要 NIF。

## CUDA 推理路径

前置条件：

- NVIDIA GPU，推荐 Ada SM89 (RTX 4090) 或更新
- CUDA 12.0+ toolkit (`nvcc`)
- Driver 555+
- `zig` 0.14+（build system）
- `g++` 14（GCC 16 在 `<functional>` 上存在已知 nvcc 破坏；Makefile 会自动检测 `g++-15` 作为 host compiler）

构建：

```bash
make cutlass-libs     # CUTLASS + cuSPARSELt static archives
make zig              # the NIF .so
gleam test            # 792 tests, all should pass with NIF loaded
```

如果 NIF .so 不存在，相同的 `gleam test` 仍会运行，只是跳过 native-only paths。

## 端到端运行 TinyLlama-1.1B

完整 walkthrough 见 [`inference.md`](inference.md)。快速版本：

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

预期：模型通过公共 handle 路径以约 ~2.31 ms/token 生成 "Hello" 的续写。
`dev/llama_forward.erl` runner 保留用于高级调试和 kernel bisects。

## 验证安装

```bash
# Check NIF loaded
erl -pa build/dev/erlang/viva_tensor/ebin -noshell -eval \
    'io:format("~p~n", [viva_tensor_zig:cuda_available()]), halt(0).'
# -> true   (or false on CPU-only build)

# Run a quick CUDA matmul
gleam run -m viva_tensor/bench/peak
```

## 后续步骤

- [`inference.md`](inference.md) — 完整端到端 Llama walkthrough。
- [`../api/tensor.md`](../api/tensor.md) — 公共张量 API reference。
- [`../api/inference.md`](../api/inference.md) — prepack / linear /
  sampling / tokenizer reference。
- [`ffi-architecture.md`](ffi-architecture.md) — Gleam ↔ NIF 边界如何工作（面向维护者）。
