# 项目结构

`viva_tensor` 是一个 Gleam package，包含稳定的根 facade、内部实现模块以及可选 native acceleration。
添加功能时请保持这种拆分清晰：package users 应依赖公共 Gleam contract，而不是当前的 native 或 planner internals。

## 包布局

| 路径                                                               | 目的                                                                                                                                                                 |
|:-------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `src/viva_tensor.gleam`                                            | 稳定公共 facade。用户示例应优先使用 `import viva_tensor as t`。                                                                                                      |
| `src/viva_tensor/axis.gleam`, `layout.gleam`, `named.gleam`        | 面向持久张量概念的公共 companion modules。                                                                                                                          |
| `src/viva_tensor/tensor.gleam`                                     | facade 使用的内部 tensor implementation。它拥有 pure operations、native dispatch、shape behavior 和 fallback paths。                                                |
| `src/viva_tensor/core/`                                            | 内部 storage、shape、dtype、errors、layout math 和 FFI wrappers。                                                                                                    |
| `src/viva_tensor/backend/`                                         | Planner-style selection code 使用的 backend protocol 和 capability descriptions。                                                                                    |
| `src/viva_tensor/native/`                                          | 面向 Gleam 的 native helpers，用于 BLAS、CUDA、sparse kernels 和 TFLOPS/backend diagnostics。                                                                        |
| `src/viva_tensor/quant.gleam`                                      | 内部 quantization entrypoint，重新导出支持的 quantization modules。                                                                                                  |
| `src/viva_tensor/quant/`                                           | Quantization implementations：compression、NF4、AWQ、Hadamard preprocessing、tensor-core layout helpers 和 TurboQuant reference code。                               |
| `src/viva_tensor/nn/`, `optim/`, `observability/`, `experimental/` | 内部 domain modules，直到它们的契约稳定到足以进入 public API。                                                                                                      |
| `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl`                  | BEAM target 和 NIF loading path 所需的 Erlang bridge modules。                                                                                                      |
| `zig_src/`                                                         | 可选 NIF 的 native C、CUDA 和 Zig implementation。                                                                                                                   |
| `priv/`                                                            | Erlang 在存在时加载的 runtime native artifacts。                                                                                                                     |
| `test/`                                                            | Unit、behavior、public API contract 以及 NIF/no-NIF compatibility tests。                                                                                            |
| `dev/`                                                             | 仅开发用 Gleam examples 和 benchmark entrypoints。这些模块可用 `gleam run -m ...` 运行，但不是受支持的 package API。                                               |
| `bench/`                                                           | 外部 benchmark scripts，按 runtime 或 tool 分组：`python/`、`r/`、`erlang/`、`cuda/`、`scripts/` 和 `windows/`。生成的 `data/` 和 `reports/` 保持 ignored。 |
| `docs/`                                                            | 由维护者编写的 guides 和长文档。                                                                                                                                    |

## Public API 边界

Package 边界由 `gleam.toml` 定义：根模块和少量 companion modules 是 public，而 `backend`、`core`、
`native`、`quant`、`tensor`、`nn`、`optim`、`observability` 和 `experimental` 是 internal。

只有当一个模块具备以下条件时，才将其移出 `internal_modules`：

- 已记录 shape 和 dtype behavior
- 可恢复失败用 `Result` 表示
- 通过 root facade 或 stable companion module 测试
- native acceleration 不可用时有明确的纯 Gleam behavior
- 生成的文档帮助 package users，而不只是 maintainers

Benchmarks、demos 和 research probes 在成为受支持 runtime features 之前，应放在 `dev/` 或 `bench/`。

## Pure Gleam And NIF Fallback

Native acceleration 是可选的。Public tensor operations 必须在 NIF 缺失、加载失败或返回错误时继续工作。

通常流程是：

1. Public facade 委托给内部 tensor code。
2. 内部代码在 Gleam 中验证 shape/broadcasting behavior。
3. 如果输入是 native tensors，且存在匹配的 NIF operation，则通过 `core/ffi.gleam` 和 Erlang bridge modules 尝试 native path。
4. 如果 native path 不可用或失败，operation 回退到纯 Gleam implementation。

该契约对 Hex users、CI portability 以及在没有 CUDA、MKL 或已编译 `priv/viva_tensor_zig.*` artifact 的机器上开发很重要。
真正 native-only 的函数必须在 docs 和 tests 中明确说明。

详细 FFI 所有权和拆分契约位于 [`FFI Architecture`](ffi-architecture.md)。在任何 `core/ffi/*`
split modules 于 Gleam 中验证，并按不相交 resource family 逐个迁移之前，请保持 `core/ffi.gleam`
作为 forwarding facade。

## Backend Planner

Backend selection 被拆分在小型内部层之间：

- `backend/protocol.gleam` 定义 backend types、availability checks、pure operations、local auto-selection
  和 distributed matmul hooks。
- `backend/capability.gleam` 描述 planner 可推理的内容，包括 CPU、native、CUDA 和 tensor-core capability records。
- `native/cuda.gleam` 包含面向 CUDA、MKL/native CPU 和 CPU fallback 的 higher-level acceleration planner。
  CUDA tensors 会保持在设备端，直到 API boundary 需要转换回 CPU tensors。
- `native/blas.gleam`、`native/sparse.gleam` 和 `native/tflops.gleam` 暴露 tests、benchmarks 和 planner
  decisions 使用的 backend detection 和 diagnostics。

Planner code 应保持描述性，而不是魔法化：记录选择某个 backend 的原因，保留 CPU fallback，
不要让 benchmark-only assumptions 泄漏到 stable facade。

## Quantization Layout

Quantization code 有意分层：

- `quant/compression.gleam`、`nf4.gleam` 和 `awq.gleam` 持有具体 quantization algorithms。
- `quant/hadamard.gleam` 和 `quant/turboquant.gleam` 是 Hadamard-style preprocessing 和 low-bit experiments
  在 native kernels 存在前的纯 Gleam reference paths。
- `quant/layout.gleam` 记录面向 tensor-core 的 packing assumptions，例如 block 和 tile shapes。
- 一旦 quantization contract 被纯 Gleam implementation 和 tests 锁定，`zig_src/nif_quant.c` 和
  `zig_src/` 中的 CUDA files 就是 native landing zone。

优先提供可读 reference implementation。只有在 Gleam contract、invalid-input behavior 和 no-NIF fallback
覆盖之后，才将 hot loops 移到 NIF/CUDA。

## Native Backend 位置

Native build 以 `zig_src/build.zig` 为中心。

- MKL 从 `zig_src/build.zig` 和 CPU/NIF 代码（例如 `zig_src/nif_entry.c` 和 `zig_src/nif_cpu_ops.c`）接线。
- macOS Accelerate support 位于 `zig_src/accelerate.c`。
- CUDA 和 sparse GPU work 位于 `zig_src/cuda_*.c`、`zig_src/cuda_*.cu`、
  `zig_src/nif_cuda_*.c`、`zig_src/nif_sparse.c` 和 `zig_src/sage/`。
- NIF registration 和 shared declarations 位于 `zig_src/nif_entry.c` 和 `zig_src/viva_nif.h`。

Gleam modules 只能通过现有 FFI wrappers 和 bridge modules 调用 native code。
除非 API 明确记录为 native-only，否则不要让 public APIs 依赖某个特定 native library 已安装。
