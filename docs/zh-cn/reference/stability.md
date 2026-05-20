# 稳定性策略

`viva_tensor` 将根 `viva_tensor` 模块视为稳定的用户表面。新的面向用户示例应以以下方式开始：

```gleam
import viva_tensor as t
```

这样，即使 native kernels、quantization、sparse formats 和 neural-network helpers 持续演进，
该 package 仍可作为库使用。

## 稳定表面

稳定公共表面是：

| 模块                 | 状态   | 目的                                                                                                                |
|:---------------------|:------:|:-------------------------------------------------------------------------------------------------------------------|
| `viva_tensor`        | Stable | 张量创建、数学运算、归约、布局检查、broadcasting、backend planning 和安全 fallback execution。                     |
| `viva_tensor/layout` | Stable | Canonical tensor layout metadata。                                                                                  |
| `viva_tensor/axis`   | Stable | 语义轴名称和 axis specifications。                                                                                  |
| `viva_tensor/named`  | Stable | 带 named axes 的 tensor wrapper。                                                                                   |

Stable functions 应保持 semantic-versioning 兼容性，对可恢复失败返回 `Result`，并保留纯 BEAM fallback，
除非函数明确记录为 native-only。

可失败 tensor operations 不得静默地将 backend 或 materialization failures 转换为空 tensors、zeros
或部分计算值。当必须从 native storage materialize 数据时，请在返回 `Result` 的路径中使用
`try_to_list()`。

为兼容性保留返回普通 `Tensor` 或 `Float` 的 legacy convenience functions，但当 native storage
可能参与时，新的严肃代码应优先使用 `try_map()`、`try_scale()` 和 `try_sum()` 等可失败 variants。

## 实验表面

以下区域在其契约被记录并被兼容性测试覆盖之前，均有意保持实验性：

- `viva_tensor/core/*`
- `viva_tensor/backend/*`
- direct CUDA、BLAS、sparse、quantization、neural-network、optimization、
  telemetry 和 benchmark modules
- `dev/` 下仅用于开发的 examples 和 benchmark entrypoints
- 暴露 backend-specific resource details 的 native NIF entrypoints

随着实现成熟，实验模块可能改变形状。除非你在处理 internals 或 benchmark 某个特定 backend，
否则请优先使用根模块。

目录和模块所有权规则记录在 [项目结构](project-structure.md) 中。

## Public API 护栏

每个 stable API 新增项都应包含：

- 说明函数作用的 doc comment
- 相关时的 argument 和 shape constraints
- return 和 error behavior
- 涉及 native acceleration 时的 backend selection 或 fallback behavior
- 当函数属于根模块时，至少一个 public API contract test

`test/public_api_contract_test.gleam` suite 是根 facade 的兼容性 tripwire。它通过
`import viva_tensor as t` 验证 creation、layout metadata、broadcasting、softmax、linear algebra
和 backend planning。

Broadcasting 遵循 NumPy 和 PyTorch 使用的成熟 tensor-library 约定：shapes 右对齐；
当维度相等或其中一侧为 `1` 时匹配；expanded tensors 尽可能表示为带 zero strides 的 views。

## Backend 成熟度

Pure Gleam execution 是可移植 baseline。Zig SIMD、MKL、CUDA FP32、CUDA FP16、CUDA INT8、
sparse、FP8 和 fused kernels 首先通过 capability records 和 backend planner 暴露。
在 operation contracts、dtype support、shape constraints、error behavior 和 fallback rules
记录之前，direct low-level backend modules 不应被视为 stable。
