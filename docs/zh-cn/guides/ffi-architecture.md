# FFI 架构

本页定义 `viva_tensor` FFI 代码的所有权契约。它是面向维护者的契约，不是公共 API 保证。

## 当前边界

受支持的调用路径是：

```text
public Gleam API
  -> src/viva_tensor/tensor.gleam or domain modules
  -> src/viva_tensor/core/ffi.gleam
  -> src/viva_tensor_ffi.erl, src/viva_tensor_nif.erl, or src/viva_tensor_zig.erl
  -> zig_src/
```

`src/viva_tensor/core/ffi.gleam` 仍是 Gleam 代码的单一兼容 facade。
在拆分完成并验证之前，现有调用方应继续直接导入它。

## 所有权规则

| 区域                  | 所有者                                            | 规则                                                                       |
|:----------------------|:--------------------------------------------------|:---------------------------------------------------------------------------|
| Public API            | `src/viva_tensor.gleam` and documented companions | 除非明确写入文档，否则不得暴露 native-only requirements。                  |
| Tensor behavior       | `src/viva_tensor/tensor.gleam` and domain modules | 拥有 fallback selection 和 tensor semantics。                               |
| FFI facade            | `src/viva_tensor/core/ffi.gleam`                  | 拥有 Gleam call sites 使用的稳定内部 wrapper names。                        |
| FFI split modules     | `src/viva_tensor/core/ffi/*`                      | 在 import 兼容性验证后，可拥有分组的内部 wrappers。                         |
| Erlang bridge         | `src/*_ffi.erl`, `src/*_nif.erl`, `src/*_zig.erl` | 拥有 BEAM module exports 和 NIF stubs。                                     |
| Native implementation | `zig_src/`                                        | 拥有 C、CUDA 和 Zig implementation details。                                |
| Documentation         | `docs/en/ffi-architecture.md`                     | 拥有 split contract 和 migration rules。                                    |

## 拆分契约

未来 `src/viva_tensor/core/ffi/` 下的 FFI modules 必须按 backend 或 resource family 保持不相交。
如果按任意 operation names 拆分会导致同一 resource type 上出现重复所有权，则不要这样拆分。

推荐分组：

- `core/ffi/erlang_array.gleam`：`ErlangArray` 和纯 Erlang array helpers。
- `core/ffi/math.gleam`：薄封装的 Erlang `math` 和 `rand` wrappers。
- `core/ffi/native_tensor.gleam`：`NativeTensorRef` resource constructors、element-wise operations、
  reductions、matrix operations、mutation 和 fused CPU kernels。
- `core/ffi/cuda.gleam`：CUDA tensor resource families。
- `core/ffi/research.gleam`：LNS、Horde、HDC、sparse、quantized 和 experimental native resources，
  直到它们升级为 domain-specific owner。

每个 split module 必须同时拥有自己的 resource types 和私有 `@external` bindings。
wrapper 不应位于一个模块，而其 opaque type 或匹配的 external declaration 位于另一个模块，
除非存在刻意设计的 shared type module。

## 迁移规则

1. 先添加 split modules，不改变现有 call sites。
2. 验证 Gleam 在该 package 中接受 `src/viva_tensor/core/ffi.gleam` 和
   `src/viva_tensor/core/ffi/*.gleam` 共存。
3. 每次移动一个不相交 group，并保持 `core/ffi.gleam` 作为 forwarding facade。
4. 每个 group 之后运行 formatting、type checking、no-NIF tests 和 native-path tests。
5. 只有当变更是纯机械的，并且旧 import path 仍可用时，才更新 `tensor.gleam` 或 public facade。

## Fallback 要求

Native acceleration 是可选的。新的 FFI wrappers 必须对 native failures 返回可恢复的 `Result` values，
除非它们封装的是确定性的 Erlang standard-library functions。Tensor-level code 仍负责选择
native execution，并回退到纯 Gleam behavior。

任何 split module 都不得在 package load time 要求已编译的 NIF。如果 NIF 缺失，package 仍必须可编译，
并且 no-NIF path 必须保持可测试。
