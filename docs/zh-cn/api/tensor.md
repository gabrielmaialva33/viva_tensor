# API 指南

`viva_tensor` 从根 `viva_tensor` 模块暴露一个小而稳定的表面。面向 native backends、
quantization、sparse kernels、telemetry、benchmarking 以及实验性 neural-network helpers 的实现模块，
在契约稳定之前都保持内部可见。
仅用于开发的 benchmark 和 example entrypoints 位于 `dev/` 下，因此可以在本地运行，
但不会成为打包库表面的一部分。

有关稳定/实验边界以及根模块新增项的兼容性预期，见 [稳定性策略](stability.md)。
有关包布局和模块边界规则，见 [项目结构](project-structure.md)。

## 稳定导入

```gleam
import gleam/result
import viva_tensor as t
```

常规张量工作请使用根模块。需要检查 storage metadata 时导入 `viva_tensor/layout`，
处理语义维度时导入 `viva_tensor/axis` 或 `viva_tensor/named`。

## 张量创建

| 函数                                      | 描述                                                 |
|:-----------------------------------------|:----------------------------------------------------|
| `zeros(shape)`                           | 创建填充为零的张量。                               |
| `ones(shape)`                            | 创建填充为一的张量。                               |
| `fill(shape, value)`                     | 创建填充为单个标量值的张量。                       |
| `from_list(data)`                        | 创建一维张量。                                     |
| `from_list2d(rows)`                      | 从 rows 创建矩阵，并验证 row 大小。                 |
| `linspace(start, stop, steps)`           | 在闭区间上创建均匀间隔的值。                       |
| `try_linspace(start, stop, steps)`       | 可失败 linspace，拒绝无效 step counts。             |
| `logspace(start, stop, steps, base)`     | 创建对数间隔的值。                                 |
| `try_logspace(start, stop, steps, base)` | 可失败 logspace，拒绝无效 steps/base。              |
| `zeros_like(tensor)`                     | 创建与原张量 shape 相同的零张量。                   |
| `ones_like(tensor)`                      | 创建与原张量 shape 相同的一张量。                   |
| `full_like(tensor, value)`               | 创建与原张量 shape 相同的填充张量。                 |
| `eye(n)` / `identity(n)`                 | 创建方形 identity matrix。                          |
| `try_eye(n)`                             | 可失败 identity matrix，拒绝无效大小。              |
| `diag(tensor)`                           | 从 vector 创建对角矩阵。                            |
| `try_diag(tensor)`                       | 可失败的对角矩阵创建。                             |
| `matrix(rows, cols, data)`               | 使用显式维度创建矩阵。                             |

```gleam
let a = t.zeros([2, 3])
let b = t.fill([2, 3], 1.5)
```

## 可失败操作

改变 shape 的操作和二元张量操作返回 `Result`，而不是 panic。可使用 `gleam/result.try`
串联它们。在可失败代码中，如果需要保留 native materialization failures，请使用
`try_to_list()` 而不是 `to_list()`。

```gleam
pub fn example() {
  let a = t.ones([2, 3])
  let b = t.fill([2, 3], 2.0)

  use c <- result.try(t.add(a, b))
  use flat <- result.try(t.reshape(c, [6]))

  Ok(t.mean(flat))
}
```

## 逐元素数学

| 函数                                             | 描述                                                               |
|:------------------------------------------------|:------------------------------------------------------------------|
| `add(a, b)`                                     | 相同 shape 的逐元素加法。                                         |
| `sub(a, b)`                                     | 相同 shape 的逐元素减法。                                         |
| `mul(a, b)`                                     | 相同 shape 的逐元素乘法。                                         |
| `div(a, b)`                                     | 相同 shape 的逐元素除法。                                         |
| `scale(tensor, scalar)`                         | 将每个元素乘以标量。                                             |
| `try_scale(tensor, scalar)`                     | 可失败标量乘法，保留 native materialization errors。              |
| `add_scalar(tensor, scalar)`                    | 给每个元素加上标量。                                             |
| `try_add_scalar(tensor, scalar)`                | 可失败标量加法，保留 native materialization errors。              |
| `negate(tensor)`                                | 对每个元素取负。                                                 |
| `try_negate(tensor)`                            | 可失败取负，保留 native materialization errors。                  |
| `clamp(tensor, min, max)`                       | 将值 clamp 到闭区间。                                             |
| `try_clamp(tensor, min, max)`                   | 可失败 clamp，保留 native materialization errors。                 |
| `clip(tensor, min, max)`                        | 将值 clamp 到闭区间的别名。                                       |
| `try_clip(tensor, min, max)`                    | 可失败 clip，拒绝无效区间。                                       |
| `abs(tensor)` / `try_abs(tensor)`               | 每个元素的绝对值。                                               |
| `square(tensor)` / `try_square(tensor)`         | 每个元素平方。                                                   |
| `sqrt(tensor)` / `try_sqrt(tensor)`             | 平方根；`try_sqrt` 拒绝负值。                                    |
| `exp(tensor)` / `try_exp(tensor)`               | 每个元素的指数。                                                 |
| `log(tensor)` / `try_log(tensor)`               | 自然对数；`try_log` 拒绝非正值。                                 |
| `floor(tensor)` / `try_floor(tensor)`           | 每个元素向下取整。                                               |
| `ceil(tensor)` / `try_ceil(tensor)`             | 每个元素向上取整。                                               |
| `round(tensor)` / `try_round(tensor)`           | 每个元素四舍五入到最近整数值。                                   |
| `sign(tensor)` / `try_sign(tensor)`             | 对每个元素返回 -1、0 或 1。                                      |
| `reciprocal(tensor)` / `try_reciprocal(tensor)` | 倒数；`try_reciprocal` 拒绝零值。                                |
| `map(tensor, fun)`                              | 对每个元素应用标量函数。                                         |
| `try_map(tensor, fun)`                          | 可失败标量映射，保留 native materialization errors。              |
| `softmax_axis(tensor, axis)`                    | 沿一个 axis 归一化每个 slice。                                   |
| `try_softmax_axis(tensor, axis)`                | 可失败 softmax，保留 native materialization 和 indexing errors。  |

当 shape 不同时，请使用 broadcasting 专用函数。

| 函数                                             | 描述                                               |
|:------------------------------------------------|:--------------------------------------------------|
| `can_broadcast(a, b)`                           | 检查两个 shapes 是否兼容。                         |
| `broadcast_shape(a, b)`                         | 计算两个 shapes 的公共 shape。                     |
| `broadcast_shapes(shapes)`                      | 计算多个 shapes 的公共 shape。                     |
| `broadcast_to(tensor, shape)`                   | 在可能时创建 broadcast view。                      |
| `broadcast_pair(a, b)`                          | 将两个张量 broadcast 为公共 views。                |
| `add_broadcast(a, b)`                           | 使用 NumPy 风格 broadcasting 做加法。              |
| `sub_broadcast(a, b)`                           | 使用 NumPy 风格 broadcasting 做减法。              |
| `mul_broadcast(a, b)`                           | 使用 NumPy 风格 broadcasting 做乘法。              |
| `div_broadcast(a, b)`                           | 使用 NumPy 风格 broadcasting 做除法。              |
| `maximum(a, b)`                                 | 带 broadcasting 的逐元素最大值。                   |
| `minimum(a, b)`                                 | 带 broadcasting 的逐元素最小值。                   |
| `equal(a, b)` / `not_equal(a, b)`               | 带 broadcasting 的逐元素相等 mask。                |
| `greater(a, b)` / `greater_equal(a, b)`         | 带 broadcasting 的逐元素比较 mask。                |
| `less(a, b)` / `less_equal(a, b)`               | 带 broadcasting 的逐元素比较 mask。                |
| `where(condition, true, false)`                 | 使用非零 condition mask 选择值。                   |
| `logical_not(mask)`                             | 反转 numeric mask。                                |
| `logical_and(a, b)` / `logical_or(a, b)`        | 使用 broadcasting 组合 numeric masks。             |
| `logical_xor(a, b)`                             | 对 numeric masks 做 exclusive-or。                 |
| `any(mask)` / `all(mask)`                       | 将 numeric mask 归约为 boolean。                   |
| `count_nonzero(tensor)`                         | 统计非零张量值数量。                              |
| `any_axis(mask, axis)` / `all_axis(mask, axis)` | 沿一个 axis 归约 numeric masks。                   |
| `count_nonzero_axis(tensor, axis)`              | 沿一个 axis 统计非零值。                           |
| `take(tensor, indices)`                         | 通过显式 indices 取 flattened values。             |
| `nonzero(tensor)`                               | 以 floats 返回 flattened non-zero indices。        |
| `masked_select(tensor, mask)`                   | 使用 broadcast mask 选择 flattened values。        |

## 归约

| 函数                                    | 描述                                                                 |
|:---------------------------------------|:--------------------------------------------------------------------|
| `sum(tensor)`                          | 对所有元素求和。                                                     |
| `try_sum(tensor)`                      | 可失败求和，保留 native materialization errors。                      |
| `sum_axis(tensor, axis)`               | 沿一个 axis 求和。                                                    |
| `try_sum_axis(tensor, axis)`           | 可失败地沿一个 axis 求和。                                            |
| `sum_axis_keepdims(tensor, axis)`      | 沿一个 axis 求和，同时保留 size-1 维度。                              |
| `mean(tensor)`                         | 所有元素的均值。                                                      |
| `try_mean(tensor)`                     | 可失败均值，保留 materialization 和 empty-tensor errors。             |
| `product(tensor)`                      | 所有元素的乘积。                                                      |
| `try_product(tensor)`                  | 可失败乘积，保留 materialization errors。                             |
| `cumsum(tensor)`                       | flattened values 的累积和，并保留 shape。                            |
| `try_cumsum(tensor)`                   | 可失败累积和，保留 materialization errors。                           |
| `cumsum_axis(tensor, axis)`            | 沿一个 axis 累积求和，并保留 shape。                                  |
| `try_cumsum_axis(tensor, axis)`        | 可失败地沿一个 axis 累积求和。                                        |
| `cumprod(tensor)`                      | flattened values 的累积乘积，并保留 shape。                          |
| `try_cumprod(tensor)`                  | 可失败累积乘积，保留 materialization errors。                         |
| `cumprod_axis(tensor, axis)`           | 沿一个 axis 累积乘积，并保留 shape。                                  |
| `try_cumprod_axis(tensor, axis)`       | 可失败地沿一个 axis 累积乘积。                                        |
| `median(tensor)`                       | 所有元素的中位数。                                                    |
| `try_median(tensor)`                   | 可失败中位数，保留 materialization 和 empty-tensor errors。           |
| `percentile(tensor, percentile)`       | 使用线性插值的 percentile。                                           |
| `try_percentile(tensor, percentile)`   | 可失败 percentile，具有显式边界和 empty-tensor errors。               |
| `mean_axis(tensor, axis)`              | 沿一个 axis 求均值。                                                  |
| `try_mean_axis(tensor, axis)`          | 可失败地沿一个 axis 求均值。                                          |
| `mean_axis_keepdims(tensor, axis)`     | 沿一个 axis 求均值，同时保留 size-1 维度。                            |
| `variance_axis(tensor, axis)`          | 沿一个 axis 求方差。                                                  |
| `try_variance_axis(tensor, axis)`      | 可失败地沿一个 axis 求方差。                                          |
| `variance_axis_keepdims(tensor, axis)` | 沿一个 axis 求方差，同时保留 size-1 维度。                            |
| `std_axis(tensor, axis)`               | 沿一个 axis 求标准差。                                                |
| `try_std_axis(tensor, axis)`           | 可失败地沿一个 axis 求标准差。                                        |
| `std_axis_keepdims(tensor, axis)`      | 沿一个 axis 求标准差，同时保留 size-1 维度。                          |
| `max_axis(tensor, axis)`               | 沿一个 axis 求最大值。                                                |
| `try_max_axis(tensor, axis)`           | 可失败地沿一个 axis 求最大值。                                        |
| `max_axis_keepdims(tensor, axis)`      | 沿一个 axis 求最大值，同时保留 size-1 维度。                          |
| `min_axis(tensor, axis)`               | 沿一个 axis 求最小值。                                                |
| `try_min_axis(tensor, axis)`           | 可失败地沿一个 axis 求最小值。                                        |
| `min_axis_keepdims(tensor, axis)`      | 沿一个 axis 求最小值，同时保留 size-1 维度。                          |
| `argmax_axis(tensor, axis)`            | 沿一个 axis 的 argmax index，以 floats 表示。                         |
| `try_argmax_axis(tensor, axis)`        | 可失败地沿一个 axis 求 argmax index。                                 |
| `argmin_axis(tensor, axis)`            | 沿一个 axis 的 argmin index，以 floats 表示。                         |
| `try_argmin_axis(tensor, axis)`        | 可失败地沿一个 axis 求 argmin index。                                 |
| `max(tensor)`                          | 最大值。                                                              |
| `try_max(tensor)`                      | 可失败最大值，保留 materialization 和 empty-tensor errors。           |
| `min(tensor)`                          | 最小值。                                                              |
| `try_min(tensor)`                      | 可失败最小值，保留 materialization 和 empty-tensor errors。           |
| `argmax(tensor)`                       | 最大值的 flat index。                                                 |
| `try_argmax(tensor)`                   | 可失败地求最大值的 flat index。                                       |
| `argmin(tensor)`                       | 最小值的 flat index。                                                 |
| `try_argmin(tensor)`                   | 可失败地求最小值的 flat index。                                       |
| `variance(tensor)`                     | 所有元素的方差。                                                      |
| `try_variance(tensor)`                 | 可失败方差，保留 materialization 和 empty-tensor errors。             |
| `std(tensor)`                          | 所有元素的标准差。                                                    |
| `try_std(tensor)`                      | 可失败标准差，保留 materialization 和 empty-tensor errors。           |

## 线性代数

| 函数                         | 描述                                                           |
|:-----------------------------|:--------------------------------------------------------------|
| `dot(a, b)`                  | vectors 的 dot product。                                      |
| `matmul(a, b)`               | 矩阵乘法。                                                     |
| `matmul_planned(a, b)`       | 使用稳定 backend planner 并带 fallback 的矩阵乘法。            |
| `matmul_vec(matrix, vector)` | Matrix-vector multiplication。                                |
| `transpose(tensor)`          | 矩阵转置。                                                     |
| `outer(a, b)`                | Outer product。                                                |

根模块提供 native-backed variants，例如 `matmul_into`、`to_accelerated` 和
`matmul_accelerated_into`，用于可复用 buffers 或 persistent GPU memory 的 hot paths。

使用 `capabilities()` 检查当前 VM 是否加载了 native NIF、Zig SIMD backend、哪些 TFLOPS
backends 可见，以及稳定 backend capability records。只需要 capability table 时使用
`backend_capabilities()`；想查看稳定 planner 会为某个 operation 选择哪个 backend 时使用
`plan_backend(operation)`。Plans 会包含 `rejected` backend entries，并带有人类可读的原因，
说明 backend 不可用或不适合。
提前规划 accelerator-specific 工作时使用 `hardware_profiles()`：只有检测到的当前硬件才会标记为
available，而 Blackwell、Rubin、Vera 和 Rubin CPX 等 future profiles 会保持显式存在但
unavailable，直到 runtime path 能证明支持。

```gleam
let plan = t.plan_backend(t.OperationMatmul(m: 1024, n: 1024, k: 1024))
```

## 量化就绪能力

| 函数                                                   | 描述                                                       |
|:-------------------------------------------------------|:----------------------------------------------------------|
| `nvfp4_block_scaled_layout(shape)`                     | 描述 Rubin-style NVFP4 micro-block layout。               |
| `int2_progressive_layout(shape, block_size)`           | 描述实验性 INT2 progressive quantization layout。         |
| `int3_progressive_layout(shape, block_size)`           | 描述实验性 INT3 progressive quantization layout。         |
| `quant_layout_memory_bytes(layout)`                    | 估算 quantized layout 的 payload bytes。                  |
| `quant_layout_compression_ratio_against(layout, bits)` | 估算相对 baseline element width 的压缩率。                |
| `quant_layout_is_rubin_native_candidate(layout)`       | 检查 layout 是否匹配 Rubin micro-block assumptions。      |
| `try_hadamard_preprocess(tensor, seed)`                | 对 vector 应用可逆随机 Hadamard preprocessing。           |
| `try_inverse_hadamard_preprocess(plan)`                | 在 Hadamard preprocessing 后恢复 vector。                 |
| `try_normalized_walsh_hadamard(values)`                | 使用 normalized WHT 变换 power-of-two vector data。       |

## Shape 与布局

| 函数                          | 描述                                                           |
|:------------------------------|:--------------------------------------------------------------|
| `shape(tensor)`               | 张量维度。                                                     |
| `size(tensor)`                | 总元素数。                                                     |
| `rank(tensor)`                | 维度数量。                                                     |
| `reshape(tensor, shape)`      | 在保持元素数量不变的情况下改变 shape。                         |
| `device(tensor)`              | Payload device class。                                        |
| `dtype(tensor)`               | 张量元素类型。                                                 |
| `try_to_list(tensor)`         | Materialize tensor data，同时保留 native failures。            |
| `flatten(tensor)`             | 转换为一维。                                                   |
| `try_flatten(tensor)`         | 可失败 flatten，保留 materialization failures。                |
| `squeeze(tensor)`             | 移除 size-one dimensions。                                     |
| `unsqueeze(tensor, axis)`     | 插入 size-one dimension。                                      |
| `try_unsqueeze(tensor, axis)` | 可失败 unsqueeze，保留 invalid-axis errors。                   |
| `to_strided(tensor)`          | 将 dense data 转换为 zero-copy strided view。                  |
| `try_to_strided(tensor)`      | 可失败 strided conversion，保留 native materialization errors。 |
| `to_contiguous(tensor)`       | 将 strided view materialize 为 contiguous dense storage。      |
| `try_to_contiguous(tensor)`   | 可失败 contiguous conversion，保留 materialization errors。    |
| `layout(tensor)`              | 检查 storage、device、dtype、strides、offset 和 contiguity。   |

```gleam
let info = t.layout(t.zeros([2, 3]))
```

Broadcasting、squeeze、unsqueeze 和 contiguous reshape 会在可能时保留 strided views。
如果 view 比 dense buffer 更慢，请在重型 native hot path 前调用 `to_contiguous()`。

## 工具函数

| 函数                              | 描述                                                     |
|:---------------------------------|:--------------------------------------------------------|
| `norm(tensor)`                   | L2 norm。                                               |
| `try_norm(tensor)`               | 可失败 L2 norm，保留 materialization errors。           |
| `normalize(tensor)`              | 归一化到 unit length。                                  |
| `try_normalize(tensor)`          | 可失败归一化，保留 materialization errors。             |
| `abs(tensor)`                    | 每个元素的绝对值。                                      |
| `square(tensor)`                 | 每个元素平方。                                          |
| `sqrt(tensor)`                   | 每个元素的平方根。                                      |
| `try_sqrt(tensor)`               | 可失败平方根，拒绝负值。                                |
| `exp(tensor)`                    | 每个元素的指数。                                        |
| `log(tensor)`                    | 每个元素的自然对数。                                    |
| `try_log(tensor)`                | 可失败自然对数，拒绝非正值。                            |
| `is_close(a, b, rtol, atol)`     | 使用数值 tolerances 比较两个 scalars。                  |
| `all_close(a, b, rtol, atol)`    | 使用数值 tolerances 逐元素比较两个 tensors。            |
| `euclidean_distance(a, b)`       | 相同 shape 张量的 Euclidean distance。                  |
| `try_euclidean_distance(a, b)`   | 可失败 Euclidean distance。                             |
| `manhattan_distance(a, b)`       | 相同 shape 张量的 Manhattan distance。                  |
| `cosine_similarity(a, b)`        | 相同 shape 张量的 Cosine similarity。                   |
| `dot_similarity(a, b)`           | 相同 shape 张量的 Dot similarity。                      |
| `zscore(tensor)`                 | 对所有元素做 Z-score 标准化。                           |
| `standardize(tensor)`            | `zscore` 的别名。                                       |
| `minmax_scale(tensor, min, max)` | 将值缩放到目标区间。                                    |
| `clip_by_norm(tensor, max_norm)` | 将 L2 norm 裁剪到最大值。                               |

## 公共 companion modules

| 模块                 | 目的                                      |
|:---------------------|:-----------------------------------------|
| `viva_tensor/layout` | Canonical tensor layout metadata。        |
| `viva_tensor/axis`   | 语义轴名称和 axis specifications。        |
| `viva_tensor/named`  | 带 named axes 的 tensor wrapper。         |

## 稳定性策略

公共模块由 `gleam docs build` 生成文档，并应避免 panic；对可恢复错误优先使用 `Result`；
保持 semantic-versioning 兼容性；并在可能时保留可移植 fallback。随着 native acceleration、
quantization、sparse 和 neural-network APIs 继续成熟，内部模块可能会变化。详细策略见
[稳定性策略](stability.md)。
