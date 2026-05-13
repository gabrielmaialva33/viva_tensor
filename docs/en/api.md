# API Guide

`viva_tensor` exposes a small stable surface from the root `viva_tensor` module.
Implementation modules for native backends, quantization, sparse kernels,
telemetry, benchmarking, and experimental neural-network helpers are kept
internal until their contracts are stable.
Development-only benchmark and example entrypoints live under `dev/` so they can
be run locally without becoming part of the packaged library surface.

See [Stability Policy](stability.md) for the stable/experimental boundary and
the compatibility expectations for root-module additions. See
[Project Structure](project-structure.md) for the package layout and module
boundary rules.

## Stable Imports

```gleam
import gleam/result
import viva_tensor as t
```

Use the root module for normal tensor work. Import `viva_tensor/layout` when
you need to inspect storage metadata, and `viva_tensor/axis` or
`viva_tensor/named` when working with semantic dimensions.

## Tensor Creation

| Function                   | Description                                      |
|:---------------------------|:-------------------------------------------------|
| `zeros(shape)`             | Create a tensor filled with zeros.               |
| `ones(shape)`              | Create a tensor filled with ones.                |
| `fill(shape, value)`       | Create a tensor filled with one scalar value.    |
| `from_list(data)`          | Create a one-dimensional tensor.                 |
| `from_list2d(rows)`        | Create a matrix from rows, validating row sizes. |
| `linspace(start, stop, steps)` | Create evenly spaced values over a closed interval. |
| `try_linspace(start, stop, steps)` | Fallible linspace rejecting invalid step counts. |
| `logspace(start, stop, steps, base)` | Create logarithmically spaced values.       |
| `try_logspace(start, stop, steps, base)` | Fallible logspace rejecting invalid steps/base. |
| `zeros_like(tensor)`       | Create a zero tensor with the same shape.        |
| `ones_like(tensor)`        | Create a one tensor with the same shape.         |
| `full_like(tensor, value)` | Create a filled tensor with the same shape.      |
| `eye(n)` / `identity(n)`   | Create a square identity matrix.                 |
| `try_eye(n)`               | Fallible identity matrix rejecting invalid size. |
| `diag(tensor)`             | Create a diagonal matrix from a vector.          |
| `try_diag(tensor)`         | Fallible diagonal matrix creation.               |
| `matrix(rows, cols, data)` | Create a matrix with explicit dimensions.        |

```gleam
let a = t.zeros([2, 3])
let b = t.fill([2, 3], 1.5)
```

## Fallible Operations

Shape-changing and binary tensor operations return `Result` rather than
panicking. Chain them with `gleam/result.try`. Use `try_to_list()` instead of
`to_list()` inside fallible code when native materialization failures need to be
preserved.

```gleam
pub fn example() {
  let a = t.ones([2, 3])
  let b = t.fill([2, 3], 2.0)

  use c <- result.try(t.add(a, b))
  use flat <- result.try(t.reshape(c, [6]))

  Ok(t.mean(flat))
}
```

## Element-wise Math

| Function                | Description                                   |
|:------------------------|:----------------------------------------------|
| `add(a, b)`             | Element-wise addition for equal shapes.       |
| `sub(a, b)`             | Element-wise subtraction for equal shapes.    |
| `mul(a, b)`             | Element-wise multiplication for equal shapes. |
| `div(a, b)`             | Element-wise division for equal shapes.       |
| `scale(tensor, scalar)` | Multiply every element by a scalar.           |
| `try_scale(tensor, scalar)` | Fallible scalar multiplication preserving native materialization errors. |
| `add_scalar(tensor, scalar)` | Add a scalar to every element.          |
| `try_add_scalar(tensor, scalar)` | Fallible scalar addition preserving native materialization errors. |
| `negate(tensor)`        | Negate every element.                         |
| `try_negate(tensor)`    | Fallible negation preserving native materialization errors. |
| `clamp(tensor, min, max)` | Clamp values into a closed interval.        |
| `try_clamp(tensor, min, max)` | Fallible clamp preserving native materialization errors. |
| `clip(tensor, min, max)` | Alias for clamping values into a closed interval. |
| `try_clip(tensor, min, max)` | Fallible clip rejecting invalid intervals. |
| `abs(tensor)` / `try_abs(tensor)` | Absolute value for every element.     |
| `square(tensor)` / `try_square(tensor)` | Square every element.              |
| `sqrt(tensor)` / `try_sqrt(tensor)` | Square root, rejecting negative values in `try_sqrt`. |
| `exp(tensor)` / `try_exp(tensor)` | Exponential for every element.        |
| `log(tensor)` / `try_log(tensor)` | Natural logarithm, rejecting non-positive values in `try_log`. |
| `floor(tensor)` / `try_floor(tensor)` | Floor every element.               |
| `ceil(tensor)` / `try_ceil(tensor)` | Ceiling every element.             |
| `round(tensor)` / `try_round(tensor)` | Round every element to nearest integer value. |
| `sign(tensor)` / `try_sign(tensor)` | Return -1, 0, or 1 for each element. |
| `reciprocal(tensor)` / `try_reciprocal(tensor)` | Reciprocal, rejecting zero values in `try_reciprocal`. |
| `map(tensor, fun)`      | Apply a scalar function to every element.     |
| `try_map(tensor, fun)`  | Fallible scalar mapping preserving native materialization errors. |
| `softmax_axis(tensor, axis)` | Normalize each slice along an axis.     |
| `try_softmax_axis(tensor, axis)` | Fallible softmax preserving native materialization and indexing errors. |

Use broadcasting-specific functions when shapes differ.

| Function                      | Description                               |
|:------------------------------|:------------------------------------------|
| `can_broadcast(a, b)`         | Check whether two shapes are compatible.  |
| `broadcast_shape(a, b)`       | Compute the common shape for two shapes.  |
| `broadcast_shapes(shapes)`    | Compute the common shape for many shapes. |
| `broadcast_to(tensor, shape)` | Create a broadcast view when possible.    |
| `broadcast_pair(a, b)`        | Broadcast two tensors to common views.    |
| `add_broadcast(a, b)`         | Add with NumPy-style broadcasting.        |
| `sub_broadcast(a, b)`         | Subtract with NumPy-style broadcasting.   |
| `mul_broadcast(a, b)`         | Multiply with NumPy-style broadcasting.   |
| `div_broadcast(a, b)`         | Divide with NumPy-style broadcasting.     |
| `maximum(a, b)`               | Element-wise maximum with broadcasting.   |
| `minimum(a, b)`               | Element-wise minimum with broadcasting.   |

## Reductions

| Function                         | Description                                            |
|:---------------------------------|:-------------------------------------------------------|
| `sum(tensor)`                    | Sum all elements.                                      |
| `try_sum(tensor)`                | Fallible sum preserving native materialization errors. |
| `sum_axis(tensor, axis)`         | Sum along one axis.                                    |
| `try_sum_axis(tensor, axis)`     | Fallible sum along one axis.                           |
| `sum_axis_keepdims(tensor, axis)`| Sum along one axis while keeping a size-1 dimension.   |
| `mean(tensor)`                   | Mean over all elements.                                |
| `try_mean(tensor)`               | Fallible mean preserving materialization and empty-tensor errors. |
| `product(tensor)`                | Product over all elements.                             |
| `try_product(tensor)`            | Fallible product preserving materialization errors.     |
| `cumsum(tensor)`                 | Cumulative sum over flattened values, preserving shape. |
| `try_cumsum(tensor)`             | Fallible cumulative sum preserving materialization errors. |
| `cumsum_axis(tensor, axis)`      | Cumulative sum along one axis, preserving shape.    |
| `try_cumsum_axis(tensor, axis)`  | Fallible cumulative sum along one axis.             |
| `cumprod(tensor)`                | Cumulative product over flattened values, preserving shape. |
| `try_cumprod(tensor)`            | Fallible cumulative product preserving materialization errors. |
| `cumprod_axis(tensor, axis)`     | Cumulative product along one axis, preserving shape. |
| `try_cumprod_axis(tensor, axis)` | Fallible cumulative product along one axis.         |
| `median(tensor)`                 | Median over all elements.                              |
| `try_median(tensor)`             | Fallible median preserving materialization and empty-tensor errors. |
| `percentile(tensor, percentile)` | Percentile using linear interpolation.                 |
| `try_percentile(tensor, percentile)` | Fallible percentile with explicit bounds and empty-tensor errors. |
| `mean_axis(tensor, axis)`        | Mean along one axis.                                   |
| `try_mean_axis(tensor, axis)`    | Fallible mean along one axis.                          |
| `mean_axis_keepdims(tensor, axis)`| Mean along one axis while keeping a size-1 dimension. |
| `variance_axis(tensor, axis)`    | Variance along one axis.                               |
| `try_variance_axis(tensor, axis)`| Fallible variance along one axis.                      |
| `variance_axis_keepdims(tensor, axis)`| Variance along one axis while keeping a size-1 dimension. |
| `std_axis(tensor, axis)`         | Standard deviation along one axis.                     |
| `try_std_axis(tensor, axis)`     | Fallible standard deviation along one axis.            |
| `std_axis_keepdims(tensor, axis)`| Standard deviation along one axis while keeping a size-1 dimension. |
| `max_axis(tensor, axis)`         | Maximum along one axis.                                |
| `try_max_axis(tensor, axis)`     | Fallible maximum along one axis.                       |
| `max_axis_keepdims(tensor, axis)`| Maximum along one axis while keeping a size-1 dimension. |
| `min_axis(tensor, axis)`         | Minimum along one axis.                                |
| `try_min_axis(tensor, axis)`     | Fallible minimum along one axis.                       |
| `min_axis_keepdims(tensor, axis)`| Minimum along one axis while keeping a size-1 dimension. |
| `argmax_axis(tensor, axis)`      | Argmax index along one axis, represented as floats.    |
| `try_argmax_axis(tensor, axis)`  | Fallible argmax index along one axis.                  |
| `argmin_axis(tensor, axis)`      | Argmin index along one axis, represented as floats.    |
| `try_argmin_axis(tensor, axis)`  | Fallible argmin index along one axis.                  |
| `max(tensor)`                    | Maximum value.                                         |
| `try_max(tensor)`                | Fallible maximum preserving materialization and empty-tensor errors. |
| `min(tensor)`                    | Minimum value.                                         |
| `try_min(tensor)`                | Fallible minimum preserving materialization and empty-tensor errors. |
| `argmax(tensor)`                 | Flat index of the maximum value.                       |
| `try_argmax(tensor)`             | Fallible flat index of the maximum value.              |
| `argmin(tensor)`                 | Flat index of the minimum value.                       |
| `try_argmin(tensor)`             | Fallible flat index of the minimum value.              |
| `variance(tensor)`               | Variance over all elements.                            |
| `try_variance(tensor)`           | Fallible variance preserving materialization and empty-tensor errors. |
| `std(tensor)`                    | Standard deviation over all elements.                  |
| `try_std(tensor)`                | Fallible standard deviation preserving materialization and empty-tensor errors. |

## Linear Algebra

| Function                     | Description                   |
|:-----------------------------|:------------------------------|
| `dot(a, b)`                  | Dot product for vectors.      |
| `matmul(a, b)`               | Matrix multiplication.        |
| `matmul_planned(a, b)`       | Matrix multiplication using the stable backend planner with fallback. |
| `matmul_vec(matrix, vector)` | Matrix-vector multiplication. |
| `transpose(tensor)`          | Matrix transpose.             |
| `outer(a, b)`                | Outer product.                |

Native-backed variants such as `matmul_into`, `to_accelerated`, and
`matmul_accelerated_into` are available from the root module for hot paths that
can reuse buffers or persistent GPU memory.

Use `capabilities()` to inspect whether the current VM loaded the native NIF,
the Zig SIMD backend, which TFLOPS backends are visible, and the stable backend
capability records. Use `backend_capabilities()` when you only need the
capability table, or `plan_backend(operation)` to see which backend the stable
planner would choose for an operation. Plans include `rejected` backend entries with human-readable reasons for unavailable or unsuitable backends.

```gleam
let plan = t.plan_backend(t.OperationMatmul(m: 1024, n: 1024, k: 1024))
```

## Shape And Layout

| Function                  | Description                                                      |
|:--------------------------|:-----------------------------------------------------------------|
| `shape(tensor)`           | Tensor dimensions.                                               |
| `size(tensor)`            | Total element count.                                             |
| `rank(tensor)`            | Number of dimensions.                                            |
| `reshape(tensor, shape)`  | Change shape while preserving element count.                     |
| `device(tensor)`          | Payload device class.                                            |
| `dtype(tensor)`           | Tensor element type.                                             |
| `try_to_list(tensor)`     | Materialize tensor data while preserving native failures.        |
| `flatten(tensor)`         | Convert to one dimension.                                        |
| `try_flatten(tensor)`     | Fallible flatten preserving materialization failures.            |
| `squeeze(tensor)`         | Remove size-one dimensions.                                      |
| `unsqueeze(tensor, axis)` | Insert a size-one dimension.                                     |
| `try_unsqueeze(tensor, axis)` | Fallible unsqueeze preserving invalid-axis errors.            |
| `to_strided(tensor)`      | Convert dense data to a zero-copy strided view.                  |
| `try_to_strided(tensor)`  | Fallible strided conversion preserving native materialization errors. |
| `to_contiguous(tensor)`   | Materialize a strided view into contiguous dense storage.         |
| `try_to_contiguous(tensor)` | Fallible contiguous conversion preserving materialization errors. |
| `layout(tensor)`          | Inspect storage, device, dtype, strides, offset, and contiguity. |

```gleam
let info = t.layout(t.zeros([2, 3]))
```

Broadcasting, squeeze, unsqueeze, and contiguous reshape preserve strided views
where possible. Call `to_contiguous()` before a heavy native hot path if a view
would be slower than a dense buffer.

## Utilities

| Function                | Description                                         |
|:------------------------|:----------------------------------------------------|
| `norm(tensor)`          | L2 norm.                                            |
| `try_norm(tensor)`      | Fallible L2 norm preserving materialization errors. |
| `normalize(tensor)`     | Normalize to unit length.                           |
| `try_normalize(tensor)` | Fallible normalization preserving materialization errors. |
| `abs(tensor)`           | Absolute value for every element.                   |
| `square(tensor)`        | Square every element.                               |
| `sqrt(tensor)`          | Square root every element.                          |
| `try_sqrt(tensor)`      | Fallible square root rejecting negative values.     |
| `exp(tensor)`           | Exponential for every element.                      |
| `log(tensor)`           | Natural logarithm for every element.                |
| `try_log(tensor)`       | Fallible natural logarithm rejecting non-positive values. |
| `is_close(a, b, rtol, atol)` | Compare two scalars with numeric tolerances. |
| `all_close(a, b, rtol, atol)` | Compare two tensors element-wise with numeric tolerances. |
| `euclidean_distance(a, b)` | Euclidean distance for same-shaped tensors.     |
| `try_euclidean_distance(a, b)` | Fallible Euclidean distance.                |
| `manhattan_distance(a, b)` | Manhattan distance for same-shaped tensors.     |
| `cosine_similarity(a, b)` | Cosine similarity for same-shaped tensors.      |
| `dot_similarity(a, b)`   | Dot similarity for same-shaped tensors.         |
| `zscore(tensor)`         | Z-score standardization over all elements.      |
| `standardize(tensor)`    | Alias for `zscore`.                             |
| `minmax_scale(tensor, min, max)` | Scale values into a target interval.    |
| `clip_by_norm(tensor, max_norm)` | Clip L2 norm to a maximum value.        |

## Public Companion Modules

| Module               | Purpose                                      |
|:---------------------|:---------------------------------------------|
| `viva_tensor/layout` | Canonical tensor layout metadata.            |
| `viva_tensor/axis`   | Semantic axis names and axis specifications. |
| `viva_tensor/named`  | Tensor wrapper with named axes.              |

## Stability Policy

Public modules are documented by `gleam docs build` and should avoid panics,
prefer `Result` for recoverable errors, preserve semantic-versioning
compatibility, and keep a portable fallback when possible. Internal modules may
change while the native acceleration, quantization, sparse, and neural-network
APIs continue to mature. The detailed policy lives in
[Stability Policy](stability.md).
