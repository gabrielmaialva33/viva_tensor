# API Guide

`viva_tensor` exposes a small stable surface from the root `viva_tensor` module.
Implementation modules for native backends, quantization, sparse kernels,
telemetry, benchmarking, and experimental neural-network helpers are kept
internal until their contracts are stable.
Development-only benchmark and example entrypoints live under `dev/` so they can
be run locally without becoming part of the packaged library surface.

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
| `matrix(rows, cols, data)` | Create a matrix with explicit dimensions.        |

```gleam
let a = t.zeros([2, 3])
let b = t.fill([2, 3], 1.5)
```

## Fallible Operations

Shape-changing and binary tensor operations return `Result` rather than
panicking. Chain them with `gleam/result.try`.

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
| `map(tensor, fun)`      | Apply a scalar function to every element.     |
| `softmax_axis(tensor, axis)` | Normalize each slice along an axis.     |

Use broadcasting-specific functions when shapes differ.

| Function                      | Description                              |
|:------------------------------|:-----------------------------------------|
| `can_broadcast(a, b)`         | Check whether two shapes are compatible. |
| `broadcast_to(tensor, shape)` | Create a broadcast view when possible.   |
| `add_broadcast(a, b)`         | Add with NumPy-style broadcasting.       |
| `sub_broadcast(a, b)`         | Subtract with NumPy-style broadcasting.  |
| `mul_broadcast(a, b)`         | Multiply with NumPy-style broadcasting.  |
| `div_broadcast(a, b)`         | Divide with NumPy-style broadcasting.    |

## Reductions

| Function           | Description                           |
|:-------------------|:--------------------------------------|
| `sum(tensor)`      | Sum all elements.                     |
| `mean(tensor)`     | Mean over all elements.               |
| `max(tensor)`      | Maximum value.                        |
| `min(tensor)`      | Minimum value.                        |
| `argmax(tensor)`   | Flat index of the maximum value.      |
| `argmin(tensor)`   | Flat index of the minimum value.      |
| `variance(tensor)` | Variance over all elements.           |
| `std(tensor)`      | Standard deviation over all elements. |

## Linear Algebra

| Function                     | Description                   |
|:-----------------------------|:------------------------------|
| `dot(a, b)`                  | Dot product for vectors.      |
| `matmul(a, b)`               | Matrix multiplication.        |
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
| `flatten(tensor)`         | Convert to one dimension.                                        |
| `squeeze(tensor)`         | Remove size-one dimensions.                                      |
| `unsqueeze(tensor, axis)` | Insert a size-one dimension.                                     |
| `layout(tensor)`          | Inspect storage, device, dtype, strides, offset, and contiguity. |

```gleam
let info = t.layout(t.zeros([2, 3]))
```

Broadcasting, squeeze, unsqueeze, and contiguous reshape preserve strided views
where possible. Call `to_contiguous()` before a heavy native hot path if a view
would be slower than a dense buffer.

## Public Companion Modules

| Module               | Purpose                                      |
|:---------------------|:---------------------------------------------|
| `viva_tensor/layout` | Canonical tensor layout metadata.            |
| `viva_tensor/axis`   | Semantic axis names and axis specifications. |
| `viva_tensor/named`  | Tensor wrapper with named axes.              |

## Stability Policy

Public modules are documented by `gleam docs build` and should avoid panics,
prefer `Result` for recoverable errors, and preserve semantic-versioning
compatibility. Internal modules may change while the native acceleration,
quantization, sparse, and neural-network APIs continue to mature.
