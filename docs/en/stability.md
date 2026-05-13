# Stability Policy

`viva_tensor` treats the root `viva_tensor` module as the stable user surface.
New user-facing examples should start with:

```gleam
import viva_tensor as t
```

This keeps the package usable as a library even while native kernels,
quantization, sparse formats, and neural-network helpers keep evolving.

## Stable Surface

The stable public surface is:

| Module               | Status | Purpose                                      |
|:---------------------|:------:|:---------------------------------------------|
| `viva_tensor`        | Stable | Tensor creation, math, reductions, layout inspection, broadcasting, backend planning, and safe fallback execution. |
| `viva_tensor/layout` | Stable | Canonical tensor layout metadata.            |
| `viva_tensor/axis`   | Stable | Semantic axis names and axis specifications. |
| `viva_tensor/named`  | Stable | Tensor wrapper with named axes.              |

Stable functions should preserve semantic-versioning compatibility, return
`Result` for recoverable failures, and keep a pure BEAM fallback unless the
function is explicitly documented as native-only.

## Experimental Surface

The following areas are intentionally experimental until their contracts are
documented and covered by compatibility tests:

- `viva_tensor/core/*`
- `viva_tensor/backend/*`
- direct CUDA, BLAS, sparse, quantization, neural-network, optimization,
  telemetry, and benchmark modules
- development-only examples and benchmark entrypoints under `dev/`
- native NIF entrypoints that expose backend-specific resource details

Experimental modules may change shape as the implementation matures. Prefer the
root module unless you are working on internals or benchmarking a specific
backend.

## Public API Guardrails

Each stable API addition should include:

- a doc comment that states what the function does
- argument and shape constraints when relevant
- return and error behavior
- backend selection or fallback behavior when native acceleration is involved
- at least one public API contract test when the function belongs to the root
  module

The `test/public_api_contract_test.gleam` suite is the compatibility tripwire for
the root facade. It verifies creation, layout metadata, broadcasting, softmax,
linear algebra, and backend planning through `import viva_tensor as t`.

Broadcasting follows the mature tensor-library convention used by NumPy and
PyTorch: shapes are right-aligned, dimensions match when equal or when one side
is `1`, and expanded tensors are represented as views with zero strides where
possible.

## Backend Maturity

Pure Gleam execution is the portable baseline. Zig SIMD, MKL, CUDA FP32, CUDA
FP16, CUDA INT8, sparse, FP8, and fused kernels are exposed through capability
records and the backend planner first. Direct low-level backend modules should
not be treated as stable until operation contracts, dtype support, shape
constraints, error behavior, and fallback rules are documented.
