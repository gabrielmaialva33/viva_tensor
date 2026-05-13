# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

viva_tensor is a pure Gleam tensor library targeting the Erlang runtime (BEAM). It provides NumPy-inspired tensor
operations with memory optimization through quantization (INT8, NF4, AWQ). The tagline "Memory × 8" reflects NF4's
ability to compress 24GB VRAM to effectively handle 192GB of data.

## Build Commands

```bash
make build          # Compile the project
make test           # Run all tests (~50 tests)
make fmt            # Format code
make check          # Type check without compiling
make bench          # Run full benchmarks → output/benchmark_*.txt
make docs           # Generate HTML docs → build/docs/
```

For specific benchmarks: `make bench-int8`, `make bench-nf4`, `make bench-awq`, `make bench-flash`, `make bench-sparse`

Direct gleam commands also work: `gleam build`, `gleam test`, `gleam format src test`

## Architecture

```
src/viva_tensor/
├── core/           # Foundational tensor operations
│   ├── tensor.gleam   # Opaque tensor type with smart constructors
│   ├── ops.gleam      # Mathematical operations (add, mul, matmul, etc.)
│   ├── shape.gleam    # Shape manipulation (reshape, slice, concat)
│   ├── error.gleam    # Centralized error types
│   ├── dtype.gleam    # Phantom types for dtype safety
│   ├── ffi.gleam      # Erlang FFI bindings
│   └── config.gleam   # Builder patterns for configurations
├── quant/          # Quantization algorithms
│   ├── compression.gleam  # INT8 (4x compression)
│   ├── nf4.gleam          # NormalFloat4/QLoRA (7.5x compression)
│   └── awq.gleam          # Activation-Aware (7.7x compression)
├── nn/             # Neural network components
│   ├── autograd.gleam     # Automatic differentiation engine
│   ├── layers.gleam       # NN layers (Linear, etc.)
│   └── flash_attention.gleam  # Flash Attention v2
├── optim/          # Optimization & backends
│   ├── blackwell.gleam    # NVFP4-inspired compression
│   ├── rtx4090.gleam      # RTX 4090 optimized engine
│   ├── sparsity.gleam     # 2:4 Sparsity patterns
│   └── ...
└── viva_tensor.gleam   # Public API re-exports
```

### Design Patterns

**Opaque Types with Smart Constructors**: The core `Tensor` type is opaque to ensure valid states:

```gleam
// In core/tensor.gleam
pub opaque type Tensor {
  Dense(data: List(Float), shape: List(Int))
  Strided(storage: ErlangArray, shape: List(Int), strides: List(Int), offset: Int)
}

// Smart constructor validates invariants
pub fn new(data: List(Float), shape: List(Int)) -> Result(Tensor, TensorError)
```

**Builder Pattern with Labelled Arguments**: Configuration objects use fluent builders:

```gleam
// In core/config.gleam
let config = conv2d()
  |> with_stride(2)
  |> with_padding(1)
  |> with_dilation(1)
```

**Centralized Error Types**: All errors flow through `core/error.gleam`:

```gleam
pub type TensorError {
  ShapeMismatch(expected: List(Int), got: List(Int))
  InvalidShape(reason: String)
  DimensionError(reason: String)
  BroadcastError(shape_a: List(Int), shape_b: List(Int))
  IndexOutOfBounds(index: Int, size: Int)
  DtypeError(reason: String)
}
```

**Phantom Types for Dtype Safety**: Compile-time dtype verification:

```gleam
// In core/dtype.gleam
pub type Float32
pub type Float16
pub type Int8
pub type NF4
```

### Module Responsibilities

**core/tensor.gleam**: Tensor construction and accessors

- `new`, `zeros`, `ones`, `fill` - constructors
- `shape`, `size`, `rank`, `to_list` - accessors
- `get`, `get2d`, `get_row`, `get_col` - indexing
- Strided tensor support for zero-copy ops

**core/ops.gleam**: Mathematical operations

- Element-wise: `add`, `sub`, `mul`, `div`, `negate`
- Reductions: `sum`, `mean`, `max`, `min`, `variance`
- Matrix: `matmul`, `transpose`, `dot`, `outer`
- Activations: `relu`, `sigmoid`, `tanh`, `softmax`
- Broadcasting: `add_broadcast`, `mul_broadcast`, `broadcast_to`

**core/shape.gleam**: Shape manipulation

- `reshape`, `flatten`, `squeeze`, `unsqueeze`
- `slice`, `take_first`, `take_last`
- `concat`, `concat_axis`, `stack`

**nn/autograd.gleam**: Automatic differentiation

- `Tape` - computation graph
- `Variable` - tracked tensor
- `Traced(a)` - state monad for traced operations
- `backward` - backpropagation engine

### Performance Patterns

- Tail-recursive implementations throughout to prevent stack growth
- Direct index computation in conv2d/pooling (no intermediate list allocations)
- Strided views for zero-copy transpose and reshape operations
- Erlang `:array` for O(1) element access in hot paths

## Testing

Tests are organized by domain:

- [test/viva_tensor_test.gleam](test/viva_tensor_test.gleam) - Core tensor tests
- [test/autograd_test.gleam](test/autograd_test.gleam) - Autograd tests

Test categories:

- Tensor constructors (zeros, ones, from_list, random)
- Element-wise operations (add, sub, mul, div with broadcasting)
- Reductions (sum, mean, max, min, variance)
- Matrix operations (matmul, transpose, outer)
- Shape operations (reshape, flatten, squeeze, slice)
- Conv2D and pooling operations
- Autograd backward pass

## Adding New Operations

1. Add function to appropriate module:

- Math operations → `core/ops.gleam`
- Shape operations → `core/shape.gleam`
- NN operations → `nn/` directory

2. Return `Result(Tensor, TensorError)` for fallible operations
3. Add public re-export in [viva_tensor.gleam](src/viva_tensor.gleam) if needed
4. Add tests in appropriate test file
5. Run `make fmt && make test`

## Dependencies

- Runtime: `gleam_stdlib` only (pure Gleam, no native extensions)
- Dev: `gleeunit` (testing), `gleamy_bench` (benchmarking)
- Requires: Gleam 1.14.0+, OTP 27+
