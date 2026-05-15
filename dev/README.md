# dev/

Gleam modules used while developing `viva_tensor`, not packaged into Hex.

These live outside `src/` on purpose — Hex consumers don't see them in
the stable library surface, and `viva_tensor.gleam` (the public facade)
doesn't import anything from here.

Public examples should always go through the root facade:

```gleam
import viva_tensor as t
```

## Layout

| Path                                                       | Purpose                                                              |
| :--------------------------------------------------------- | :------------------------------------------------------------------- |
| [`viva_tensor/bench/`](viva_tensor/bench/)                 | Benchmark entrypoints reachable via `gleam run -m viva_tensor/bench/...`. |
| [`viva_tensor/bench/legacy/`](viva_tensor/bench/legacy/)   | Older benchmark modules superseded by the current set — kept for reproducibility but not wired into Makefile targets. |
| [`viva_tensor/examples/`](viva_tensor/examples/)           | Runnable examples and demos for maintainers.                          |
| `viva_tensor/benchmark.gleam`                              | Shared benchmark helpers used by the modules above.                   |

## Current benchmark set (`viva_tensor/bench/`)

The eight modules that are actively maintained:

| Module       | Purpose                                                                                          |
| :----------- | :----------------------------------------------------------------------------------------------- |
| `peak`       | Tour every accelerated backend (FP16 / FP8 / sparse) at fixed shapes; the canonical TFLOPS dump. |
| `autotune`   | Sweep CUTLASS configs + cuSPARSELt modes per shape; pick the winner.                             |
| `cache`      | Persist autotune winners to `priv/autotune_cache.term` for later runs.                           |
| `showdown`   | Head-to-head viva_tensor vs PyTorch / NumPy using `bench/compare/pytorch_results.term`.          |
| `graph`      | CUDA Graphs vs loop launches — quantifies per-kernel driver overhead.                            |
| `full`       | Quantisation quality benchmark (INT8 / NF4 / AWQ) for `make bench`.                              |
| `rtx`        | RTX 4090 vs MKL comparison for `make bench-rtx`.                                                 |
| `regression` | Small stable-API regression benchmark for `make bench-regression`.                               |

Run any of them with:

```bash
gleam run -m viva_tensor/bench/<module>
```

## Legacy benchmark set (`viva_tensor/bench/legacy/`)

Older modules that overlap with the current set; kept because their CSV
output formats or specific kernels are referenced by older blog posts /
issues. Not wired into Makefile targets:

- `concurrent`, `gflops`, `gpu`, `nif`, `perf`, `tflops`.

If you find yourself reaching for one of these, check the current set
first — the same numbers are usually available in `peak.gleam` or
`showdown.gleam` with cleaner output.
