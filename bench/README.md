# bench/

External benchmark and comparison tools. Anything that runs **outside**
the Gleam package API lives here — Python and R scripts for statistical
analysis, cross-runtime comparisons against PyTorch / NumPy, shell
runners, and the results / methodology markdowns.

Gleam-side benchmark entrypoints (the ones invoked via
`gleam run -m viva_tensor/bench/...`) live under
[`dev/viva_tensor/bench/`](../dev/viva_tensor/bench/) instead — keeping
them out of `src/` so they don't ship in the Hex package.

## Layout

| Path        | Purpose                                                                      |
| :---------- | :--------------------------------------------------------------------------- |
| `compare/`  | Current head-to-head: viva_tensor vs PyTorch / NumPy. Methodology + results. |
| `perf/`     | NumPy comparison scripts + saved CSV baselines.                              |
| `python/`   | Statistical benchmark scripts (bootstrap CI, throughput sweeps).             |
| `r/`        | R-based statistical analysis + plots.                                        |
| `scripts/`  | Shell runners for CI / batch execution.                                      |

## Quick start

```bash
# Head-to-head vs PyTorch / NumPy (matches viva_tensor/bench/showdown.gleam)
.bench-venv/bin/python3 bench/compare/numpy_pytorch.py

# Full statistical sweep with bootstrap CI
python3 bench/python/benchmark.py

# Drive everything in one go (CI)
./bench/scripts/run_benchmarks.sh
```

The `compare/` directory carries the markdowns that document what the
numbers mean:

- [`compare/RESULTS.md`](compare/RESULTS.md) — measured benchmarks
  across all backends and shapes.
- [`compare/INFERENCE_API_PLAN.md`](compare/INFERENCE_API_PLAN.md) —
  status of the `prepack_*` / `linear_*` NIFs.
- [`compare/CUTLASS_DSL_NOTES.md`](compare/CUTLASS_DSL_NOTES.md) —
  design notes for a future migration to CUTLASS 4 CuTeDSL.
- [`compare/NVFP4_EVT_PLAN.md`](compare/NVFP4_EVT_PLAN.md) —
  design notes for NVFP4 fused dequant + GEMM.

## Removed in 2.2.101

- `bench/erlang/` — 22 escript benchmarks superseded by the
  `dev/viva_tensor/bench/*.gleam` modern generation.
- `bench/cuda/test_int8_imma.cu` — covered by `peak.gleam`.
- `bench/windows/` — `.bat` runners (project doesn't currently support
  Windows as a first-class target; reopen if there's demand).
