# bench/python/compare/ — cross-runtime comparisons

Head-to-head benchmarks comparing `viva_tensor` against external
runtimes (NumPy on CPU, PyTorch on CPU+GPU). The goal is **not** to win —
it is honest measurement, so optimization work can point at a concrete
delta.

> Realistic expectation: pure-Gleam `viva_tensor` is **10×–1000×** slower
> than NumPy on CPU-bound ops without the NIF. That gap is exactly what
> the native backends (Zig SIMD, MKL, CUDA, CUTLASS) exist to close.

## Scripts

| Script              | Purpose                                                                    |
| :------------------ | :------------------------------------------------------------------------- |
| `numpy_pytorch.py`  | vs PyTorch + NumPy across matmul shapes. Writes `results/pytorch_results.term`, consumed by `dev/viva_tensor/bench/showdown.gleam`. |
| `numpy_cpu.py`      | NumPy CPU baseline. Writes `results/numpy_cpu_baseline.csv`.               |
| `side_by_side.py`   | Reads both CSVs (`numpy_cpu_baseline.csv` + `viva_tensor_results.csv`) and prints a single comparison table. |

The viva_tensor side of the CSV pair (`viva_tensor_results.csv`) is
produced by `dev/viva_tensor/bench/legacy/perf.gleam` (legacy, may move).

## Run

```bash
# PyTorch + NumPy reference for the matmul showdown
.bench-venv/bin/python3 bench/python/compare/numpy_pytorch.py
#   -> bench/results/pytorch_results.term
#   then: gleam run -m viva_tensor/bench/showdown

# CPU-only NumPy baseline
python3 bench/python/compare/numpy_cpu.py
#   -> bench/results/numpy_cpu_baseline.csv
#   then: gleam run -m viva_tensor/bench/perf  (produces viva CSV)

# Side-by-side table
python3 bench/python/compare/side_by_side.py
```

The numpy-only path needs only stock NumPy. `numpy_pytorch.py` also
needs `torch`.

## Ops & shapes (numpy_cpu.py)

| Op           | Shape(s)                                                 |
| :----------- | :------------------------------------------------------- |
| `matmul`     | `(1024,1024) @ (1024,1024)`, `(2048,2048) @ (2048,2048)` |
| `transpose`  | `(2048,2048)`                                            |
| `add`        | `(1024,1024)`, `(2048,2048)` (element-wise)              |
| `mul`        | `(1024,1024)`, `(2048,2048)` (element-wise)              |
| `sum`        | full reduction over `(2048,2048)`                        |
| `softmax`    | `(512,1024)` per-row                                     |
| `layer_norm` | `(512,1024)` along last axis                             |

## Iteration policy

* **NumPy** — 5 warmup + 20 timed iterations per op. Median reported.
* **viva_tensor (legacy perf.gleam)** — 5 warmup + 20 timed for cheap
  ops; 1 warmup + 3 timed for heavy paths so the run stays under a few
  minutes. Median reported.

Both sides use outer-loop wall-clock timing (`time.perf_counter` and
`erlang:monotonic_time`). No statistical magic — just medians.

## CSV columns

```
op,size,iters,total_ms,per_op_us,gflops_estimate
```

`gflops_estimate` is an order-of-magnitude sanity check, not a
peer-reviewed roofline. matmul uses `2·n³`; element-wise counts `n²`;
softmax / layer_norm assume ~5 ops / element.

## Notes

* `numpy_pytorch.py` does **kernel-only timing** via CUDA events on the
  GPU side and warm-up exclusion. Comparable to
  `viva_tensor/bench/peak.gleam`.
* `viva_tensor`'s `matmul` in `numpy_cpu.py` goes through the pure-Gleam
  path (`t.matmul`). Native backends (Apple Accelerate, Zig SIMD, CUDA)
  live behind `*_accelerated` variants and get their own bench.
* The harness is **not** part of the Hex package — `bench/` is outside
  `src/`.
