# bench/results/ — measurement outputs

Single landing place for everything benchmark scripts write out (CSVs,
Erlang terms, markdown reports). Keep generated artifacts here and the
scripts themselves under `python/`.

## Files

| File                          | Producer                                                 | Notes                                              |
| :---------------------------- | :------------------------------------------------------- | :------------------------------------------------- |
| `matmul_showdown.md`          | hand-written                                             | viva vs PyTorch vs NumPy across shapes / dtypes.   |
| `pytorch_results.term`        | `python/compare/numpy_pytorch.py`                        | Consumed by `dev/viva_tensor/bench/showdown.gleam`.|
| `numpy_cpu_baseline.csv`      | `python/compare/numpy_cpu.py`                            | Median per-op timings on CPU.                      |
| `viva_tensor_results.csv`     | `dev/viva_tensor/bench/legacy/perf.gleam`                | viva-side counterpart for `side_by_side.py`.       |
| `archive/`                    | —                                                        | Historical baselines + legacy code preserved.      |

`pytorch_results.term`, `*_baseline.csv`, and `viva_tensor_results.csv`
are **regenerated** every time the producing script runs and are safe
to delete. `matmul_showdown.md` is curated.

## Reading the numbers

`matmul_showdown.md` is the canonical place to look. It includes
hardware, software stack, dtype context, and call out the apples-to-apples
caveats. Raw CSVs / `.term` files are inputs to that summary.

## Archive

`archive/` keeps:
- Snapshotted baselines from previous releases (useful for regression
  comparison when a major refactor lands).
- Legacy scripts (e.g. `benchmark_compare_legacy.py`) that have been
  superseded but are kept around in case someone needs the old
  methodology for cross-reference.
