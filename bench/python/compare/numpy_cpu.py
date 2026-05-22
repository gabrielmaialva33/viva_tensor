#!/usr/bin/env python3
"""Side-by-side CPU performance baseline: NumPy half of the comparison.

For each op below, times the NumPy implementation across a few sizes and
writes one CSV row per (op, size) under ``bench/results/numpy_cpu_baseline.csv``.

CSV columns: ``op,size,iters,total_ms,per_op_us,gflops_estimate``.

Each measurement uses 5 warmup iterations + 20 timed iterations and reports
the median per-op timing (more robust than the mean for short ops).

Run:
    python3 bench/python/compare/numpy_cpu.py

Requires only ``numpy`` (no torch, no scipy).
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

import numpy as np

WARMUP_ITERS = 5
TIMED_ITERS = 20

OUT_PATH = Path(__file__).resolve().parents[2] / "results" / "numpy_cpu_baseline.csv"


def _time_op(fn, warmup: int = WARMUP_ITERS, timed: int = TIMED_ITERS) -> list[float]:
    """Return per-iteration elapsed times in seconds (length == ``timed``)."""
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(timed):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return samples


def _record(
    rows: list[dict],
    op: str,
    size_label: str,
    samples: list[float],
    flops: int | None,
) -> None:
    median_s = statistics.median(samples)
    total_ms = sum(samples) * 1000.0
    per_op_us = median_s * 1_000_000.0
    if flops is not None and median_s > 0.0:
        gflops = flops / (median_s * 1e9)
        gflops_str = f"{gflops:.4f}"
    else:
        gflops_str = ""
    rows.append(
        {
            "op": op,
            "size": size_label,
            "iters": str(len(samples)),
            "total_ms": f"{total_ms:.4f}",
            "per_op_us": f"{per_op_us:.4f}",
            "gflops_estimate": gflops_str,
        }
    )
    print(f"  {op:<12} {size_label:<13} median={per_op_us:>12.2f} us  gflops={gflops_str or '-'}")


def bench_matmul(rows: list[dict]) -> None:
    rng = np.random.default_rng(0)
    for n in (1024, 2048):
        a = rng.standard_normal((n, n), dtype=np.float64)
        b = rng.standard_normal((n, n), dtype=np.float64)
        samples = _time_op(lambda a=a, b=b: a @ b)
        # matmul FLOPs ~= 2 * n^3
        _record(rows, "matmul", f"{n}x{n}", samples, flops=2 * n * n * n)


def bench_transpose(rows: list[dict]) -> None:
    rng = np.random.default_rng(1)
    n = 2048
    a = rng.standard_normal((n, n), dtype=np.float64)
    # Force materialization — NumPy's .T is a view; .copy() makes it honest.
    samples = _time_op(lambda a=a: np.ascontiguousarray(a.T))
    _record(rows, "transpose", f"{n}x{n}", samples, flops=None)


def bench_add(rows: list[dict]) -> None:
    rng = np.random.default_rng(2)
    for n in (1024, 2048):
        a = rng.standard_normal((n, n), dtype=np.float64)
        b = rng.standard_normal((n, n), dtype=np.float64)
        samples = _time_op(lambda a=a, b=b: a + b)
        _record(rows, "add", f"{n}x{n}", samples, flops=n * n)


def bench_mul(rows: list[dict]) -> None:
    rng = np.random.default_rng(3)
    for n in (1024, 2048):
        a = rng.standard_normal((n, n), dtype=np.float64)
        b = rng.standard_normal((n, n), dtype=np.float64)
        samples = _time_op(lambda a=a, b=b: a * b)
        _record(rows, "mul", f"{n}x{n}", samples, flops=n * n)


def bench_sum(rows: list[dict]) -> None:
    rng = np.random.default_rng(4)
    n = 2048
    a = rng.standard_normal((n, n), dtype=np.float64)
    samples = _time_op(lambda a=a: a.sum())
    _record(rows, "sum", f"{n}x{n}", samples, flops=n * n)


def bench_softmax(rows: list[dict]) -> None:
    rng = np.random.default_rng(5)
    a = rng.standard_normal((512, 1024), dtype=np.float64)

    def softmax_row(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=-1, keepdims=True)
        e = np.exp(x)
        return e / e.sum(axis=-1, keepdims=True)

    samples = _time_op(lambda a=a: softmax_row(a))
    # ~5 flops per element (sub, exp, sum, div) — rough order-of-magnitude.
    _record(rows, "softmax", "512x1024", samples, flops=5 * 512 * 1024)


def bench_layer_norm(rows: list[dict]) -> None:
    rng = np.random.default_rng(6)
    a = rng.standard_normal((512, 1024), dtype=np.float64)
    eps = 1e-5

    def layer_norm(x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    samples = _time_op(lambda a=a: layer_norm(a))
    # ~5 flops per element estimate.
    _record(rows, "layer_norm", "512x1024", samples, flops=5 * 512 * 1024)


def main() -> None:
    print(
        f"NumPy {np.__version__} CPU benchmarks (warmup={WARMUP_ITERS}, timed={TIMED_ITERS} per op)"
    )
    print("-" * 64)
    rows: list[dict] = []
    bench_matmul(rows)
    bench_transpose(rows)
    bench_add(rows)
    bench_mul(rows)
    bench_sum(rows)
    bench_softmax(rows)
    bench_layer_norm(rows)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "op",
                "size",
                "iters",
                "total_ms",
                "per_op_us",
                "gflops_estimate",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("-" * 64)
    print(f"wrote {len(rows)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    main()
