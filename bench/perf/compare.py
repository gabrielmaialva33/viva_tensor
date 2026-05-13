#!/usr/bin/env python3
"""Print a side-by-side comparison of NumPy vs viva_tensor CPU timings.

Reads ``bench/perf/numpy_results.csv`` and
``bench/perf/viva_tensor_results.csv`` (both produced by their respective
sibling scripts) and prints a single table:

    op           | size        | numpy_us | viva_us | viva / numpy
    matmul       | 1024x1024   |     6800 |  280000 | 41.18x slower

If a (op, size) row is missing from one side it shows ``-``.

Run:
    python3 bench/perf/compare.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NUMPY_CSV = ROOT / "numpy_results.csv"
VIVA_CSV = ROOT / "viva_tensor_results.csv"


def _load(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        print(f"warning: missing {path}", file=sys.stderr)
        return {}
    out: dict[tuple[str, str], float] = {}
    with path.open() as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            try:
                out[(row["op"], row["size"])] = float(row["per_op_us"])
            except (KeyError, ValueError):
                continue
    return out


def _fmt_us(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:>9.0f}"


def _fmt_ratio(numpy_us: float | None, viva_us: float | None) -> str:
    if numpy_us is None or viva_us is None or numpy_us <= 0.0:
        return "-"
    ratio = viva_us / numpy_us
    if ratio >= 1.0:
        return f"{ratio:>7.2f}x slower"
    return f"{1.0 / ratio:>7.2f}x faster"


def main() -> int:
    numpy_rows = _load(NUMPY_CSV)
    viva_rows = _load(VIVA_CSV)

    # Union of keys, preserving a stable display order.
    op_order = [
        "matmul",
        "transpose",
        "add",
        "mul",
        "sum",
        "softmax",
        "layer_norm",
    ]
    keys = sorted(
        set(numpy_rows.keys()) | set(viva_rows.keys()),
        key=lambda k: (
            op_order.index(k[0]) if k[0] in op_order else len(op_order),
            k[1],
        ),
    )

    header = f"{'op':<12} | {'size':<11} | {'numpy_us':>9} | {'viva_us':>9} | viva / numpy"
    print(header)
    print("-" * len(header))

    if not keys:
        print("(no results — run compare_numpy.py and the Gleam bench first)")
        return 1

    for op, size in keys:
        numpy_us = numpy_rows.get((op, size))
        viva_us = viva_rows.get((op, size))
        print(
            f"{op:<12} | {size:<11} | {_fmt_us(numpy_us)} | "
            f"{_fmt_us(viva_us)} | {_fmt_ratio(numpy_us, viva_us)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
