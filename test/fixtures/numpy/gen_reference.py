#!/usr/bin/env python3
"""Generate NumPy reference outputs for viva_tensor's reference test suite.

For each test case, this script runs the operation in NumPy and writes one JSON
fixture per case under ``test/fixtures/numpy/<op>/<case>.json``. The Gleam test
suite (``test/reference_test.gleam``) loads these fixtures and asserts that
viva_tensor's output is close to NumPy's via an ``np.allclose``-style check.

Run with:
    uv run python test/fixtures/numpy/gen_reference.py
    # or
    python3 test/fixtures/numpy/gen_reference.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Tolerance defaults. Tight on purpose so we catch numerical regressions early.
DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-9

FIXTURES_DIR = Path(__file__).resolve().parent


def _tensor_payload(arr: np.ndarray) -> dict:
    """Serialise a numpy array as ``{"shape": [...], "data": [...]}``.

    Scalars are represented as ``{"shape": [], "data": [value]}`` so the Gleam
    side has a single uniform shape to decode.
    """
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        return {"shape": [], "data": [float(arr)]}
    return {
        "shape": [int(d) for d in arr.shape],
        "data": [float(v) for v in arr.reshape(-1).tolist()],
    }


def _write_case(
    op: str,
    case: str,
    inputs: list[np.ndarray],
    output: np.ndarray,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Path:
    op_dir = FIXTURES_DIR / op
    op_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "op": op,
        "case": case,
        "inputs": [_tensor_payload(x) for x in inputs],
        "output": _tensor_payload(output),
        "tolerance": {"rtol": rtol, "atol": atol},
    }
    path = op_dir / f"{case}.json"
    # default float repr round-trips through Python's json module.
    path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    return path


def cases():
    """Yield (op, case_name, inputs, output) tuples."""

    # --- Elementwise add/sub/mul/div (1D 4-elem and 2D 3x3) -------------------
    a1 = np.array([1.0, 2.0, 3.0, 4.0])
    b1 = np.array([0.5, 1.5, 2.5, 3.5])
    yield "add", "vec4", [a1, b1], a1 + b1
    yield "sub", "vec4", [a1, b1], a1 - b1
    yield "mul", "vec4", [a1, b1], a1 * b1
    yield "div", "vec4", [a1, b1], a1 / b1

    a2 = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
        ]
    )
    b2 = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]
    )
    yield "add", "mat3x3", [a2, b2], a2 + b2
    yield "sub", "mat3x3", [a2, b2], a2 - b2
    yield "mul", "mat3x3", [a2, b2], a2 * b2
    yield "div", "mat3x3", [a2, b2], a2 / b2

    # --- Matmul (2x3 @ 3x4 -> 2x4) -------------------------------------------
    m_a = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    m_b = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    yield "matmul", "2x3_at_3x4", [m_a, m_b], m_a @ m_b

    # --- Reductions on a 1D 5-elem vector -------------------------------------
    v5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    yield "sum", "vec5", [v5], np.sum(v5)
    yield "mean", "vec5", [v5], np.mean(v5)
    # NumPy var/std default to ddof=0 (population). viva_tensor.variance and
    # gleam_community_maths.variance also use ddof=0, so we match by default.
    yield "var", "vec5", [v5], np.var(v5)
    yield "std", "vec5", [v5], np.std(v5)

    # --- Transpose (2D 3x4 -> 4x3) -------------------------------------------
    mt = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ]
    )
    yield "transpose", "mat3x4", [mt], mt.T

    # --- Unary float ops on positive vectors ----------------------------------
    pos5 = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    yield "exp", "vec5", [pos5], np.exp(pos5)
    yield "log", "vec5", [pos5], np.log(pos5)

    # --- ReLU on a 7-elem vector with mixed signs -----------------------------
    relu_in = np.array([-2.0, -0.5, 0.0, 0.5, 1.0, -1.5, 2.5])
    yield "relu", "vec7", [relu_in], np.maximum(relu_in, 0.0)


def main() -> None:
    written: list[Path] = []
    for op, case, inputs, output in cases():
        path = _write_case(op, case, inputs, np.asarray(output))
        written.append(path)
    print(f"Wrote {len(written)} fixtures under {FIXTURES_DIR}")
    for p in written:
        print(f"  {p.relative_to(FIXTURES_DIR)}")


if __name__ == "__main__":
    main()
