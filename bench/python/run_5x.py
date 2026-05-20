#!/usr/bin/env python3
"""Roda CPU benchmark.py + GPU gpu_benchmark.py N vezes e agrega mean/median/mode."""

import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import numpy as np

N_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
ROOT = Path(__file__).resolve().parents[2]

CPU_SIZES = [1000, 2000, 3000]
CPU_TIMED = 15
CPU_WARMUP = 4

GPU_SIZES = [1000, 2000, 4000, 6000, 8000]

# ---------- CPU ----------
def parse_cpu_stdout(text: str):
    """Extract (library, size, gflops) from CPU benchmark stdout."""
    out = {}
    lib = None
    header_re = re.compile(r"^\s+(NumPy|PyTorch|viva_tensor)\s+(\d+)×\d+\.\.\.\s+([\d.]+)")
    for line in text.splitlines():
        m = header_re.match(line)
        if m:
            l, sz, gf = m.group(1).lower().replace("numpy", "numpy").replace("pytorch", "pytorch"), int(m.group(2)), float(m.group(3))
            out[(l, sz)] = gf
    return out


def run_cpu_once() -> dict:
    cmd = [sys.executable, str(ROOT / "bench/python/benchmark.py"),
           "--sizes", *map(str, CPU_SIZES),
           "--runs", str(CPU_TIMED), "--warmup", str(CPU_WARMUP)]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1800)
    return parse_cpu_stdout(r.stdout)


# ---------- GPU ----------
def parse_gpu_stdout(text: str):
    """Extract (library, dtype, size, tflops) from gpu_benchmark.py stdout."""
    out = {}
    current_lib = None
    current_dtype = None

    pt_header = re.compile(r"^---\s+PyTorch\s+(FP\d+|INT8)")
    vt_header = re.compile(r"^---\s+viva_tensor\s+(FP\d+|INT8)")
    row_re = re.compile(r"^\s+(\d+)x\d+:\s+([\d.]+)\s+(TFLOPS|TOPS)")

    for line in text.splitlines():
        m = pt_header.match(line)
        if m:
            current_lib = "pytorch"
            current_dtype = m.group(1)
            continue
        m = vt_header.match(line)
        if m:
            current_lib = "viva_tensor"
            current_dtype = m.group(1)
            continue
        m = row_re.match(line)
        if m and current_lib:
            sz = int(m.group(1))
            tflops = float(m.group(2))
            out[(current_lib, current_dtype, sz)] = tflops
    return out


def run_gpu_once() -> dict:
    cmd = [sys.executable, str(ROOT / "bench/python/gpu_benchmark.py")]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, timeout=1800)
    return parse_gpu_stdout(r.stdout)


# ---------- Aggregation ----------
def mode_binned(vals, bin_width):
    """Mode based on rounding values to nearest bin; returns center of most common bin."""
    if not vals:
        return float("nan")
    binned = [round(v / bin_width) * bin_width for v in vals]
    # Most common
    from collections import Counter
    most_common, count = Counter(binned).most_common(1)[0]
    return most_common


def aggregate(runs_data, bin_width):
    """runs_data: list of dicts {key: value}. Returns {key: (mean, median, mode, vals)}."""
    keys = set()
    for d in runs_data:
        keys.update(d.keys())
    agg = {}
    for k in sorted(keys):
        vals = [d[k] for d in runs_data if k in d]
        if vals:
            agg[k] = (mean(vals), median(vals), mode_binned(vals, bin_width), vals)
    return agg


# ---------- Run ----------
def main():
    print(f"=== Rodando {N_RUNS}x CPU + {N_RUNS}x GPU ===\n")

    cpu_runs = []
    for i in range(N_RUNS):
        print(f"[CPU run {i+1}/{N_RUNS}] ...", flush=True)
        t0 = time.time()
        cpu_runs.append(run_cpu_once())
        print(f"  → {time.time()-t0:.1f}s, parsed {len(cpu_runs[-1])} entries")

    gpu_runs = []
    for i in range(N_RUNS):
        print(f"[GPU run {i+1}/{N_RUNS}] ...", flush=True)
        t0 = time.time()
        gpu_runs.append(run_gpu_once())
        print(f"  → {time.time()-t0:.1f}s, parsed {len(gpu_runs[-1])} entries")

    cpu_agg = aggregate(cpu_runs, bin_width=10.0)   # 10 GFLOPS bins
    gpu_agg = aggregate(gpu_runs, bin_width=1.0)    # 1 TFLOPS bins

    print("\n" + "=" * 78)
    print(f"  CPU MATMUL — agregado de {N_RUNS} runs (f64, GFLOPS)")
    print("=" * 78)
    print(f"{'Library':<14} {'Size':>6}  {'Mean':>8} {'Median':>8} {'Mode':>8}  {'Runs':<40}")
    for (lib, sz), (m, md, mo, vals) in cpu_agg.items():
        vals_str = ", ".join(f"{v:.1f}" for v in vals)
        print(f"{lib:<14} {sz:>6}  {m:>8.1f} {md:>8.1f} {mo:>8.1f}  [{vals_str}]")

    print("\n" + "=" * 78)
    print(f"  GPU MATMUL — agregado de {N_RUNS} runs (TFLOPS / TOPS)")
    print("=" * 78)
    print(f"{'Library':<14} {'DType':<6} {'Size':>6}  {'Mean':>8} {'Median':>8} {'Mode':>8}  {'Runs':<40}")
    for (lib, dt, sz), (m, md, mo, vals) in gpu_agg.items():
        vals_str = ", ".join(f"{v:.1f}" for v in vals)
        print(f"{lib:<14} {dt:<6} {sz:>6}  {m:>8.2f} {md:>8.2f} {mo:>8.2f}  [{vals_str}]")

    # Save JSON
    import json
    from datetime import datetime
    out_dir = ROOT / "bench/data"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    payload = {
        "n_runs": N_RUNS,
        "timestamp": ts,
        "cpu_runs": [{f"{k[0]}|{k[1]}": v for k, v in d.items()} for d in cpu_runs],
        "gpu_runs": [{f"{k[0]}|{k[1]}|{k[2]}": v for k, v in d.items()} for d in gpu_runs],
        "cpu_aggregate": {f"{k[0]}|{k[1]}": {"mean": v[0], "median": v[1], "mode": v[2], "vals": v[3]}
                          for k, v in cpu_agg.items()},
        "gpu_aggregate": {f"{k[0]}|{k[1]}|{k[2]}": {"mean": v[0], "median": v[1], "mode": v[2], "vals": v[3]}
                          for k, v in gpu_agg.items()},
    }
    path = out_dir / f"run_5x_{ts}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
