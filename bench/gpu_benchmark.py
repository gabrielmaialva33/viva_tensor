#!/usr/bin/env python3
"""
viva_tensor GPU vs PyTorch GPU Benchmark
=========================================
Compares matmul performance on the SAME RTX 4090 GPU.

Tests: FP32, FP16, INT8 (where available)
"""

import subprocess
import time
import sys
import os
import numpy as np

# ─── Configuration ───
SIZES = [1000, 2000, 4000, 6000, 8000]
WARMUP = 10
RUNS = 30

def pytorch_gpu_benchmark():
    """Benchmark PyTorch GPU matmul across precisions and sizes."""
    import torch

    device = torch.device('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*70}")
    print(f"  PyTorch {torch.__version__} | {gpu_name}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*70}\n")

    results = {}

    for dtype_name, dtype in [('FP32', torch.float32), ('FP16', torch.float16)]:
        print(f"--- PyTorch {dtype_name} ---")
        results[dtype_name] = {}

        for n in SIZES:
            flops = 2.0 * n * n * n
            a = torch.ones(n, n, dtype=dtype, device=device)
            b = torch.ones(n, n, dtype=dtype, device=device)

            # Warmup
            for _ in range(WARMUP):
                torch.mm(a, b)
            torch.cuda.synchronize()

            # Timed runs
            times = []
            for _ in range(RUNS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                torch.mm(a, b)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms

            times = np.array(times)
            gflops = flops / (times * 1e6)

            # Remove outliers (IQR)
            q1, q3 = np.percentile(gflops, [25, 75])
            iqr = q3 - q1
            mask = (gflops >= q1 - 1.5*iqr) & (gflops <= q3 + 1.5*iqr)
            clean = gflops[mask]

            mean_gf = np.mean(clean)
            std_gf = np.std(clean)
            tflops = mean_gf / 1000

            results[dtype_name][n] = {
                'mean': mean_gf, 'std': std_gf,
                'tflops': tflops, 'samples': len(clean)
            }
            print(f"  {n:5d}x{n}: {tflops:8.2f} TFLOPS  (±{std_gf:.0f} GFLOPS, n={len(clean)})")

            del a, b
            torch.cuda.empty_cache()

        print()

    # INT8 via torch.int8 matmul (if available)
    print("--- PyTorch INT8 (torch._int_mm) ---")
    results['INT8'] = {}
    for n in SIZES:
        flops = 2.0 * n * n * n
        try:
            a = torch.ones(n, n, dtype=torch.int8, device=device)
            b = torch.ones(n, n, dtype=torch.int8, device=device)

            # Warmup
            for _ in range(WARMUP):
                torch._int_mm(a, b)
            torch.cuda.synchronize()

            times = []
            for _ in range(RUNS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                torch._int_mm(a, b)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)

            times = np.array(times)
            gflops = flops / (times * 1e6)
            q1, q3 = np.percentile(gflops, [25, 75])
            iqr = q3 - q1
            mask = (gflops >= q1 - 1.5*iqr) & (gflops <= q3 + 1.5*iqr)
            clean = gflops[mask]
            mean_gf = np.mean(clean)
            std_gf = np.std(clean)
            results['INT8'][n] = {'mean': mean_gf, 'std': std_gf, 'tflops': mean_gf/1000, 'samples': len(clean)}
            print(f"  {n:5d}x{n}: {mean_gf/1000:8.2f} TOPS   (±{std_gf:.0f}, n={len(clean)})")
            del a, b
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {n:5d}x{n}: N/A ({e})")
            results['INT8'][n] = None

    print()
    return results


def viva_tensor_gpu_benchmark():
    """Benchmark viva_tensor GPU via Erlang NIF."""
    print(f"\n{'='*70}")
    print(f"  viva_tensor GPU (Erlang NIF → cuBLAS / Tensor Cores)")
    print(f"{'='*70}\n")

    results = {}
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    for dtype_name, matmul_fn, create_fn, avail_fn in [
        ('FP32', 'ct_matmul_inplace', 'ct_from_list', None),
        ('FP16', 'ct16_matmul_inplace', 'ct16_from_list', 'ct16_available'),
        ('INT8', 'ct_int8_matmul_inplace', 'ct_int8_from_list', 'ct_int8_available'),
    ]:
        print(f"--- viva_tensor {dtype_name} ---")
        results[dtype_name] = {}

        for n in SIZES:
            erlang_code = f'''
                N = {n},
                Warmup = {WARMUP},
                Runs = {RUNS},
                Data = [1.0 || _ <- lists:seq(1, N*N)],
                {{ok, A}} = viva_tensor_zig:{create_fn}(Data, [N, N]),
                {{ok, B}} = viva_tensor_zig:{create_fn}(Data, [N, N]),
                {{ok, C}} = viva_tensor_zig:{create_fn}(Data, [N, N]),
                [viva_tensor_zig:{matmul_fn}(A, B, C, N, N, N) || _ <- lists:seq(1, Warmup)],
                Times = [begin
                    {{T, _}} = timer:tc(fun() ->
                        ok = viva_tensor_zig:{matmul_fn}(A, B, C, N, N, N)
                    end),
                    T / 1000.0
                end || _ <- lists:seq(1, Runs)],
                Gflops = [(2.0 * N * N * N) / (T * 1000000) || T <- Times],
                io:format("~w~n", [Gflops]),
                halt().
            '''

            cmd = ['erl', '-pa'] + \
                  [os.path.join(project_dir, 'build/dev/erlang', d, 'ebin')
                   for d in os.listdir(os.path.join(project_dir, 'build/dev/erlang'))
                   if os.path.isdir(os.path.join(project_dir, 'build/dev/erlang', d, 'ebin'))] + \
                  ['-noshell', '-eval', erlang_code]

            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                                       cwd=project_dir)
                output = result.stdout + result.stderr

                # Find the data line (list of floats)
                data_line = None
                for line in output.split('\n'):
                    line = line.strip()
                    if line.startswith('[') and ']' in line and 'NIF' not in line and 'viva_tensor' not in line.lower().replace('viva_tensor_zig:', ''):
                        data_line = line
                        break

                if not data_line:
                    # Try finding any line with numbers in brackets
                    for line in output.split('\n'):
                        line = line.strip()
                        if line.startswith('[') and line.endswith(']') and '.' in line:
                            data_line = line
                            break

                if data_line:
                    gflops = np.array(eval(data_line))
                    # IQR outlier removal
                    q1, q3 = np.percentile(gflops, [25, 75])
                    iqr = q3 - q1
                    mask = (gflops >= q1 - 1.5*iqr) & (gflops <= q3 + 1.5*iqr)
                    clean = gflops[mask]
                    mean_gf = np.mean(clean)
                    std_gf = np.std(clean)
                    unit = "TOPS" if dtype_name == "INT8" else "TFLOPS"
                    results[dtype_name][n] = {
                        'mean': mean_gf, 'std': std_gf,
                        'tflops': mean_gf/1000, 'samples': len(clean)
                    }
                    print(f"  {n:5d}x{n}: {mean_gf/1000:8.2f} {unit}  (±{std_gf:.0f} GFLOPS, n={len(clean)})")
                else:
                    print(f"  {n:5d}x{n}: PARSE ERROR")
                    results[dtype_name][n] = None

            except subprocess.TimeoutExpired:
                print(f"  {n:5d}x{n}: TIMEOUT")
                results[dtype_name][n] = None
            except Exception as e:
                print(f"  {n:5d}x{n}: ERROR ({e})")
                results[dtype_name][n] = None

        print()

    return results


def print_comparison(pt_results, vt_results):
    """Print side-by-side comparison table."""
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON: viva_tensor GPU vs PyTorch GPU (RTX 4090)")
    print(f"{'='*70}\n")

    for dtype in ['FP32', 'FP16', 'INT8']:
        unit = "TOPS" if dtype == "INT8" else "TFLOPS"
        print(f"┌{'─'*68}┐")
        print(f"│  {dtype} Matmul ({unit}){' '*(50-len(dtype)-len(unit))}│")
        print(f"├{'─'*10}┬{'─'*18}┬{'─'*18}┬{'─'*9}┬{'─'*9}┤")
        print(f"│ {'Size':>8s} │ {'viva_tensor':>16s} │ {'PyTorch':>16s} │ {'Delta':>7s} │ {'Winner':>7s} │")
        print(f"├{'─'*10}┼{'─'*18}┼{'─'*18}┼{'─'*9}┼{'─'*9}┤")

        for n in SIZES:
            vt = vt_results.get(dtype, {}).get(n)
            pt = pt_results.get(dtype, {}).get(n)

            if vt and pt:
                vt_tf = vt['tflops']
                pt_tf = pt['tflops']
                delta = ((vt_tf - pt_tf) / pt_tf) * 100
                winner = "viva" if vt_tf > pt_tf else "torch"
                sign = "+" if delta > 0 else ""
                print(f"│ {n:>5d}x{n:<2d} │ {vt_tf:>12.2f} {unit:>3s} │ {pt_tf:>12.2f} {unit:>3s} │ {sign}{delta:>5.1f}% │ {winner:>7s} │")
            elif vt:
                print(f"│ {n:>5d}x{n:<2d} │ {vt['tflops']:>12.2f} {unit:>3s} │ {'N/A':>16s} │ {'':>7s} │ {'':>7s} │")
            elif pt:
                print(f"│ {n:>5d}x{n:<2d} │ {'N/A':>16s} │ {pt['tflops']:>12.2f} {unit:>3s} │ {'':>7s} │ {'':>7s} │")
            else:
                print(f"│ {n:>5d}x{n:<2d} │ {'N/A':>16s} │ {'N/A':>16s} │ {'':>7s} │ {'':>7s} │")

        print(f"└{'─'*10}┴{'─'*18}┴{'─'*18}┴{'─'*9}┴{'─'*9}┘")
        print()


if __name__ == '__main__':
    print("╔═══════════════════════════════════════════════════════════════════════╗")
    print("║     viva_tensor GPU  vs  PyTorch GPU  —  RTX 4090 Showdown          ║")
    print("╠═══════════════════════════════════════════════════════════════════════╣")
    print(f"║  Sizes: {SIZES}                                  ║")
    print(f"║  Warmup: {WARMUP}, Runs: {RUNS}, Outlier removal: IQR×1.5              ║")
    print("╚═══════════════════════════════════════════════════════════════════════╝")

    pt_results = pytorch_gpu_benchmark()
    vt_results = viva_tensor_gpu_benchmark()
    print_comparison(pt_results, vt_results)
