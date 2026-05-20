"""
NumPy + PyTorch matmul reference benchmarks (CPU + GPU).

Designed to match viva_tensor/bench/peak.gleam:
- Same sizes (2048, 4096)
- Same iteration count (30)
- Kernel-only timing via cuda events for GPU
- Warm-up iteration excluded from measurement

Outputs a TSV-style table to stdout that's easy to splice next to the
viva_tensor numbers. Run with the bench venv:

  .bench-venv/bin/python3 bench/python/compare/numpy_pytorch.py
"""

import time

import numpy as np
import torch

ITERS = 30
SIZES = (2048, 4096)


def fmt(name, n, tflops, ms_per_iter):
    print(f"  {name:<44s} {tflops:7.1f} TFLOPS  ({ms_per_iter:.2f} ms/iter)")


def bench_numpy(n: int, dtype: np.dtype) -> tuple[float, float]:
    a = np.random.randn(n, n).astype(dtype)
    b = np.random.randn(n, n).astype(dtype)
    _ = a @ b  # warmup
    start = time.perf_counter()
    for _ in range(ITERS):
        _ = a @ b
    elapsed = time.perf_counter() - start
    flops = 2.0 * n * n * n * ITERS
    return flops / elapsed / 1e12, elapsed / ITERS * 1000.0


def bench_torch_cpu(n: int, dtype: torch.dtype) -> tuple[float, float]:
    a = torch.randn(n, n, dtype=dtype, device="cpu")
    b = torch.randn(n, n, dtype=dtype, device="cpu")
    _ = a @ b
    start = time.perf_counter()
    for _ in range(ITERS):
        _ = a @ b
    elapsed = time.perf_counter() - start
    flops = 2.0 * n * n * n * ITERS
    return flops / elapsed / 1e12, elapsed / ITERS * 1000.0


def bench_torch_cuda(n: int, dtype: torch.dtype) -> tuple[float, float]:
    a = torch.randn(n, n, dtype=dtype, device="cuda")
    b = torch.randn(n, n, dtype=dtype, device="cuda")
    _ = a @ b
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(ITERS):
        _ = a @ b
    stop_ev.record()
    torch.cuda.synchronize()
    elapsed_ms = start_ev.elapsed_time(stop_ev)
    flops = 2.0 * n * n * n * ITERS
    return flops / (elapsed_ms / 1000.0) / 1e12, elapsed_ms / ITERS


def bench_torch_cuda_fp8(n: int) -> tuple[float, float]:
    """FP8 E4M3 via PyTorch's scaled_mm (Ada/Hopper)."""
    a = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(n, n, device="cuda").to(torch.float8_e4m3fn).t().contiguous().t()
    scale = torch.tensor(1.0, device="cuda")
    _ = torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.float16)
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    stop_ev = torch.cuda.Event(enable_timing=True)
    start_ev.record()
    for _ in range(ITERS):
        _ = torch._scaled_mm(a, b, scale_a=scale, scale_b=scale, out_dtype=torch.float16)
    stop_ev.record()
    torch.cuda.synchronize()
    elapsed_ms = start_ev.elapsed_time(stop_ev)
    flops = 2.0 * n * n * n * ITERS
    return flops / (elapsed_ms / 1000.0) / 1e12, elapsed_ms / ITERS


def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║         NumPy + PyTorch reference matmul benchmark               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"\nPyTorch: {torch.__version__}  NumPy: {np.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}  CUDA: {torch.version.cuda}")
    print(f"CPU threads (torch): {torch.get_num_threads()}")
    print()

    results = []  # (path, n, tflops, ms_per_iter)

    for n in SIZES:
        print(f"── {n}×{n} (iters={ITERS}) ──")

        tflops, ms = bench_numpy(n, np.float32)
        fmt("NumPy CPU FP32 (BLAS auto-detect)", n, tflops, ms)
        results.append(("numpy_cpu_fp32", n, tflops, ms))

        tflops, ms = bench_torch_cpu(n, torch.float32)
        fmt("PyTorch CPU FP32 (oneDNN/MKL)", n, tflops, ms)
        results.append(("torch_cpu_fp32", n, tflops, ms))

        tflops, ms = bench_torch_cuda(n, torch.float32)
        fmt("PyTorch GPU FP32 (cuBLAS)", n, tflops, ms)
        results.append(("torch_gpu_fp32", n, tflops, ms))

        tflops, ms = bench_torch_cuda(n, torch.float16)
        fmt("PyTorch GPU FP16 (cuBLAS Tensor Core)", n, tflops, ms)
        results.append(("torch_gpu_fp16", n, tflops, ms))

        tflops, ms = bench_torch_cuda(n, torch.bfloat16)
        fmt("PyTorch GPU BF16 (cuBLAS Tensor Core)", n, tflops, ms)
        results.append(("torch_gpu_bf16", n, tflops, ms))

        try:
            tflops, ms = bench_torch_cuda_fp8(n)
            fmt("PyTorch GPU FP8 E4M3 (_scaled_mm)", n, tflops, ms)
            results.append(("torch_gpu_fp8", n, tflops, ms))
        except Exception as exc:
            print(f"  PyTorch GPU FP8 E4M3                          skipped: {exc}")

        print()

    # Dump as Erlang term file for the Gleam showdown to consume.
    out_path = "bench/results/pytorch_results.term"
    with open(out_path, "w") as f:
        for path, n, tflops, ms in results:
            f.write(f'{{"{path}", {n}, {tflops:.2f}, {ms:.4f}}}.\n')
    print(f"✓ wrote {len(results)} entries to {out_path}")


if __name__ == "__main__":
    main()
