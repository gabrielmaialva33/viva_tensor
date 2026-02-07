#!/usr/bin/env python3
"""
VIVA TENSOR vs NumPy/PyTorch - Fair Comparison
Run this on Windows native AND WSL to get fair numbers.
"""
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def benchmark_matmul(lib_name, matmul_fn, sizes, warmup=3, iters=5):
    """Benchmark matrix multiplication."""
    print(f"\n>>> {lib_name} MATMUL")
    print("-" * 60)

    for n in sizes:
        # Create matrices
        if lib_name == "NumPy":
            a = np.random.rand(n, n).astype(np.float64)
            b = np.random.rand(n, n).astype(np.float64)
        else:  # PyTorch
            a = torch.rand(n, n, dtype=torch.float64)
            b = torch.rand(n, n, dtype=torch.float64)

        # Warmup
        for _ in range(warmup):
            c = matmul_fn(a, b)

        # Sync for GPU (if applicable)
        if lib_name == "PyTorch" and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        iterations = max(1, iters * (1000 // n))
        t0 = time.perf_counter()
        for _ in range(iterations):
            c = matmul_fn(a, b)

        if lib_name == "PyTorch" and torch.cuda.is_available():
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        avg_ms = (t1 - t0) * 1000 / iterations
        gflops = 2 * n**3 / (avg_ms * 1e6)

        print(f"  {n}x{n}: {avg_ms:.2f} ms  {gflops:.1f} GFLOPS")

def main():
    print("=" * 60)
    print("  VIVA TENSOR COMPARISON BENCHMARK")
    print("=" * 60)

    import platform
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    # NumPy info
    print(f"\nNumPy: {np.__version__}")
    try:
        config = np.show_config(mode='dicts')
        if config and 'Build Dependencies' in config:
            blas_info = config['Build Dependencies'].get('blas', {})
            print(f"  BLAS: {blas_info.get('name', 'unknown')}")
    except:
        pass

    if HAS_TORCH:
        print(f"PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA: Not available")

    sizes = [100, 200, 500, 1000, 1500, 2000]

    # NumPy benchmark
    benchmark_matmul("NumPy", np.matmul, sizes)

    # PyTorch benchmark (CPU)
    if HAS_TORCH:
        benchmark_matmul("PyTorch (CPU)", torch.matmul, sizes)

        # PyTorch GPU if available
        if torch.cuda.is_available():
            def torch_cuda_matmul(a, b):
                return torch.matmul(a.cuda(), b.cuda())
            benchmark_matmul("PyTorch (CUDA)", torch_cuda_matmul, sizes)

    print("\n" + "=" * 60)
    print("  COMPARISON TARGETS")
    print("=" * 60)
    print("""
  viva_tensor WSL Results (i9-13900K, 16 threads):
    1000x1000: 202.6 GFLOPS
    1500x1500: 268.0 GFLOPS
    2000x2000: 268.1 GFLOPS

  To beat NumPy, we need > {numpy_gflops} GFLOPS at 2000x2000
""")

if __name__ == "__main__":
    main()
